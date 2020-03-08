#include "jit.h"
#include "log.h"
#include "ssa.h"
#include "eval.h"

// ====================================================================
//  The following data structures are temporarily used during program
//  generation. They are declared as global variables to enable memory
//  reuse across jit_eval() calls.
// ====================================================================

/// Ordered unique list containing sizes of variables to be computed
static std::vector<uint32_t> schedule_sizes;

/// Map variable size => ordered list of variables that should be computed
static tsl::robin_map<uint32_t, std::vector<uint32_t>> schedule;

/// Auxiliary data structure needed to compute 'schedule_sizes' and 'schedule'
static tsl::robin_set<std::pair<uint32_t, uint32_t>, pair_hash> visited;

/// Maps between Enoki variable indices and program variables
static tsl::robin_map<uint32_t, uint32_t> reg_map;

// ====================================================================

static const char *cuda_register_type(uint32_t type) {
    switch (type) {
        case EnokiType::UInt8:    return "u8";
        case EnokiType::Int8:     return "s8";
        case EnokiType::UInt16:   return "u16";
        case EnokiType::Int16:    return "s16";
        case EnokiType::UInt32:   return "u32";
        case EnokiType::Int32:    return "s32";
        case EnokiType::Pointer:
        case EnokiType::UInt64:   return "u64";
        case EnokiType::Int64:    return "s64";
        case EnokiType::Float16:  return "f16";
        case EnokiType::Float32:  return "f32";
        case EnokiType::Float64:  return "f64";
        case EnokiType::Bool:     return "pred";
        default: jit_fail("cuda_register_type(): invalid type!");
    }
}

static const char *cuda_register_type_bin(uint32_t type) {
    switch (type) {
        case EnokiType::UInt8:
        case EnokiType::Int8:    return "b8";
        case EnokiType::UInt16:
        case EnokiType::Float16:
        case EnokiType::Int16:   return "b16";
        case EnokiType::Float32:
        case EnokiType::UInt32:
        case EnokiType::Int32:   return "b32";
        case EnokiType::Pointer:
        case EnokiType::Float64:
        case EnokiType::UInt64:
        case EnokiType::Int64:   return "b64";
        case EnokiType::Bool:    return "pred";
        default: jit_fail("cuda_register_type_bin(): invalid type!");
    }
}

static const char *cuda_register_name(uint32_t type) {
    switch (type) {
        case EnokiType::UInt8:
        case EnokiType::Int8:    return "%b";
        case EnokiType::UInt16:
        case EnokiType::Int16:   return "%w";
        case EnokiType::UInt32:
        case EnokiType::Int32:   return "%r";
        case EnokiType::Pointer:
        case EnokiType::UInt64:
        case EnokiType::Int64:   return "%rd";
        case EnokiType::Float32: return "%f";
        case EnokiType::Float64: return "%d";
        case EnokiType::Bool:    return "%p";
        default: jit_fail("cuda_register_name(): invalid type!");
    }
}

// ====================================================================

/// Recursively traverse the computation graph to find variables needed by a computation
static void jit_var_traverse(uint32_t size, uint32_t idx) {
    std::pair<uint32_t, uint32_t> key(size, idx);

    if (idx == 0 || visited.find(key) != visited.end())
        return;

    visited.insert(key);

    Variable *v = jit_var(idx);
    const uint32_t *dep = v->dep;

    std::pair<uint32_t, uint32_t> ch[3] = {
        { dep[0], dep[0] ? jit_var(dep[0])->tsize : 0 },
        { dep[1], dep[1] ? jit_var(dep[1])->tsize : 0 },
        { dep[2], dep[2] ? jit_var(dep[2])->tsize : 0 }
    };

    // Simple sorting network
    if (ch[1].second < ch[2].second)
        std::swap(ch[1], ch[2]);
    if (ch[0].second < ch[2].second)
        std::swap(ch[0], ch[2]);
    if (ch[0].second < ch[1].second)
        std::swap(ch[0], ch[1]);

    for (auto [id, tsize] : ch)
        jit_var_traverse(size, id);

    schedule[size].push_back(idx);
}

void jit_assemble(uint32_t size) {
    const std::vector<uint32_t> &sched = schedule[size];

    uint32_t n_vars_in = 0, n_vars_out = 0, n_vars_total = 0;

    jit_log(Trace, "jit_assemble(size=%u): register map:", size);

    for (uint32_t index : sched) {
        const Variable *v = jit_var(index);

        if (v->data || v->direct_pointer) {
            n_vars_in++;
        } else {
            if (!v->side_effect &&
                 v->ref_count_ext > 0 &&
                 v->size == size)
            n_vars_out++;
        }

        if (state.log_level >= 4) {
            buffer.clear();
            buffer.fmt("   - %s%u -> %u",
                       cuda_register_name(v->type),
                       n_vars_total, index);

            if (v->label)
                buffer.fmt(" \"%s\"", v->label);
            if (v->size == 1)
                buffer.put(" [scalar]");
            if (v->data != nullptr || v->direct_pointer)
                buffer.put(" [in]");
            else if (v->side_effect)
                buffer.put(" [se]");
            else if (v->ref_count_ext > 0 && v->size == size)
                buffer.put(" [out]");

            jit_log(Trace, "%s", buffer.get());
        }

        reg_map[index] = n_vars_total++;
    }

    buffer.clear();
    buffer.put(".version 6.3\n");
    buffer.put(".target sm_61\n");
    buffer.put(".address_size 64\n");

}

/// Evaluate all computation that is queued on the current device & stream
void jit_eval() {
    Stream *stream = active_stream;
    if (unlikely(!stream))
        jit_fail("jit_var_copy_to_device(): device and stream must be set! "
                 "(call jit_device_set() beforehand)!");

    if (stream->todo.empty())
        return;

    visited.clear();
    schedule.clear();
    schedule_sizes.clear();

    for (uint32_t idx : stream->todo) {
        size_t size = jit_var_size(idx);
        jit_var_traverse(size, idx);
        schedule_sizes.push_back(size);
    }

    stream->todo.clear();

    std::sort(schedule_sizes.begin(), schedule_sizes.end(),
              std::greater<uint32_t>());

    bool parallel_dispatch = state.parallel_dispatch && schedule.size() > 1;

    if (!parallel_dispatch) {
        jit_log(Debug, "jit_eval(): begin.");
    } else {
        jit_log(Debug, "jit_eval(): begin (parallel dispatch to %zu streams).",
                schedule.size());
        cuda_check(cudaEventRecord(stream->event, stream->handle));
    }

    uint32_t stream_idx = 1000 * stream->stream;
    for (uint32_t size : schedule_sizes) {
        jit_assemble(size);

        Stream *sub_stream = stream;
        if (parallel_dispatch) {
            jit_device_set(stream->device, stream_idx);
            sub_stream = active_stream;
            cuda_check(cudaStreamWaitEvent(sub_stream->handle, stream->event, 0));
        }

        // jit_run(std::move(std::get<0>(result)),
        //         std::get<1>(result),
        //         size, stream_idx, start, mid);

        if (parallel_dispatch) {
            cuda_check(cudaEventRecord(sub_stream->event, sub_stream->handle));
            cuda_check(cudaStreamWaitEvent(stream->handle, sub_stream->event, 0));
        }

        stream_idx++;
    }

    jit_device_set(stream->device, stream->stream);

    /**
     * At this point, all variables and their dependencies are computed, which
     * means that we can remove internal edges between them. This in turn will
     * cause many of the variables to be garbage-collected.
     */
    jit_log(Debug, "jit_eval(): cleaning up..");

    for (uint32_t size : schedule_sizes) {
        const std::vector<uint32_t> &sched = schedule[size];

        for (uint32_t idx : sched) {
            auto it = state.variables.find(idx);
            if (it == state.variables.end())
                continue;

            Variable *v = &it.value();
            bool side_effect = v->side_effect;
            v->side_effect = false;
            v->dirty = false;

            if (v->data != nullptr && v->cmd != nullptr) {
                uint32_t dep[3], extra_dep = v->extra_dep;
                memcpy(dep, v->dep, sizeof(uint32_t) * 3);
                memset(v->dep, 0, sizeof(uint32_t) * 3);
                v->extra_dep = 0;
                for (int j = 0; j < 3; ++j)
                    jit_dec_ref_int(dep[j]);
                jit_dec_ref_ext(extra_dep);
            }

            if (side_effect)
                jit_dec_ref_ext(idx);
        }
    }

    jit_free_flush();
    jit_log(Debug, "jit_eval(): done.");
}
