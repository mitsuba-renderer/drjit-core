#include "record_ts.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "eval.h"
#include "internal.h"
#include "log.h"
#include "var.h"

static bool dry_run = false;

// This struct holds the data and tracks the size of varaibles,
// used during replay.
struct ReplayVariable {
    void *data = nullptr;
    // Tracks the capacity, this allocation has been allocated for
    size_t alloc_size = 0;
    // Tracks the size in bytes, of this allocation
    size_t data_size = 0;
    uint32_t index;
    RecordType rv_type;

    ReplayVariable(RecordVariable &rv) {
        this->index = rv.index;

        this->rv_type = rv.rv_type;

        if (rv_type == RecordType::Captured) {
            Variable *v = jitc_var(this->index);
            this->data = v->data;
            this->alloc_size = v->size * type_size[v->type];
            this->data_size = this->alloc_size;
        }
    }

    void init_from_input(Variable *input_variable){
        this->data = input_variable->data;
        this->alloc_size = type_size[input_variable->type] * input_variable->size;
        this->data_size = this->alloc_size;
    }


    /**
     * Calculate the number of elements given some variable type.
     */
    uint32_t size(VarType vtype){
        return size(type_size[(uint32_t)vtype]);
    }
    /**
     * Calculate the number of elements given some type size.
     */
    uint32_t size(uint32_t tsize){
        if (tsize == 0)
            jitc_fail("replay(): Invalid var type!");
        size_t size = (this->data_size / (size_t)tsize);

        if (size == 0)
            jitc_fail("replay(): Error, determining size of variable! rv_type "
                      "%u, dsize=%zu",
                      (uint32_t) this->rv_type, this->data_size);

        if(size * (size_t)tsize != this->data_size)
            jitc_fail("replay(): Error, determining size of variable!");

        return (uint32_t)size;
    }

    void alloc(JitBackend backend, uint32_t size, VarType type) {
        alloc(backend, size, type_size[(uint32_t) type]);
    }
    void alloc(JitBackend backend, uint32_t size, uint32_t isize){
        size_t dsize = ((size_t)size) * ((size_t)isize);
        return alloc(backend, dsize);
    }
    /**
     * Allocates the data for this replay variable.
     * If this is atempted twice, we test weather the allocated size is
     * sufficient and re-allocate the memory if necesarry.
     */
    void alloc(JitBackend backend, size_t dsize) {
        AllocType alloc_type =
            backend == JitBackend::CUDA ? AllocType::Device : AllocType::Host;

        if (!data) {
            this->alloc_size = dsize;
            jitc_log(LogLevel::Debug, "    allocating output of size %zu.",
                     dsize);
            if (!dry_run)
                data = jitc_malloc(alloc_type, dsize);
        } else if (this->alloc_size < dsize) {
            this->alloc_size = dsize;
            jitc_log(LogLevel::Debug, "    re-allocating output of size %zu.",
                     dsize);
            if (!dry_run) {
                jitc_free(this->data);
                data = jitc_malloc(alloc_type, dsize);
            }
        } else {
            // Do not reallocate if the size is enough
        }

        this->data_size = dsize;
    }

    void free(){
        jitc_free(this->data);
        this->data = nullptr;
        this->data_size = 0;
        this->alloc_size = 0;
    }
};

/// Kernel parameter buffer
static std::vector<void *> kernel_params;

/// Temporary variables used for replaying a recording.
static std::vector<ReplayVariable> replay_variables;

int Recording::replay(const uint32_t *replay_inputs, uint32_t *outputs) {

    uint32_t n_kernels = 0;

    this->validate();

    ThreadState *ts = thread_state(backend);
    jitc_assert(dynamic_cast<RecordThreadState *>(ts) == nullptr,
                "replay(): Tried to replay while recording!");

    OptixShaderBindingTable *tmp_sbt = ts->optix_sbt;
    scoped_set_context_maybe guard2(ts->context);

    replay_variables.clear();

    replay_variables.reserve(this->record_variables.size());
    for (RecordVariable &rv : this->record_variables) {
        replay_variables.push_back(ReplayVariable(rv));
    }

    jitc_log(LogLevel::Info, "replay(): inputs");
    // Populate with input variables
    for (uint32_t i = 0; i < this->inputs.size(); ++i) {
        Variable *input_variable = jitc_var(replay_inputs[i]);
        replay_variables[this->inputs[i]].init_from_input(input_variable);
        jitc_log(LogLevel::Debug, "    input %u: r%u mapped to slot(%u)", i,
                 replay_inputs[i], this->inputs[i]);
    }

    // Execute kernels and allocate missing output variables

    for (uint32_t i = 0; i < this->operations.size(); ++i) {
        Operation &op = this->operations[i];
        if (!op.enabled)
            continue;

        switch (op.type) {
        case OpType::KernelLaunch: {
            jitc_log(LogLevel::Debug, "replay(): launching kernel %u ",
                     n_kernels++);
            kernel_params.clear();

            if (backend == JitBackend::CUDA) {
                // First parameter contains kernel size.
                // Assigned later.
                kernel_params.push_back(nullptr);
            } else {
                // First 3 parameters reserved for: kernel ptr, size, ITT
                // identifier
                for (int i = 0; i < 3; ++i)
                    kernel_params.push_back(nullptr);
            }

            // Inferr launch size.

            jitc_log(LogLevel::Debug, "replay(): inferring kernel launch size");

            // Size of direct input variables
            uint32_t input_size = 0;
            // Size of variables referenced by pointers
            uint32_t ptr_size = 0;

            for (uint32_t j = op.dependency_range.first;
                 j < op.dependency_range.second; ++j) {
                ParamInfo info = this->dependencies[j];
                ReplayVariable &rv = replay_variables[info.slot];

                if (info.type == ParamType::Input) {
                    jitc_log(LogLevel::Debug, "infering size of s%u", info.slot);
                    uint32_t size = rv.size(info.vtype);
                    jitc_log(LogLevel::Debug, "    size=%u", size);

                    if(rv.data == nullptr && !dry_run)
                        jitc_fail("replay(): Kernel input variable (slot=%u) "
                                  "not allocated!",
                                  info.slot);

                    if (!info.pointer_access)
                        input_size = std::max(input_size, size);
                    else
                        ptr_size = std::max(ptr_size, size);
                }
            }

            uint32_t launch_size = 0;
            if (op.input_size > 0) {
                launch_size = input_size != 0 ? input_size : ptr_size;
                // Apply the factor
                if (op.size > op.input_size && (op.size % op.input_size == 0)) {
                    if (op.size % op.input_size != 0)
                        jitc_raise(
                            "replay(): Could not infer launch size, using "
                            "heuristic!");
                    size_t ratio = op.size / op.input_size;
                    jitc_log(LogLevel::Debug,
                             "replay(): Inferring launch size by heuristic, "
                             "launch_size=%u, ratio=%zu",
                             launch_size, ratio);
                    launch_size = launch_size * ratio;
                } else if (op.input_size % op.size == 0) {
                    if (op.input_size % op.size != 0)
                        jitc_raise(
                            "replay(): Could not infer launch size, using "
                            "heuristic!");

                    uint32_t fraction = op.input_size / op.size;
                    jitc_log(LogLevel::Debug,
                             "replay(): Inferring launch size by heuristic, "
                             "launch_size(%u), fraction=%u",
                             launch_size, fraction);
                    launch_size = launch_size / fraction;
                }
            }
            if (launch_size == 0) {
                jitc_log(LogLevel::Debug, "replay(): Could not infer launch "
                                         "size, using recorded size");
                launch_size = op.size;
            }

            // Allocate Missing variables for kernel launch.
            // The assumption here is that for every kernel launch, the inputs
            // are already allocated. Therefore we only allocate output
            // variables, which have the same size as the kernel.
            for (uint32_t j = op.dependency_range.first;
                 j < op.dependency_range.second; ++j) {
                ParamInfo info = this->dependencies[j];
                ReplayVariable &rv = replay_variables[info.slot];

                if (info.type == ParamType::Input) {
                    uint32_t size = rv.size(info.vtype);
                    jitc_log(LogLevel::Info,
                             " -> param slot(%u, is_pointer=%u, size=%u)",
                             info.slot, info.pointer_access, size);
                    if(rv.rv_type == RecordType::Captured){
                        jitc_log(LogLevel::Debug, "    label=%s",
                                 jitc_var_label(rv.index));
                        jitc_log(LogLevel::Debug, "    data=%s",
                                 jitc_var_str(rv.index));
                    }
                } else {
                    jitc_log(LogLevel::Info,
                             " <- param slot(%u, is_pointer=%u)", info.slot,
                             info.pointer_access);
                }

                if (info.type == ParamType::Output) {
                    rv.alloc(backend, launch_size, info.vtype);
                }
                jitc_log(LogLevel::Info, "    data=%p", rv.data);
                jitc_assert(
                    rv.data != nullptr || dry_run,
                    "replay(): Encountered nullptr in kernel parameters.");
                kernel_params.push_back(rv.data);
            }

            // Change kernel size in `kernel_params`
            if (backend == JitBackend::CUDA) {
                uintptr_t size = 0;
                std::memcpy(&size, &launch_size, sizeof(uint32_t));
                kernel_params[0] = (void *)size;
            }

            if (!dry_run) {
                jitc_log(LogLevel::Info, "    launch_size=%u, uses_optix=%u",
                         launch_size, op.uses_optix);
                std::vector<uint32_t> kernel_param_ids;
                std::vector<uint32_t> kernel_calls;
                Kernel kernel = op.kernel;
                if (op.uses_optix) {
                    uses_optix = true;
                    ts->optix_sbt = op.sbt;
                }
                ts->launch(kernel, launch_size, &kernel_params,
                           &kernel_param_ids, &kernel_calls);
                if (op.uses_optix)
                    uses_optix = false;
            }

        }

        break;
        case OpType::Barrier:
            if (!dry_run)
                ts->barrier();
            break;
        case OpType::MemsetAsync: {
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo ptr_info = this->dependencies[dependency_index];

            ReplayVariable &ptr_var = replay_variables[ptr_info.slot];
            ptr_var.alloc(backend, op.size, op.input_size);

            jitc_log(LogLevel::Debug, "replay(): MemsetAsync -> slot(%u) [%zu], op.input_size=%zu",
                     ptr_info.slot, op.size, op.input_size);

            uint32_t size = ptr_var.size(op.input_size);

            if (!dry_run)
                ts->memset_async(ptr_var.data, size, op.input_size,
                                 &op.data);
        } break;
        case OpType::Reduce: {
            jitc_log(LogLevel::Debug, "replay(): Reduce");
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo ptr_info = this->dependencies[dependency_index];
            ParamInfo out_info = this->dependencies[dependency_index + 1];

            ReplayVariable &ptr_var = replay_variables[ptr_info.slot];
            ReplayVariable &out_var = replay_variables[out_info.slot];

            out_var.alloc(backend, 1, out_info.vtype);

            VarType type = ptr_info.vtype;
            ReduceOp rtype = op.rtype;

            jitc_log(LogLevel::Debug, " -> slot(%u, data=%p)", ptr_info.slot,
                     ptr_var.data);
            jitc_log(LogLevel::Debug, " <- slot(%u, data=%p)", out_info.slot,
                     out_var.data);

            uint32_t size = ptr_var.size(type);

            if (!dry_run)
                ts->reduce(type, rtype, ptr_var.data, size, out_var.data);
                
        } break;
        case OpType::ReduceExpanded: {
            jitc_log(LogLevel::Debug, "replay(): ReduceExpand");

            uint32_t dependency_index = op.dependency_range.first;
            ParamInfo data_info = this->dependencies[dependency_index];

            ReplayVariable &data_var = replay_variables[data_info.slot];

            VarType vt = data_info.vtype;
            ReduceOp rop = op.rtype;
            uint32_t size = data_var.size(vt);
            uint32_t tsize = type_size[(uint32_t)vt];
            uint32_t workers = pool_size() + 1;

            uint32_t replication_per_worker = size == 1u ? (64u / tsize) : 1u;

            if (!dry_run)
                ts->reduce_expanded(vt, rop, data_var.data,
                                    replication_per_worker * workers, size);

        } break;
        case OpType::Expand: {
            jitc_log(LogLevel::Debug, "replay(): expand");

            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo src_info = this->dependencies[dependency_index];
            ParamInfo dst_info = this->dependencies[dependency_index + 1];

            ReplayVariable &src_rv = replay_variables[src_info.slot];
            ReplayVariable &dst_rv = replay_variables[dst_info.slot];

            VarType vt = src_info.vtype;
            uint32_t size = src_rv.size(vt);
            uint32_t tsize = type_size[(uint32_t)vt];
            uint32_t workers = pool_size() + 1;

            if (size != op.size)
                return false;

            uint32_t replication_per_worker = size == 1u ? (64u / tsize) : 1u;
            size_t new_size =
                size * (size_t)replication_per_worker * (size_t)workers;

            dst_rv.alloc(backend, new_size, dst_info.vtype);

            jitc_log(LogLevel::Debug, "    data=0x%lx", op.data);
            if (!dry_run)
                ts->memset_async(dst_rv.data, new_size, tsize, &op.data);
            jitc_log(LogLevel::Debug,
                     "jitc_memcpy_async(dst=%p, src=%p, size=%zu)", dst_rv.data,
                     src_rv.data, (size_t)size * tsize);
            if (!dry_run)
                ts->memcpy_async(dst_rv.data, src_rv.data,
                                 (size_t)size * tsize);

            dst_rv.data_size = size * type_size[(uint32_t)dst_info.vtype];
            // dst_rv.size = size;
        } break;
        case OpType::PrefixSum: {
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo in_info = this->dependencies[dependency_index];
            ParamInfo out_info = this->dependencies[dependency_index + 1];

            ReplayVariable &in_var = replay_variables[in_info.slot];
            ReplayVariable &out_var = replay_variables[out_info.slot];

            VarType type = in_info.vtype;

            out_var.alloc(backend, in_var.size(type), out_info.vtype);

            if (!dry_run)
                ts->prefix_sum(type, op.exclusive, in_var.data, op.size,
                               out_var.data);
        } break;
        case OpType::Compress: {
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo in_info = this->dependencies[dependency_index];
            ParamInfo out_info = this->dependencies[dependency_index + 1];

            ReplayVariable &in_rv = replay_variables[in_info.slot];
            ReplayVariable &out_rv = replay_variables[out_info.slot];

            uint32_t size = in_rv.size(in_info.vtype);
            out_rv.alloc(backend, size, out_info.vtype);

            if (dry_run)
                jitc_fail(
                    "replay(): dry_run compress operation not supported!");

            uint32_t out_size = ts->compress((uint8_t *)in_rv.data, size,
                                             (uint32_t *)out_rv.data);

            out_rv.data_size = out_size * type_size[(uint32_t)out_info.vtype];
        } break;
        case OpType::MemcpyAsync: {
            uint32_t dependency_index = op.dependency_range.first;
            ParamInfo src_info = this->dependencies[dependency_index];
            ParamInfo dst_info = this->dependencies[dependency_index + 1];

            ReplayVariable &src_var = replay_variables[src_info.slot];
            ReplayVariable &dst_var = replay_variables[dst_info.slot];


            // size_t size = src_var.size(src_info.vtype);
            jitc_log(LogLevel::Debug,
                     "replay(): MemcpyAsync slot(%u) <- slot(%u) [%zu]",
                     dst_info.slot, src_info.slot, src_var.data_size);

            dst_var.alloc(backend, src_var.data_size);

            jitc_log(LogLevel::Debug, "    src.data=%p", src_var.data);
            jitc_log(LogLevel::Debug, "    dst.data=%p", dst_var.data);

            if (!dry_run)
                ts->memcpy_async(dst_var.data, src_var.data, src_var.data_size);
        } break;
        case OpType::Mkperm: {
            jitc_log(LogLevel::Debug, "Mkperm:");
            uint32_t dependency_index = op.dependency_range.first;
            ParamInfo values_info = this->dependencies[dependency_index];
            ParamInfo perm_info = this->dependencies[dependency_index + 1];
            ParamInfo offsets_info = this->dependencies[dependency_index + 2];

            ReplayVariable &values_var = replay_variables[values_info.slot];
            ReplayVariable &perm_var = replay_variables[perm_info.slot];
            ReplayVariable &offsets_var = replay_variables[offsets_info.slot];

            uint32_t size = values_var.size(values_info.vtype);
            uint32_t bucket_count = op.bucket_count;

            jitc_log(LogLevel::Info, " -> values: slot(%u, data=%p, size=%u)",
                     values_info.slot, values_var.data, size);

            jitc_log(LogLevel::Info, " <- perm: slot(%u)", perm_info.slot);
            perm_var.alloc(backend, size, perm_info.vtype);

            jitc_log(LogLevel::Info, " <- offsets: slot(%u)",
                     offsets_info.slot);
            offsets_var.alloc(backend, bucket_count * 4 + 1,
                              offsets_info.vtype);

            jitc_log(LogLevel::Debug,
                     "    mkperm(values=%p, size=%u, "
                     "bucket_count=%u, perm=%p, offsets=%p)",
                     values_var.data, size, bucket_count, perm_var.data,
                     offsets_var.data);

            if (!dry_run)
                ts->mkperm((uint32_t *)values_var.data, size, bucket_count,
                           (uint32_t *)perm_var.data,
                           (uint32_t *)offsets_var.data);

        } break;
        case OpType::Aggregate: {
            jitc_log(LogLevel::Debug, "replay(): Aggregate");

            uint32_t i = op.dependency_range.first;

            ParamInfo dst_info = this->dependencies[i++];
            ReplayVariable &dst_rv = replay_variables[dst_info.slot];

            AggregationEntry *agg = nullptr;

            size_t agg_size = sizeof(AggregationEntry) * op.size;

            if (backend == JitBackend::CUDA)
                agg = (AggregationEntry *)jitc_malloc(AllocType::HostPinned,
                                                      agg_size);
            else
                agg = (AggregationEntry *)malloc_check(agg_size);

            AggregationEntry *p = agg;

            for (; i < op.dependency_range.second; ++i) {
                ParamInfo param = this->dependencies[i];

                if (param.type == ParamType::Input) {
                    ReplayVariable &rv = replay_variables[param.slot];
                    jitc_assert(
                        rv.data != nullptr || dry_run,
                        "replay(): Encountered nullptr input parameter.");

                    jitc_log(LogLevel::Debug,
                             " -> slot(%u, is_pointer=%u, data=%p, offset=%u)",
                             param.slot, param.pointer_access, rv.data,
                             param.extra.offset);

                    p->size = param.pointer_access
                                  ? 8
                                  : -(int)type_size[(uint32_t)param.vtype];
                    p->offset = param.extra.offset;
                    p->src = rv.data;
                } else {
                    jitc_log(LogLevel::Debug, " -> literal: offset=%u, size=%u",
                             param.extra.offset, param.extra.type_size);
                    p->size = param.extra.type_size;
                    p->offset = param.extra.offset;
                    p->src = (void *)param.extra.data;
                }

                p++;
            }

            AggregationEntry *last = p - 1;
            uint32_t data_size =
                last->offset + (last->size > 0 ? last->size : -last->size);
            // Restore to full alignment
            data_size = (data_size + 7) / 8 * 8;

            dst_rv.alloc(backend, data_size, VarType::UInt8);

            jitc_log(LogLevel::Debug,
                     " <- slot(%u, is_pointer=%u, data=%p, size=%u)",
                     dst_info.slot, dst_info.pointer_access, dst_rv.data,
                     data_size);

            jitc_assert(dst_rv.data != nullptr || dry_run,
                        "replay(): Error allocating dst.");

            if (!dry_run)
                ts->aggregate(dst_rv.data, agg, (uint32_t)(p - agg));

        } break;
        case OpType::Free: {
            uint32_t i         = op.dependency_range.first;
            ParamInfo info     = dependencies[i];
            ReplayVariable &rv = replay_variables[info.slot];

            rv.free();

        } break;
        default:
            jitc_fail("An operation has been recorded, that is not known to "
                      "the replay functionality!");
            break;
        }
        jitc_sync_thread(ts);
    }

    ts->optix_sbt = tmp_sbt;

    if (dry_run)
        return true;

    // Create output variables
    for (uint32_t i = 0; i < this->outputs.size(); ++i) {
        ParamInfo info = this->outputs[i];
        uint32_t slot = info.slot;
        // uint32_t index = this->outputs[i];
        jitc_log(LogLevel::Debug, "replay(): output(%u, slot=%u)", i, slot);
        ReplayVariable &rv = replay_variables[slot];
        // if (rv.type != info.vtype)
        //     rv.prepare_input(info.vtype);

        if (rv.rv_type == RecordType::Input) {
            // Use input variable
            jitc_log(LogLevel::Debug, "    uses input %u", rv.index);
            jitc_assert(rv.data, "replay(): freed an input variable "
                                 "that is passed through!");
            uint32_t var_index = replay_inputs[rv.index];
            jitc_var_inc_ref(var_index);
            outputs[i] = var_index;
        } else if (rv.rv_type == RecordType::Captured) {
            jitc_log(LogLevel::Info, "    uses captured variable %u", rv.index);
            jitc_assert(rv.data, "replay(): freed an input variable "
                                 "that is passed through!");

            jitc_var_inc_ref(rv.index);

            outputs[i] = rv.index;
        } else {
            jitc_log(LogLevel::Info, "    uses internal variable");
            if(!rv.data)
                jitc_fail("replay(): freed slot %u used for output.", slot);
            outputs[i] = jitc_var_mem_map(this->backend, info.vtype, rv.data,
                                          rv.size(info.vtype), true);
        }
        // Set data to nullptr for next step, where we free all remaining
        // temporary variables
        jitc_log(LogLevel::Info, "    data=%p", rv.data);
        rv.data = nullptr;
    }

    for (uint32_t slot = 0; slot < replay_variables.size(); ++slot) {
        ReplayVariable &rv = replay_variables[slot];
        if (rv.data && rv.rv_type == RecordType::Other) {
            jitc_log(LogLevel::Debug,
                     "replay(): Freeing temporary data %p at slot %u", rv.data,
                     slot);

            jitc_free(rv.data);
            rv.data = nullptr;
        }
    }

    return true;
}

void Recording::validate() {
}

void jitc_record_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs) {

    if (jitc_flags() & (uint32_t)JitFlag::FreezingScope)
        jitc_fail("Tried to record a thread_state while inside another "
                  "FreezingScope!");

    // Increment scope, can be used to track missing inputs
    jitc_new_scope(backend);

    ThreadState *ts = thread_state(backend);
    RecordThreadState *record_ts = new RecordThreadState(ts);

    if (backend == JitBackend::CUDA) {
        thread_state_cuda = record_ts;
    } else {
        thread_state_llvm = record_ts;
    }

    for (uint32_t i = 0; i < n_inputs; ++i) {
        record_ts->add_input(inputs[i]);
    }

    uint32_t flags = jitc_flags();
    flags |= (uint32_t)JitFlag::FreezingScope;
    jitc_set_flags(flags);
}
Recording *jitc_record_stop(JitBackend backend, const uint32_t *outputs,
                            uint32_t n_outputs) {
    if (RecordThreadState *rts =
            dynamic_cast<RecordThreadState *>(thread_state(backend));
        rts != nullptr) {
        ThreadState *internal = rts->internal;

        // Perform reasignments to internal thread-state of possibly changed
        // variables
        internal->scope = rts->scope;

        jitc_assert(rts->record_stack.empty(),
                    "Kernel recording ended while still recording loop!");

        for (uint32_t i = 0; i < n_outputs; ++i) {
            rts->add_output(outputs[i]);
        }

        if (backend == JitBackend::CUDA) {
            thread_state_cuda = internal;
        } else {
            thread_state_llvm = internal;
        }
        Recording *recording = new Recording(rts->recording);
        recording->validate();
        delete rts;

        uint32_t flags = jitc_flags();
        flags &= ~(uint32_t)JitFlag::FreezingScope;
        jitc_set_flags(flags);

        return recording;
    } else {
        jitc_fail(
            "jit_record_stop(): Tried to stop recording a thread state "
            "for backend %u, while no recording was started for this backend. "
            "Try to start the recording with jit_record_start.",
            (uint32_t)backend);
    }
}

void jitc_record_abort(JitBackend backend) {
    if (RecordThreadState *rts =
            dynamic_cast<RecordThreadState *>(thread_state(backend));
        rts != nullptr) {

        ThreadState *internal = rts->internal;

        // Perform reasignments to internal thread-state of possibly changed
        // variables
        internal->scope = rts->scope;

        if (backend == JitBackend::CUDA) {
            thread_state_cuda = internal;
        } else {
            thread_state_llvm = internal;
        }

        delete rts;

        uint32_t flags = jitc_flags();
        flags &= ~(uint32_t)JitFlag::FreezingScope;
        jitc_set_flags(flags);
    }
}

void jitc_record_destroy(Recording *recording) {
    for (RecordVariable &rv : recording->record_variables) {
        if (rv.rv_type == RecordType::Captured) {
            jitc_var_dec_ref(rv.index);
        }
    }
    delete recording;
}

bool jitc_record_pause(JitBackend backend) {

    if (RecordThreadState *rts =
            dynamic_cast<RecordThreadState *>(thread_state(backend));
        rts != nullptr) {
        return rts->pause();
    } else {
        jitc_fail(
            "jit_record_stop(): Tried to pause recording a thread state "
            "for backend %u, while no recording was started for this backend. "
            "Try to start the recording with jit_record_start.",
            (uint32_t)backend);
    }
}
bool jitc_record_resume(JitBackend backend) {
    if (RecordThreadState *rts =
            dynamic_cast<RecordThreadState *>(thread_state(backend));
        rts != nullptr) {
        return rts->resume();
    } else {
        jitc_fail(
            "jit_record_stop(): Tried to resume recording a thread state "
            "for backend %u, while no recording was started for this backend. "
            "Try to start the recording with jit_record_start.",
            (uint32_t)backend);
    }
}

void jitc_record_replay(Recording *recording, const uint32_t *inputs,
                        uint32_t *outputs) {
    dry_run = false;
    recording->replay(inputs, outputs);
}

int jitc_record_dry_run(Recording *recording, const uint32_t *inputs,
                        uint32_t *outputs) {
    int result = true;
    if(recording->requires_dry_run){
        jitc_log(LogLevel::Debug, "Replaying in dry-run mode");
        dry_run = true;
        result = recording->replay(inputs, outputs);
        dry_run = false;
    }
    
    return result;
}
