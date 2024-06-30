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
    void *data = 0;
    size_t alloc_size = 0;
    uint32_t size = 0;
    VarType type = VarType::Void;
    uint32_t index;
    bool is_literal;
    RecordType rv_type;
    uint32_t rc;

    ReplayVariable(RecordVariable &rv) {
        this->index = rv.index;
        this->is_literal = rv.is_literal;
        this->rc = rv.rc;

        this->rv_type = rv.rv_type;

        if (rv_type == RecordType::Captured) {
            Variable *v = jitc_var(this->index);
            this->data = v->data;
            this->size = v->size;
            this->alloc_size = v->size * type_size[v->type];
        }
    }

    // void set_type(VarType type) {
    //     if (type != VarType::Void)
    //         this->type = type;
    // }
    void prepare_input(VarType type) {
        if (type != VarType::Void) {
            uint32_t tsize = type_size[(uint32_t)type];
            this->size = (uint32_t)(this->alloc_size / (size_t)tsize);
            jitc_log(LogLevel::Debug,
                     "replay(): reinterpreted as %u with size [%u]",
                     (uint32_t)type, this->size);
            if (this->size == 0)
                jitc_fail("replay(): Error, chaning type of replay variable!");
            this->type = type;
        }
    }

    void alloc(JitBackend backend, uint32_t size, VarType type) {
        if (type != VarType::Void)
            this->type = type;
        alloc(backend, size);
    }
    void alloc(JitBackend backend, uint32_t size) {
        this->size = size;
        alloc(backend);
    }
    void alloc(JitBackend backend) {
        alloc(backend, this->size, type_size[(uint32_t)this->type]);
    }
    /**
     * Allocates the data for this replay variable.
     * If this is atempted twice, we test weather the allocated size is
     * sufficient and re-allocate the memory if necesarry.
     */
    void alloc(JitBackend backend, uint32_t size, uint32_t isize) {
        // size_t dsize = ((size_t)size) * ((size_t)type_size[(int)type]);
        this->size = size;
        size_t dsize = ((size_t)size) * ((size_t)isize);
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
    }
};

/// Kernel parameter buffer
static std::vector<void *> kernel_params;

/// Temporary variables used for replaying a recording.
static std::vector<ReplayVariable> replay_variables;

void Recording::replay(const uint32_t *replay_inputs, uint32_t *outputs) {

    this->validate();

    ThreadState *ts = thread_state(backend);
    jitc_assert(dynamic_cast<RecordThreadState *>(ts) == nullptr,
                "replay(): Tried to replay while recording!");

    OptixShaderBindingTable *tmp_sbt = ts->optix_sbt;
    scoped_set_context_maybe guard2(ts->context);

    replay_variables.clear();
    uint32_t last_free = 0;

    replay_variables.reserve(this->record_variables.size());
    for (RecordVariable &rv : this->record_variables) {
        replay_variables.push_back(ReplayVariable(rv));
    }

    jitc_log(LogLevel::Info, "replay(): inputs");
    // Populate with input variables
    for (uint32_t i = 0; i < this->inputs.size(); ++i) {
        Variable *input_variable = jitc_var(replay_inputs[i]);
        ReplayVariable &rv = replay_variables[this->inputs[i]];
        rv.size = input_variable->size;
        rv.data = input_variable->data;
        rv.type = (VarType)input_variable->type;
        rv.alloc_size = type_size[input_variable->type] * rv.size;
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
            jitc_log(LogLevel::Info, "replay(): launching kernel:");
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

            // Determine type of input variables
            for (uint32_t j = op.dependency_range.first;
                 j < op.dependency_range.second; ++j) {
                ParamInfo info = this->dependencies[j];
                ReplayVariable &rv = replay_variables[info.slot];
                if (info.type == ParamType::Input &&
                    rv.rv_type != RecordType::Captured) {
                    jitc_log(
                        LogLevel::Debug,
                        "preparing input variable s%u type=%u, alloc_size=%zu",
                        info.slot, (uint32_t)info.vtype, rv.alloc_size);
                    rv.prepare_input(info.vtype);
                }
            }

            // Inferr launch size.

            // Size of direct input variables
            uint32_t input_size = 0;
            // Size of variables referenced by pointers
            uint32_t ptr_size = 0;

            for (uint32_t j = op.dependency_range.first;
                 j < op.dependency_range.second; ++j) {
                ParamInfo info = this->dependencies[j];
                ReplayVariable &rv = replay_variables[info.slot];

                if (info.type == ParamType::Input) {
                    jitc_assert(rv.data != nullptr || dry_run,
                                "replay(): Kernel input variable (slot=%u) not "
                                "allocated!",
                                info.slot);

                    if (!info.pointer_access)
                        input_size = std::max(input_size, rv.size);
                    else
                        ptr_size = std::max(ptr_size, rv.size);
                }
            }

            uint32_t launch_size = input_size != 0 ? input_size : ptr_size;
            if (op.input_size > 0) {
                // Apply the factor
                if (op.size > op.input_size) {
                    if (op.size % op.input_size != 0)
                        jitc_raise(
                            "replay(): Could not infer launch size, using "
                            "heuristic!");
                    size_t ratio = op.size / op.input_size;
                    jitc_log(LogLevel::Warn,
                             "replay(): Inferring launch size by heuristic, "
                             "launch_size=%u, ratio=%zu",
                             launch_size, ratio);
                    launch_size = launch_size * ratio;
                } else {
                    if (op.input_size % op.size != 0)
                        jitc_raise(
                            "replay(): Could not infer launch size, using "
                            "heuristic!");

                    uint32_t fraction = op.input_size / op.size;
                    jitc_log(LogLevel::Warn,
                             "replay(): Inferring launch size by heuristic, "
                             "launch_size(%u), fraction=%u",
                             launch_size, fraction);
                    launch_size = launch_size / fraction;
                }
            }
            if (launch_size == 0) {
                jitc_log(LogLevel::Warn, "replay(): Could not infer launch "
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

                jitc_log(LogLevel::Info,
                         " %s param slot(%u, rc=%u, is_pointer=%u, size=%u)",
                         info.type == ParamType::Output ? "<-" : "->",
                         info.slot, rv.rc, info.pointer_access, rv.size);

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

            jitc_log(LogLevel::Debug, "replay(): MemsetAsync -> slot(%u) [%zu]",
                     ptr_info.slot, op.size);

            if (!dry_run)
                ts->memset_async(ptr_var.data, ptr_var.size, op.input_size,
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

            VarType type = ptr_var.type;
            ReduceOp rtype = op.rtype;

            jitc_log(LogLevel::Debug, " -> slot(%u, data=%p)", ptr_info.slot,
                     ptr_var.data);
            jitc_log(LogLevel::Debug, " <- slot(%u, data=%p)", out_info.slot,
                     out_var.data);

            if (!dry_run)
                ts->reduce(type, rtype, ptr_var.data, ptr_var.size,
                           out_var.data);
        } break;
        case OpType::ReduceExpanded: {
            jitc_log(LogLevel::Debug, "replay(): ReduceExpand");

            uint32_t dependency_index = op.dependency_range.first;
            ParamInfo data_info = this->dependencies[dependency_index];

            ReplayVariable &data_var = replay_variables[data_info.slot];

            VarType vt = data_var.type;
            ReduceOp rop = op.rtype;
            uint32_t size = data_var.size;
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

            VarType vt = src_rv.type;
            uint32_t size = src_rv.size;
            uint32_t tsize = type_size[(uint32_t)vt];
            uint32_t workers = pool_size() + 1;

            if (size != op.size)
                throw RequiresRetraceException();

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
            dst_rv.size = size;
        } break;
        case OpType::PrefixSum: {
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo in_info = this->dependencies[dependency_index];
            ParamInfo out_info = this->dependencies[dependency_index + 1];

            ReplayVariable &in_var = replay_variables[in_info.slot];
            ReplayVariable &out_var = replay_variables[out_info.slot];

            out_var.alloc(backend, in_var.size, out_info.vtype);

            VarType type = in_var.type;

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

            uint32_t size = in_rv.size;
            out_rv.alloc(backend, size, out_info.vtype);

            if (dry_run)
                jitc_fail(
                    "replay(): dry_run compress operation not supported!");

            uint32_t out_size = ts->compress((uint8_t *)in_rv.data, size,
                                             (uint32_t *)out_rv.data);

            out_rv.size = out_size;
        } break;
        case OpType::MemcpyAsync: {
            uint32_t dependency_index = op.dependency_range.first;
            ParamInfo src_info = this->dependencies[dependency_index];
            ParamInfo dst_info = this->dependencies[dependency_index + 1];

            ReplayVariable &src_var = replay_variables[src_info.slot];
            ReplayVariable &dst_var = replay_variables[dst_info.slot];

            jitc_log(LogLevel::Debug,
                     "replay(): MemcpyAsync slot(%u) <- slot(%u) [%u]",
                     dst_info.slot, src_info.slot, src_var.size);

            dst_var.alloc(backend, src_var.size, dst_info.vtype);

            jitc_log(LogLevel::Debug, "    src.data=%p", src_var.data);
            jitc_log(LogLevel::Debug, "    dst.data=%p", dst_var.data);

            size_t size = src_var.size * type_size[(uint32_t)src_var.type];

            if (!dry_run)
                ts->memcpy_async(dst_var.data, src_var.data, size);
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

            uint32_t size = values_var.size;
            uint32_t bucket_count = op.bucket_count;

            jitc_log(LogLevel::Info, " -> values: slot(%u, data=%p, size=%u)",
                     values_info.slot, values_var.data, values_var.size);

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
                                  : -(int)type_size[(uint32_t)rv.type];
                    p->offset = param.extra.offset;
                    p->src = rv.data;
                } else {
                    jitc_log(LogLevel::Debug, " -> literal: offset=%u",
                             param.extra.offset);
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
        default:
            jitc_fail("An operation has been recorded, that is not known to "
                      "the replay functionality!");
            break;
        }

        // Only kernel launches have to be synchronized
        if (op.type != OpType::KernelLaunch) {
            // Free unused memory, allocated since last barrier
            for (uint32_t j = last_free; j < i; ++j) {
                jitc_log(LogLevel::Debug, "replay(): gc for operation %u", j);
                Operation &op = this->operations[j];
                for (uint32_t p = op.dependency_range.first;
                     p < op.dependency_range.second; ++p) {
                    ParamInfo &info = this->dependencies[p];
                    if (info.type != ParamType::Input &&
                        info.type != ParamType::Output)
                        continue;

                    ReplayVariable &rv = replay_variables[info.slot];
                    rv.rc--;
                    jitc_log(LogLevel::Debug,
                             "replay(): decrement rc for slot %u new rc=%u",
                             info.slot, rv.rc);
                    if (rv.rc == 0) {
                        jitc_log(LogLevel::Debug,
                                 "replay(): free memory for slot %u",
                                 info.slot);
                        jitc_free(rv.data);
                        rv.data = nullptr;
                    }
                }
            }
            last_free = i;
        }
    }

    for (uint32_t i = 0; i < replay_variables.size(); ++i) {
        ReplayVariable &rv = replay_variables[i];
        jitc_log(LogLevel::Debug,
                 "replay(): rv(%u, rc=%u, is_input=%u, data=%p)", i, rv.rc,
                 rv.rv_type == RecordType::Input, rv.data);
    }

    ts->optix_sbt = tmp_sbt;

    if (dry_run)
        return;

    // Create output variables
    for (uint32_t i = 0; i < this->outputs.size(); ++i) {
        uint32_t index = this->outputs[i];
        jitc_log(LogLevel::Debug, "replay(): output(%u, slot=%u)", i, index);
        ReplayVariable &rv = replay_variables[index];
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
            jitc_assert(rv.data && rv.rc > 0,
                        "replay(): freed variable used for output.");
            outputs[i] = jitc_var_mem_map(this->backend, rv.type, rv.data,
                                          rv.size, true);
        }
        jitc_log(LogLevel::Info, "    data=%p", rv.data);
    }
}

void Recording::compute_rc() {
    for (auto &op : this->operations) {
        for (uint32_t j = op.dependency_range.first;
             j < op.dependency_range.second; ++j) {
            ParamInfo &info = this->dependencies[j];
            if (info.type != ParamType::Input && info.type != ParamType::Output)
                continue;

            RecordVariable &rv = this->record_variables[info.slot];
            rv.rc++;
        }
    }
    // Captured and Input variables should not be garbage collected.
    // TODO: inputs should be fine, but have to be freed by decrementing
    // variable refcount.
    for (RecordVariable &rv : this->record_variables) {
        if (rv.rv_type == RecordType::Input ||
            rv.rv_type == RecordType::Captured) {
            rv.rc++;
        }
    }
    // Do not touch inputs/outputs therefore increment their refcount
    for (uint32_t slot : this->outputs) {
        RecordVariable &rv = this->record_variables[slot];
        rv.rc++;
    }
}
void Recording::validate() {
    for (uint32_t i = 0; i < this->record_variables.size(); ++i) {
        RecordVariable &rv = this->record_variables[i];
        // jitc_assert(rv.type != VarType::Void || rv.rc == 0,
        //             "Recorded Variable at slot(%u) was added, it's type is "
        //             "unknown! This can occur, if a variable is only used by "
        //             "operations, that do not provide a type.",
        //             i);
    }
}

void jitc_record_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs) {

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
        recording->compute_rc();
        recording->validate();
        delete rts;
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
    if (recording->requires_dry_run) {
        jitc_log(LogLevel::Debug, "Replaying in dry-run mode");
        dry_run = true;
        recording->replay(inputs, outputs);
        dry_run = false;
    }
    recording->replay(inputs, outputs);
}
