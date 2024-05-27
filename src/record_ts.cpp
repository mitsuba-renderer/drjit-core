#include "record_ts.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"
#include "var.h"

// This struct holds the data and tracks the size of varaibles,
// used during replay.
struct ReplayVariable {
    void *data = 0;
    uint32_t size = 0;
    VarType type;
    uint32_t index;
    bool is_literal;
    RecordType rv_type;
    // bool is_input;
    // bool is_captured;
    uint32_t rc;

    ReplayVariable(RecordVariable &rv) {
        this->type = rv.type;
        this->size = rv.size;
        this->index = rv.index;
        this->is_literal = rv.is_literal;
        this->rc = rv.rc;

        this->rv_type = rv.rv_type;

        if (rv_type == RecordType::Captured) {
            Variable *v = jitc_var(this->index);
            this->data = v->data;
        }
    }

    void alloc(JitBackend backend) {
        if (!data) {

            size_t dsize = ((size_t)size) * ((size_t)type_size[(int)type]);

            jitc_log(LogLevel::Debug, "    allocating output of size %zu.",
                     dsize);

            AllocType alloc_type = backend == JitBackend::CUDA
                                       ? AllocType::Device
                                       : AllocType::Host;

            data = jitc_malloc(alloc_type, dsize);
        } else {
            jitc_log(LogLevel::Warn,
                     "replay(): Tried to allocate replay variable twice! "
                     "Usually, only output variables are allocated. This "
                     "indicates that something went wrong when recording.");
        }
    }
};

/// Kernel parameter buffer
static std::vector<void *> kernel_params;

/// Temporary variables used for replaying a recording.
static std::vector<ReplayVariable> replay_variables;

void Recording::replay(const uint32_t *replay_inputs, uint32_t *outputs) {

    // Perform validation
    for (uint32_t i = 0; i < this->record_variables.size(); ++i) {
        RecordVariable &rv = this->record_variables[i];
        jitc_assert(rv.type != VarType::Void && rv.size > 0,
                    "Recorded Variable at slot(%u) was added, it's type is "
                    "unknown! This can occur, if a variable is only used by "
                    "operations, that do not provide a type.",
                    i);
    }

    ThreadState *ts = thread_state(backend);
    jitc_assert(dynamic_cast<RecordThreadState *>(ts) == nullptr,
                "replay(): Tried to replay while recording!");

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
        jitc_log(LogLevel::Debug, "    input %u: r%u mapped to slot(%u)", i,
                 replay_inputs[i], this->inputs[i]);
    }

    // Execute kernels and allocate missing output variables

    for (uint32_t i = 0; i < this->operations.size(); ++i) {
        Operation &op = this->operations[i];

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

            // Inferr launch size.

            // Size of direct input variables
            uint32_t input_size = 0;
            // Size of variables referenced by pointers
            uint32_t ptr_size = 0;

            for (uint32_t j = op.dependency_range.first;
                 j < op.dependency_range.second; ++j) {
                ParamInfo info = this->dependencies[j];
                ReplayVariable &rv = replay_variables[info.index];

                if (info.type == ParamType::Input) {
                    jitc_assert(
                        rv.data != nullptr,
                        "replay(): Kernel input variable not allocated!");

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
                             "launch_size(%u) *= ratio(%zu)",
                             ptr_size, ratio);
                    launch_size = launch_size * ratio;
                } else {
                    if (op.input_size % op.size != 0)
                        jitc_raise(
                            "replay(): Could not infer launch size, using "
                            "heuristic!");

                    uint32_t fraction = op.input_size / op.size;
                    jitc_log(LogLevel::Warn,
                             "replay(): Inferring launch size by heuristic, "
                             "launch_size(%u) /= fraction(%u)",
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
                ReplayVariable &rv = replay_variables[info.index];

                jitc_log(LogLevel::Info,
                         " %s param slot(%u, rc=%u, is_pointer=%u, size=%u)",
                         info.type == ParamType::Output ? "<-" : "->",
                         info.index, rv.rc, info.pointer_access, rv.size);

                if (info.type == ParamType::Output) {
                    rv.size = launch_size;
                    rv.alloc(backend);
                }
                jitc_log(LogLevel::Info, "    data=%p", rv.data);
                jitc_assert(
                    rv.data != nullptr,
                    "replay(): Encountered nullptr in kernel parameters.");
                kernel_params.push_back(rv.data);
            }

            // Change kernel size in `kernel_params`
            if (backend == JitBackend::CUDA) {
                uintptr_t size = 0;
                std::memcpy(&size, &launch_size, sizeof(uint32_t));
                kernel_params[0] = (void *)size;
            }

            {
                jitc_log(LogLevel::Info, "    launch_size=%u", launch_size);
                scoped_set_context_maybe guard2(ts->context);
                std::vector<uint32_t> tmp;
                Kernel kernel = op.kernel;
                ts->launch(kernel, launch_size, &kernel_params, &tmp);
            }

        }

        break;
        case OpType::Barrier:
            ts->barrier();
            break;
        case OpType::MemsetAsync: {
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo ptr_info = this->dependencies[dependency_index];
            ParamInfo src_info = this->dependencies[dependency_index + 1];

            ReplayVariable &ptr_var = replay_variables[ptr_info.index];
            ReplayVariable &src_var = replay_variables[src_info.index];

            VarType type = replay_variables[src_info.index].type;

            ts->memset_async(ptr_var.data, op.size, type_size[(uint32_t)type],
                             src_var.data);
        } break;
        case OpType::Reduce: {
            jitc_log(LogLevel::Debug, "replay(): Reduce");
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo ptr_info = this->dependencies[dependency_index];
            ParamInfo out_info = this->dependencies[dependency_index + 1];

            ReplayVariable &ptr_var = replay_variables[ptr_info.index];
            ReplayVariable &out_var = replay_variables[out_info.index];

            out_var.alloc(backend);

            VarType type = ptr_var.type;
            ReduceOp rtype = op.rtype;

            jitc_log(LogLevel::Debug, " -> slot(%u, data=%p)", ptr_info.index,
                     ptr_var.data);
            jitc_log(LogLevel::Debug, " <- slot(%u, data=%p)", out_info.index,
                     out_var.data);

            ts->reduce(type, rtype, ptr_var.data, ptr_var.size, out_var.data);
        } break;
        case OpType::PrefixSum: {
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo in_info = this->dependencies[dependency_index];
            ParamInfo out_info = this->dependencies[dependency_index + 1];

            ReplayVariable &in_var = replay_variables[in_info.index];
            ReplayVariable &out_var = replay_variables[out_info.index];

            out_var.alloc(backend);

            VarType type = in_var.type;

            ts->prefix_sum(type, op.exclusive, in_var.data, op.size,
                           out_var.data);
        } break;
        case OpType::MemcpyAsync: {
            uint32_t dependency_index = op.dependency_range.first;
            ParamInfo src_info = this->dependencies[dependency_index];
            ParamInfo dst_info = this->dependencies[dependency_index + 1];

            ReplayVariable &src_var = replay_variables[src_info.index];
            ReplayVariable &dst_var = replay_variables[dst_info.index];

            jitc_log(LogLevel::Debug,
                     "replay(): MemcpyAsync slot(%u) <- slot(%u)",
                     dst_info.index, src_info.index);

            dst_var.size = src_var.size;
            dst_var.alloc(backend);

            jitc_log(LogLevel::Debug, "    src.data=%p", src_var.data);
            jitc_log(LogLevel::Debug, "    dst.data=%p", dst_var.data);

            size_t size = src_var.size * type_size[(uint32_t)src_var.type];

            ts->memcpy_async(dst_var.data, src_var.data, size);
        } break;
        case OpType::Mkperm: {
            jitc_log(LogLevel::Debug, "Mkperm:");
            uint32_t dependency_index = op.dependency_range.first;
            ParamInfo values_info = this->dependencies[dependency_index];
            ParamInfo perm_info = this->dependencies[dependency_index + 1];
            ParamInfo offsets_info = this->dependencies[dependency_index + 2];

            ReplayVariable &values_var = replay_variables[values_info.index];
            ReplayVariable &perm_var = replay_variables[perm_info.index];
            ReplayVariable &offsets_var = replay_variables[offsets_info.index];

            uint32_t size = values_var.size;
            uint32_t bucket_count = op.bucket_count;

            jitc_log(LogLevel::Info, " -> values: slot(%u, data=%p, size=%u)",
                     values_info.index, values_var.data, values_var.size);

            jitc_log(LogLevel::Info, " <- perm: slot(%u)", perm_info.index);
            perm_var.size = size;
            perm_var.alloc(backend);

            jitc_log(LogLevel::Info, " <- offsets: slot(%u)",
                     offsets_info.index);
            offsets_var.size = bucket_count * 4 + 1;
            offsets_var.alloc(backend);

            jitc_log(LogLevel::Debug,
                     "    mkperm(values=%p, size=%u, "
                     "bucket_count=%u, perm=%p, offsets=%p)",
                     values_var.data, size, bucket_count, perm_var.data,
                     offsets_var.data);

            ts->mkperm((uint32_t *)values_var.data, size, bucket_count,
                       (uint32_t *)perm_var.data, (uint32_t *)offsets_var.data);

        } break;
        case OpType::Aggregate: {
            jitc_log(LogLevel::Debug, "replay(): Aggregate");

            uint32_t i = op.dependency_range.first;

            ParamInfo dst_info = this->dependencies[i++];
            ReplayVariable &dst_rv = replay_variables[dst_info.index];
            // Assume, that size is known
            dst_rv.alloc(backend);

            jitc_log(LogLevel::Debug, " <- slot(%u, is_pointer=%u, data=%p)",
                     dst_info.index, dst_info.pointer_access, dst_rv.data);

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
                jitc_assert(param.type == ParamType::Input, "");

                ReplayVariable &rv = replay_variables[param.index];
                jitc_assert(rv.data != nullptr,
                            "replay(): Encountered nullptr input parameter.");

                jitc_log(LogLevel::Debug,
                         " -> slot(%u, is_pointer=%u, data=%p)", param.index,
                         param.pointer_access, rv.data);

                p->size = param.pointer_access
                              ? 8
                              : -(int)type_size[(uint32_t)rv.type];
                p->offset = param.extra;
                p->src = rv.data;
                p++;
            }

            jitc_assert(dst_rv.data != nullptr,
                        "replay(): Error allocating dst.");

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
                    ReplayVariable &rv = replay_variables[info.index];
                    rv.rc--;
                    jitc_log(LogLevel::Debug,
                             "replay(): decrement rc for slot %u new rc=%u",
                             info.index, rv.rc);
                    if (rv.rc == 0) {
                        jitc_log(LogLevel::Debug,
                                 "replay(): free memory for slot %u",
                                 info.index);
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
            RecordVariable &rv = this->record_variables[info.index];
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
