#include "record_ts.h"
#include "common.h"
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
        jitc_assert(data == nullptr,
                    "replay(): Output parameters should not be "
                    "allocate before replaying the kernel!");

        uint32_t dsize = size * type_size[(int)type];

        jitc_log(LogLevel::Info, "    allocating output of size %u.", dsize);

        AllocType alloc_type =
            backend == JitBackend::CUDA ? AllocType::Device : AllocType::Host;

        data = jitc_malloc(alloc_type, dsize);
    }
};

/// Kernel parameter buffer
static std::vector<void *> kernel_params;

/// Temporary scratch space for scheduled tasks (LLVM only)
static std::vector<Task *> scheduled_tasks;

/// Temporary variables used for replaying a recording.
static std::vector<ReplayVariable> replay_variables;

void Recording::replay(const uint32_t *replay_inputs, uint32_t *outputs) {

    // Perform validation
    for (uint32_t i = 0; i < this->record_variables.size(); ++i) {
        RecordVariable &rv = this->record_variables[i];
        jitc_assert(
            rv.type != VarType::Void,
            "Recorded Variable at slot(%u) was added, but not completed!", i);
    }

    ThreadState *ts = thread_state(backend);
    jitc_assert(dynamic_cast<RecordThreadState *>(ts) == nullptr,
                "replay(): Cannot replay while recording!");

    replay_variables.clear();
    scheduled_tasks.clear();
    uint32_t last_free = 0;

    replay_variables.reserve(this->record_variables.size());
    for (RecordVariable &rv : this->record_variables) {
        replay_variables.push_back(ReplayVariable(rv));
    }

    // Populate with input variables
    for (uint32_t i = 0; i < this->inputs.size(); ++i) {
        Variable *input_variable = jitc_var(replay_inputs[i]);
        ReplayVariable &rv = replay_variables[this->inputs[i]];
        rv.size = input_variable->size;
        rv.data = input_variable->data;
        jitc_log(LogLevel::Info, "input: var(%u) (input %u) -> slot(%u)",
                 replay_inputs[i], i, this->inputs[i]);
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
            uint32_t launch_size = 0;
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
                        launch_size = std::max(launch_size, rv.size);
                    else
                        ptr_size = std::max(launch_size, rv.size);
                }
            }
            if (op.input_size > 0) {
                // If we only have pointer inputs
                if (op.size > op.input_size) {
                    jitc_assert(op.size % op.input_size == 0,
                                "replay(): Could not infer launch size, from "
                                "pointer inputs!");
                    size_t ratio = op.size / op.input_size;
                    jitc_log(LogLevel::Warn,
                             "replay(): Inferring launch size by heuristic, "
                             "launch_size(%u) = ptr_size(%u) * ratio(%zu)",
                             launch_size, ptr_size, ratio);
                    launch_size = ptr_size * ratio;
                } else {
                    jitc_assert(op.input_size % op.size == 0,
                                "replay(): Could not infer launch size, from "
                                "pointer inputs!");

                    uint32_t fraction = op.input_size / op.size;
                    jitc_log(LogLevel::Warn,
                             "replay(): Inferring launch size by heuristic, "
                             "launch_size(%u) = ptr_size(%u) fraction(%u)",
                             launch_size, ptr_size, fraction);
                    launch_size = ptr_size / fraction;
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
                         " -> param slot(%u, rc=%u, is_pointer=%u, "
                         "is_output=%u)",
                         info.index, rv.rc, info.pointer_access,
                         info.type == ParamType::Output);

                if (info.type == ParamType::Output) {
                    rv.size = launch_size;
                    rv.alloc(backend);
                }
                jitc_log(LogLevel::Info, "    data=%p", rv.data);
                jitc_assert(
                    rv.data,
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
                scheduled_tasks.push_back(
                    ts->launch(kernel, launch_size, &kernel_params, &tmp));
            }

        }

        break;
        case OpType::Barrier:

            // Synchronize tasks
            if (this->backend == JitBackend::LLVM) {
                if (scheduled_tasks.size() == 1) {
                    task_release(jitc_task);
                    jitc_task = scheduled_tasks[0];
                } else {
                    jitc_assert(!scheduled_tasks.empty(),
                                "jit_eval(): no tasks generated!");

                    // Insert a barrier task
                    Task *new_task =
                        task_submit_dep(nullptr, scheduled_tasks.data(),
                                        (uint32_t)scheduled_tasks.size());
                    task_release(jitc_task);
                    for (Task *t : scheduled_tasks)
                        task_release(t);
                    jitc_task = new_task;
                }
            }
            scheduled_tasks.clear();

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
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo ptr_info = this->dependencies[dependency_index];
            ParamInfo out_info = this->dependencies[dependency_index + 1];

            ReplayVariable &ptr_var = replay_variables[ptr_info.index];
            ReplayVariable &out_var = replay_variables[out_info.index];

            // Allocate output variable if data is missing.
            if (out_var.data == nullptr) {
                uint32_t dsize = out_var.size * type_size[(int)out_var.type];
                AllocType alloc_type = this->backend == JitBackend::CUDA
                                           ? AllocType::Device
                                           : AllocType::Host;

                out_var.data = jitc_malloc(alloc_type, dsize);
            }

            VarType type = ptr_var.type;
            ReduceOp rtype = op.rtype;

            ts->reduce(type, rtype, ptr_var.data, ptr_var.size, out_var.data);
        } break;
        case OpType::PrefixSum: {
            uint32_t dependency_index = op.dependency_range.first;

            ParamInfo in_info = this->dependencies[dependency_index];
            ParamInfo out_info = this->dependencies[dependency_index + 1];

            ReplayVariable &in_var = replay_variables[in_info.index];
            ReplayVariable &out_var = replay_variables[out_info.index];

            // Allocate output variable if data is missing.
            if (out_var.data == nullptr) {
                uint32_t dsize = out_var.size * type_size[(int)out_var.type];
                AllocType alloc_type = this->backend == JitBackend::CUDA
                                           ? AllocType::Device
                                           : AllocType::Host;

                out_var.data = jitc_malloc(alloc_type, dsize);
            }

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

            jitc_log(LogLevel::Info,
                     "replay(): MemcpyAsync slot(%u) <- slot(%u)",
                     dst_info.index, src_info.index);

            dst_var.size = src_var.size;
            dst_var.alloc(backend);

            jitc_log(LogLevel::Info, "    src.data=%p", src_var.data);
            jitc_log(LogLevel::Info, "    dst.data=%p", dst_var.data);

            size_t size = src_var.size * type_size[(uint32_t)src_var.type];

            ts->memcpy_async(dst_var.data, src_var.data, size);
        } break;
        default:
            jitc_fail("An operation has been recorded, that is not known to "
                      "the replay functionality!");
            break;
        }

        // Only Kernel launches are synchronized
        if (op.type != OpType::KernelLaunch) {
            // Free unused memory, allocated since last barrier
            for (uint32_t j = last_free; j < i; ++j) {
                jitc_log(LogLevel::Info, "replay(): gc for operation %u", j);
                Operation &op = this->operations[j];
                for (uint32_t p = op.dependency_range.first;
                     p < op.dependency_range.second; ++p) {
                    ParamInfo &info = this->dependencies[p];
                    ReplayVariable &rv = replay_variables[info.index];
                    rv.rc--;
                    jitc_log(LogLevel::Info,
                             "replay(): decrement rc for slot %u new rc=%u",
                             info.index, rv.rc);
                    if (rv.rc == 0) {
                        jitc_log(LogLevel::Info,
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
        jitc_log(LogLevel::Info, "replay(): rv(%u, rc=%u, is_input=%u)", i,
                 rv.rc, rv.rv_type == RecordType::Input);
    }

    // Create output variables
    for (uint32_t i = 0; i < this->outputs.size(); ++i) {
        uint32_t index = this->outputs[i];
        jitc_log(LogLevel::Info, "replay(): output: %u", index);
        ReplayVariable &rv = replay_variables[index];
        if (rv.rv_type == RecordType::Input) {
            // Use input variable
            jitc_log(LogLevel::Info,
                     "replay(): output %u at slot %u uses input %u", i, index,
                     rv.index);
            jitc_assert(rv.data, "replay(): freed an input variable "
                                 "that is passed through!");
            uint32_t var_index = replay_inputs[rv.index];
            jitc_var_inc_ref(var_index);
            outputs[i] = var_index;
        } else if (rv.rv_type == RecordType::Captured) {
            jitc_log(LogLevel::Info,
                     "replay(): output %u at slot %u uses captured variable %u",
                     i, index, rv.index);
            jitc_assert(rv.data, "replay(): freed an input variable "
                                 "that is passed through!");

            jitc_var_inc_ref(rv.index);

            outputs[i] = rv.index;
        } else {
            jitc_assert(rv.data && rv.rc > 0,
                        "replay(): freed variable used for output.");
            outputs[i] = jitc_var_mem_map(this->backend, rv.type, rv.data,
                                          rv.size, false);
        }
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
