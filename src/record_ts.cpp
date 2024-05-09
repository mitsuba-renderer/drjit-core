#include "record_ts.h"
#include "common.h"
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"

// This struct holds the data and tracks the size of varaibles,
// used during replay.
struct ReplayVariable {
    void *data = 0;
    uint32_t size = 0;
    VarType type;
    uint32_t input_index;
    bool is_input;

    ReplayVariable(RecordVariable &rv) {
        this->type = rv.type;
        this->size = rv.size;
        this->input_index = rv.input_index;
        this->is_input = rv.is_input;
    }
};

/// Kernel parameter buffer
static std::vector<void *> kernel_params;

/// Temporary scratch space for scheduled tasks (LLVM only)
static std::vector<Task *> scheduled_tasks;

/// Temporary variables used for replaying a recording.
static std::vector<ReplayVariable> replay_variables;

void Recording::replay(const uint32_t *replay_inputs, uint32_t *outputs) {

    ThreadState *ts = thread_state(backend);

    replay_variables.clear();
    scheduled_tasks.clear();

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
    }

    // Execute kernels and allocate missing output variables

    for (uint32_t i = 0; i < this->operations.size(); ++i) {
        Operation &op = this->operations[i];

        switch (op.type) {
        case OpType::KernelLaunch: {
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

            // Inferr kernel size.
            uint32_t launch_size = 0;
            for (uint32_t j = op.dependency_range.first;
                 j < op.dependency_range.second; ++j) {
                ParamInfo info = this->dependencies[j];
                ReplayVariable &rv = replay_variables[info.index];

                if (info.type == ParamType::Input) {
                    jitc_assert(
                        rv.data != nullptr,
                        "replay(): Kernel input variable not allocated!");

                    launch_size = std::max(launch_size, rv.size);
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
            // TODO: deallocate unused memory.
            for (uint32_t j = op.dependency_range.first;
                 j < op.dependency_range.second; ++j) {
                ParamInfo info = this->dependencies[j];
                ReplayVariable &rv = replay_variables[info.index];

                if (info.type == ParamType::Output) {
                    jitc_assert(rv.data == nullptr,
                                "replay(): Output parameters should not be "
                                "allocate before replaying the kernel!");

                    rv.size = launch_size;

                    jitc_log(LogLevel::Info,
                             "replay(): Allocating output variable of size %u.",
                             rv.size);

                    uint32_t dsize = rv.size * type_size[(int)rv.type];

                    AllocType alloc_type = this->backend == JitBackend::CUDA
                                               ? AllocType::Device
                                               : AllocType::Host;

                    rv.data = jitc_malloc(alloc_type, dsize);
                }
                kernel_params.push_back(rv.data);
            }

            // Change kernel size in `kernel_params`
            if (backend == JitBackend::CUDA) {
                uintptr_t size = 0;
                std::memcpy(&size, &launch_size, sizeof(uint32_t));
                kernel_params[0] = (void *)size;
            }

            {
                jitc_log(LogLevel::Info, "replay(): launching kernel (n=%u)",
                         launch_size);
                scoped_set_context_maybe guard2(ts->context);
                std::vector<uint32_t> tmp;
                Kernel kernel = op.kernel;
                scheduled_tasks.push_back(
                    ts->launch(kernel, launch_size, &kernel_params, &tmp));
            }

        }

        break;
        case OpType::Barrier:

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
        default:
            jitc_fail("An operation has been recorded, that is not known to "
                      "the replay functionality!");
            break;
        }
    }

    // Create output variables
    for (uint32_t i = 0; i < this->outputs.size(); ++i) {
        uint32_t index = this->outputs[i];
        ReplayVariable &rv = replay_variables[index];
        if (rv.is_input) {
            // Use input variable
            uint32_t var_index = replay_inputs[rv.input_index];
            jitc_var_inc_ref(var_index);
            outputs[i] = var_index;
        } else {
            Variable v;
            v.kind = VarKind::Evaluated;
            v.type = (uint32_t)rv.type;
            v.size = rv.size;
            v.data = rv.data;
            v.backend = (uint32_t)this->backend;
            outputs[i] = jitc_var_new(v);
        }
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
        record_ts->set_input(inputs[i]);
    }
}
Recording *jitc_record_stop(JitBackend backend, const uint32_t *outputs,
                            uint32_t n_outputs) {
    // ThreadState *ts = thread_state(backend);
    RecordThreadState *ts =
        dynamic_cast<RecordThreadState *>(thread_state(backend));
    ThreadState *internal = ts->internal;

    // Perform reasignments to internal thread-state of possibly changed
    // variables
    internal->scope = ts->scope;

    jitc_assert(ts->record_stack.empty(),
                "Kernel recording ended while still recording loop!");

    for (uint32_t i = 0; i < n_outputs; ++i) {
        ts->set_output(outputs[i]);
    }

    if (backend == JitBackend::CUDA) {
        thread_state_cuda = internal;
    } else {
        thread_state_llvm = internal;
    }
    return new Recording(ts->recording);
}
