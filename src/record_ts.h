#include "drjit-core/hash.h"
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include <algorithm>
#include <cstdint>

// HashMap used to deduplicate variables
using PtrToSlot = tsl::robin_map<const void *, uint32_t, PointerHasher>;

enum class OpType {
    Barrier,
    KernelLaunch,
    MemsetAsync,
    Reduce,
    Expand,
    ReduceExpanded,
    PrefixSum,
    Compress,
    MemcpyAsync,
    Mkperm,
    Aggregate,
};

struct Operation {
    OpType type;
    // Indices into the dependencies vector
    std::pair<uint32_t, uint32_t> dependency_range;
    // Kernel hash if a kernel was launched
    union {
        Kernel kernel;
        ReduceOp rtype;
        bool exclusive;
        uint32_t bucket_count;
        uint64_t data;
    };
    size_t size;
    size_t input_size = 0;
    bool enabled = true;
};

/// Denotes the type of variable.
///
/// Output variables are only tracked through the outputs array, as this
/// information is only needed when constructing the output variables.
///
enum class RecordType {
    Other,
    Input,
    Captured,
};

struct RecordVariable {
    VarType type = VarType::Void;
    /// Stores index into input array if variable is input or index of captured
    /// variable
    uint32_t index;
    bool is_literal = false;
    RecordType rv_type = RecordType::Other;
    // Nubmer of operations that reference this variable
    // used to deallocate unused variables during replay.
    uint32_t rc = 0;
    uint32_t last_memset = 0;
    uint32_t last_memcpy = 0;

    RecordVariable() {
    }
    RecordVariable(VarType type) : type(type) {
    }
    RecordVariable(VarType type, bool is_literal, RecordType rv_type,
                   uint32_t input_index)
        : type(type), index(input_index), is_literal(is_literal),
          rv_type(rv_type) {
    }

    /**
     * Not all information about variables might be known right away (see
     * memcpy). When new information about the variable is available, we can add
     * it to the already saved RecordVariable.
     */
    RecordVariable &operator|=(const RecordVariable &rhs) {
        if (this->type == VarType::Void)
            this->type = rhs.type;
        else
            jitc_assert(this->type == rhs.type,
                        "record(): Missmatched types during update of "
                        "RecordVariable, %s != %s",
                        type_name[(uint32_t)this->type],
                        type_name[(uint32_t)rhs.type]);
        if (this->rv_type == RecordType::Other) {
            this->rv_type = rhs.rv_type;
            this->index = rhs.index;
        }
        this->is_literal |= rhs.is_literal;
        this->last_memcpy = rhs.last_memcpy;
        this->last_memset = rhs.last_memset;
        return *this;
    }
    bool compatible(const RecordVariable &rhs) {
        bool result = true;
        result &= this->rv_type == RecordType::Other ||
                  rhs.rv_type == RecordType::Other ||
                  this->rv_type == rhs.rv_type;
        result &= this->type == VarType::Void || rhs.type == VarType::Void ||
                  this->type == rhs.type;
        return result;
    }
};

struct ParamInfo {
    uint32_t slot;
    ParamType type = ParamType::Input;
    bool pointer_access;
    struct {
        uint32_t offset;
        uint64_t data;
        int32_t type_size;
    } extra;

    ParamInfo() {
    }
    ParamInfo(uint32_t index) : slot(index) {
    }
    ParamInfo(uint32_t index, ParamType type, bool pointer_access)
        : slot(index), type(type), pointer_access(pointer_access) {
    }
};

struct Recording {

    std::vector<RecordVariable> record_variables;

    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;

    std::vector<Operation> operations;
    std::vector<ParamInfo> dependencies;

    JitBackend backend;

    void replay(const uint32_t *replay_input, uint32_t *outputs);

    /// Computes the initial reference count for replay variables
    void compute_rc();
    void validate();
};

struct RecordThreadState : ThreadState {

    RecordThreadState(ThreadState *internal) {
        this->context = internal->context;
        this->stream = internal->stream;
        this->event = internal->event;
        this->sync_stream_event = internal->sync_stream_event;
        this->device = internal->device;
        this->compute_capability = internal->compute_capability;
        this->ptx_version = internal->ptx_version;
        this->memory_pool = internal->memory_pool;

        this->backend = internal->backend;
        this->scope = internal->scope;
        this->call_self_value = internal->call_self_value;
        this->call_self_index = internal->call_self_index;

#if defined(DRJIT_ENABLE_OPTIX)
        this->optix_pipeline = internal->optix_pipeline;
        this->optix_sbt = internal->optix_sbt;
#endif

        this->internal = internal;

        this->recording.backend = internal->backend;

        this->scope = internal->scope;
    };

    void barrier() override {
        if (!paused) {
            uint32_t start = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::Barrier;
            op.dependency_range = std::pair(start, start);
            this->recording.operations.push_back(op);
        }

        scoped_pause();
        return this->internal->barrier();
    }

    Task *launch(Kernel kernel, uint32_t size,
                 std::vector<void *> *kernel_params,
                 const std::vector<uint32_t> *kernel_param_ids) override {
        if (!paused) {
            uint32_t kernel_param_offset =
                this->backend == JitBackend::CUDA ? 1 : 3;

            size_t input_size = 0;
            size_t ptr_size = 0;

            // Handle reduce_expanded case
            for (uint32_t param_index = 0;
                 param_index < kernel_param_ids->size(); param_index++) {
                uint32_t index = kernel_param_ids->at(param_index);
                Variable *v = jitc_var(index);
                ParamType param_type = (ParamType)v->param_type;
                if ((VarType)v->type == VarType::Pointer) {
                    // Follow pointer
                    index = v->dep[3];
                    v = jitc_var(index);
                }

                if (param_type == ParamType::Input && v->reduce_op) {
                    uint32_t dst_slot = get_variable(v->data);
                    const RecordVariable &rv =
                        this->recording.record_variables[dst_slot];
                    Operation &memset =
                        this->recording.operations[rv.last_memset];
                    Operation &memcpy =
                        this->recording.operations[rv.last_memcpy];
                    memset.enabled = false;
                    memcpy.enabled = false;

                    uint32_t dependency_index = memcpy.dependency_range.first;
                    ParamInfo src_info =
                        this->recording.dependencies[dependency_index];

                    uint32_t start = this->recording.dependencies.size();
                    this->recording.dependencies.push_back(src_info.slot);
                    this->recording.dependencies.push_back(dst_slot);
                    uint32_t end = this->recording.dependencies.size();

                    jitc_log(LogLevel::Debug,
                             "record(): expand(dst=s%u, src=s%u)", dst_slot,
                             src_info.slot);
                    Operation op;
                    op.type = OpType::Expand;
                    op.dependency_range = std::pair(start, end);
                    op.data = memset.data;
                    this->recording.operations.push_back(op);
                }
            }

            jitc_log(LogLevel::Info, "record(): recording kernel");

            uint32_t start = this->recording.dependencies.size();
            for (uint32_t param_index = 0;
                 param_index < kernel_param_ids->size(); param_index++) {

                bool pointer_access = false;
                uint32_t index = kernel_param_ids->at(param_index);
                Variable *v = jitc_var(index);

                // Note, the ptr might not come from the variable but the
                // `ScheduledVariable` if it is an output.
                void *ptr =
                    kernel_params->at(kernel_param_offset + param_index);
                ParamType param_type = (ParamType)v->param_type;

                if (param_type == ParamType::Input &&
                    (VarType)v->type != VarType::Pointer) {
                    input_size = std::max(input_size, (size_t)v->size);
                }

                // In case the variable is a pointer, we follow the pointer to
                // the source and record the source size.
                // NOTE: this means that `v` is now the source variable
                if ((VarType)v->type == VarType::Pointer) {
                    jitc_assert(v->is_literal(),
                                "record(): Recording non-literal pointers are "
                                "not yet supported!");
                    jitc_assert(param_type != ParamType::Output,
                                "record(): A pointer, pointing to a kernel "
                                "ouptut is not yet supported!");

                    // Follow pointer
                    index = v->dep[3];
                    v = jitc_var(index);

                    pointer_access = true;
                    ptr_size = std::max(ptr_size, (size_t)v->size);
                }

                RecordVariable rv;
                rv.type = (VarType)v->type;
                rv.is_literal = v->is_literal();

                // It could happen, that a variable is created while recording.
                // This occurs for example, when recording vcalls, where a
                // `offset` buffer is created.
                // Those variables are captured here and kept for replay.
                if (param_type == ParamType::Input && !has_variable(ptr)) {
                    if (v->scope < this->internal->scope) {
                        jitc_raise(
                            "record(): Variable %u -> %p, was created before "
                            "recording was started, but it was not speciefied "
                            "as an input variable!",
                            index, ptr);
                    }

                    jitc_log(LogLevel::Warn,
                             "record(): Variable r%u(scope=%u >= "
                             "original_scope=%u) -> %p appeared out of "
                             "nowhere! It will be captured.",
                             index, v->scope, this->internal->scope, ptr);
                    rv.rv_type = RecordType::Captured;
                    rv.index = index;
                    jitc_var_inc_ref(index);
                }

                uint32_t slot;
                if (param_type == ParamType::Input)
                    slot = this->add_variable(ptr, rv);
                else if (param_type == ParamType::Output)
                    slot = this->add_variable(ptr, rv);
                else
                    jitc_fail("Parameter Type not supported!");

                if (pointer_access) {
                    jitc_log(LogLevel::Debug,
                             " %s recording param %u = var(%u, points to r%u, "
                             "size=%u, data=%p) at slot(%u)",
                             param_type == ParamType::Output ? "<-" : "->",
                             param_index, kernel_param_ids->at(param_index),
                             index, v->size, ptr, slot);
                } else {
                    jitc_log(LogLevel::Debug,
                             " %s recording param %u = var(%u, size=%u, "
                             "data=%p) at slot(%u)",
                             param_type == ParamType::Output ? "<-" : "->",
                             param_index, kernel_param_ids->at(param_index),
                             v->size, ptr, slot);
                }

                this->recording.dependencies.push_back(ParamInfo{
                    /*index=*/slot,
                    /*type=*/param_type,
                    /*pointer_access=*/pointer_access,
                });
            }
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::KernelLaunch;
            op.dependency_range = std::pair(start, end);
            op.kernel = kernel;
            op.size = size;

            // Record max_input_size if we have only pointer inputs.
            // Therefore, if max_input_size > 0 we know this at replay.
            if (input_size == 0) {
                jitc_log(LogLevel::Info, "    input_size(pointers)=%zu",
                         input_size);
                op.input_size = ptr_size;
            } else if (input_size != size) {
                jitc_log(LogLevel::Info, "    input_size(direct)=%zu",
                         input_size);
                op.input_size = input_size;
            }

            this->recording.operations.push_back(op);

            // Re-assign optix specific variables to internal thread state since
            // they might have changed
#if defined(DRJIT_ENABLE_OPTIX)
            this->internal->optix_pipeline = this->optix_pipeline;
            this->internal->optix_sbt = this->optix_sbt;
#endif
        }
        scoped_pause();
        return this->internal->launch(kernel, size, kernel_params,
                                      kernel_param_ids);
    }

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override {

        if (!paused) {
            jitc_log(LogLevel::Debug,
                     "record(): memset_async(ptr=%p, size=%u, "
                     "isize=%u, src=%p)",
                     ptr, size, isize, src);
            jitc_assert(isize <= 8,
                        "record(): Tried to call memset_async with isize=%u, "
                        "only isize<=8 is supported!",
                        isize);

            RecordVariable rv;
            rv.last_memset = this->recording.operations.size();
            uint32_t ptr_id = this->add_variable(ptr, rv);

            uint32_t start = this->recording.dependencies.size();
            this->recording.dependencies.push_back(ptr_id);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::MemsetAsync;
            op.dependency_range = std::pair(start, end);
            op.size = size;
            std::memcpy(&op.data, src, isize);

            this->recording.operations.push_back(op);
        }

        scoped_pause();
        return this->internal->memset_async(ptr, size, isize, src);
    }

    /// Reduce the given array to a single value
    void reduce(VarType type, ReduceOp rtype, const void *ptr, uint32_t size,
                void *out) override {
        if (!paused) {

            uint32_t ptr_id = this->get_variable(ptr);
            uint32_t out_id = this->add_variable(out, RecordVariable{
                                                          /*type=*/type,
                                                      });

            uint32_t start = this->recording.dependencies.size();
            this->recording.dependencies.push_back(ptr_id);
            this->recording.dependencies.push_back(out_id);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::Reduce;
            op.dependency_range = std::pair(start, end);
            op.rtype = rtype;
            op.size = size;
            this->recording.operations.push_back(op);
        }

        scoped_pause();
        return this->internal->reduce(type, rtype, ptr, size, out);
    }

    /// 'All' reduction for boolean arrays
    bool all(uint8_t *values, uint32_t size) override {
        jitc_log(LogLevel::Warn,
                 "RecordThreadState::all(): unsupported function recording!");
        scoped_pause();
        return this->internal->all(values, size);
    }

    /// 'Any' reduction for boolean arrays
    bool any(uint8_t *values, uint32_t size) override {
        jitc_log(LogLevel::Warn,
                 "RecordThreadState::any(): unsupported function recording!");
        scoped_pause();
        return this->internal->any(values, size);
    }

    /// Exclusive prefix sum
    void prefix_sum(VarType vt, bool exclusive, const void *in, uint32_t size,
                    void *out) override {
        if (!paused) {
            uint32_t in_id = this->get_variable(in);
            uint32_t out_id = this->add_variable(out, RecordVariable{
                                                          /*type=*/vt,
                                                      });

            uint32_t start = this->recording.dependencies.size();
            this->recording.dependencies.push_back(in_id);
            this->recording.dependencies.push_back(out_id);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::PrefixSum;
            op.dependency_range = std::pair(start, end);
            op.exclusive = exclusive;
            op.size = size;
            this->recording.operations.push_back(op);
        }
        scoped_pause();
        return this->internal->prefix_sum(vt, exclusive, in, size, out);
    }

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size,
                      uint32_t *out) override {

        if (!paused) {
            jitc_assert(has_variable(in),
                        "record(): Input variable has not been recorded!");
            jitc_log(LogLevel::Debug,
                     "record(): compress(in=%p, size=%u, out=%p)", in, size,
                     out);

            RecordVariable rv;
            rv.type = VarType::Bool;
            uint32_t in_slot = this->add_variable((void *)in, rv);
            rv = RecordVariable();
            rv.type = VarType::UInt32;
            uint32_t out_slot = this->add_variable((void *)out, rv);

            uint32_t start = this->recording.dependencies.size();
            this->recording.dependencies.push_back(in_slot);
            this->recording.dependencies.push_back(out_slot);
            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::Compress;
            op.dependency_range = std::pair(start, end);
            op.size = size;
            this->recording.operations.push_back(op);
        }

        scoped_pause();
        return this->internal->compress(in, size, out);
    }

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t mkperm(const uint32_t *values, uint32_t size,
                    uint32_t bucket_count, uint32_t *perm,
                    uint32_t *offsets) override {
        if (!paused) {
            if (has_variable(values)) {
                jitc_log(LogLevel::Debug,
                         "record(): mkperm(values=%p, size=%u, "
                         "bucket_count=%u, perm=%p, offsets=%p)",
                         values, size, bucket_count, perm, offsets);

                uint32_t values_id = this->get_variable(values);
                uint32_t perm_id =
                    this->add_variable(perm, RecordVariable{
                                                 /*type=*/VarType::UInt32,
                                             });
                uint32_t offsets_id =
                    this->add_variable(offsets, RecordVariable{
                                                    /*type=*/VarType::UInt32,
                                                });

                uint32_t start = this->recording.dependencies.size();
                this->recording.dependencies.push_back(values_id);
                this->recording.dependencies.push_back(perm_id);
                this->recording.dependencies.push_back(offsets_id);
                uint32_t end = this->recording.dependencies.size();

                Operation op;
                op.type = OpType::Mkperm;
                op.dependency_range = std::pair(start, end);
                op.size = size;
                op.bucket_count = bucket_count;
                this->recording.operations.push_back(op);
            }
        }
        scoped_pause();
        return this->internal->mkperm(values, size, bucket_count, perm,
                                      offsets);
    }

    /// Perform a synchronous copy operation
    void memcpy(void *dst, const void *src, size_t size) override {
        jitc_log(
            LogLevel::Warn,
            "RecordThreadState::memcpy(): unsupported function recording!");
        scoped_pause();
        return this->internal->memcpy(dst, src, size);
    }

    /// Perform an assynchronous copy operation
    void memcpy_async(void *dst, const void *src, size_t size) override {
        if (!paused) {
            if (has_variable(src)) {
                jitc_log(LogLevel::Debug,
                         "record(): memcpy_async(dst=%p, src=%p, size=%zu)",
                         dst, src, size);

                uint32_t src_id = this->get_variable(src);
                // Add an empty RecordVariable and hope that it will get filled
                // in by a kernel invocation.
                RecordVariable rv;
                rv.last_memcpy = this->recording.operations.size();
                uint32_t dst_id = this->add_variable(dst, rv);

                uint32_t start = this->recording.dependencies.size();
                this->recording.dependencies.push_back(src_id);
                this->recording.dependencies.push_back(dst_id);
                uint32_t end = this->recording.dependencies.size();

                Operation op;
                op.type = OpType::MemcpyAsync;
                op.dependency_range = std::pair(start, end);
                op.size = size;
                this->recording.operations.push_back(op);
            }
        }
        scoped_pause();
        return this->internal->memcpy_async(dst, src, size);
    }

    /// Sum over elements within blocks
    void block_reduce(VarType type, ReduceOp op, const void *in, uint32_t size,
                      uint32_t block_size, void *out) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::block_reduce(): "
                                 "unsupported function recording!");
        scoped_pause();
        return this->internal->block_reduce(type, op, in, size, block_size,
                                            out);
    }

    /// Compute a dot product of two equal-sized arrays
    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override {
        jitc_log(
            LogLevel::Warn,
            "RecordThreadState::reduce_dot(): unsupported function recording!");
        scoped_pause();
        return this->internal->reduce_dot(type, ptr_1, ptr_2, size, out);
    }

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override {
        jitc_log(LogLevel::Warn,
                 "RecordThreadState::poke(): unsupported function recording!");
        scoped_pause();
        return this->internal->poke(dst, src, size);
    }

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override {
        if (!paused) {
            jitc_log(LogLevel::Debug, "record(): aggregate(dst=%p, size=%u)",
                     dst, size);

            uint32_t dst_id =
                this->add_variable(dst, RecordVariable{
                                            /*type=*/VarType::UInt8,
                                        });

            jitc_log(LogLevel::Debug, " <- slot(%u)", dst_id);

            uint32_t start = this->recording.dependencies.size();

            this->recording.dependencies.push_back(ParamInfo{
                /*index=*/dst_id,
                /*type=*/ParamType::Output,
                /*pointer_access=*/false,
            });

            for (uint32_t i = 0; i < size; ++i) {
                AggregationEntry &p = agg[i];

                jitc_log(LogLevel::Debug, " -> entry(src=%p)", p.src);

                if (!has_variable(p.src)) {
                    // Literal
                    jitc_log(LogLevel::Debug, "    literal");
                    ParamInfo info;
                    std::memcpy(&info.extra.data, &p.src, sizeof(uint64_t));
                    info.extra.offset = p.offset;
                    info.extra.type_size = p.size;
                    info.type = ParamType::Register;
                    info.pointer_access = false;
                    this->recording.dependencies.push_back(info);
                } else {
                    uint32_t index = get_variable(p.src);
                    jitc_log(LogLevel::Debug, "    ptr at slot %u", index);

                    RecordVariable &rv =
                        this->recording.record_variables[index];

                    ParamInfo info;
                    info.slot = index;
                    info.type = ParamType::Input;
                    info.pointer_access = rv.type == VarType::Pointer;
                    info.extra.offset = p.offset;
                    this->recording.dependencies.push_back(info);
                }
            }

            uint32_t end = this->recording.dependencies.size();

            Operation op;
            op.type = OpType::Aggregate;
            op.dependency_range = std::pair(start, end);
            op.size = size;
            this->recording.operations.push_back(op);
        }
        scoped_pause();
        this->internal->aggregate(dst, agg, size);
    }

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override {
        scoped_pause();
        return this->internal->enqueue_host_func(callback, payload);
    }

    /// LLVM: reduce a variable that was previously expanded due to
    /// dr.ReduceOp.Expand
    void reduce_expanded(VarType vt, ReduceOp reduce_op, void *data,
                         uint32_t exp, uint32_t size) override {

        if (!paused) {
            jitc_log(LogLevel::Debug,
                     "record(): reduce_expanded(vt=%u, op=%u, data=%p, exp=%u, "
                     "size=%u)",
                     (uint32_t)vt, (uint32_t)reduce_op, data, exp, size);

            uint32_t data_id =
                this->add_variable(data, RecordVariable{/*type=*/vt});

            uint32_t start = this->recording.dependencies.size();
            this->recording.dependencies.push_back(data_id);
            uint32_t end = this->recording.dependencies.size();

            jitc_log(LogLevel::Debug, "<-> data: slot(%u)", data_id);

            Operation op;
            op.type = OpType::ReduceExpanded;
            op.dependency_range = std::pair(start, end);
            op.rtype = reduce_op;
            op.size = size;
            this->recording.operations.push_back(op);
        }
        scoped_pause();
        return this->internal->reduce_expanded(vt, reduce_op, data, exp, size);
    }

    ~RecordThreadState() {
    }

    void add_input(uint32_t input) {
        uint32_t input_index = this->recording.inputs.size();
        Variable *v = jitc_var(input);
        uint32_t slot =
            this->add_variable(v->data, RecordVariable{
                                            /*type=*/(VarType)v->type,
                                            /*is_literal=*/v->is_literal(),
                                            /*rv_type=*/RecordType::Input,
                                            /*input_index=*/input_index,
                                        });
        jitc_log(LogLevel::Info,
                 "record(): Adding variable %u input %u to slot %u", input,
                 input_index, slot);
        this->recording.inputs.push_back(slot);
    }
    void add_output(uint32_t output) {
        uint32_t output_index = this->recording.outputs.size();
        Variable *v = jitc_var(output);
        uint32_t slot =
            this->add_variable(v->data, RecordVariable{
                                            /*type=*/(VarType)v->type,
                                            /*is_literal=*/v->is_literal(),
                                            /*rv_type=*/RecordType::Other,
                                            /*input_index=*/0,
                                        });

        jitc_log(LogLevel::Trace,
                 "record(): Adding variable %u output %u to slot %u", output,
                 output_index, slot);
        this->recording.outputs.push_back(slot);
    }

    bool pause() {
        bool tmp = paused;
        paused = true;
        return tmp;
    }
    bool resume() {
        bool tmp = paused;
        paused = false;
        return tmp;
    }

    struct pause_scope {
        RecordThreadState *rts;
        bool tmp;

        pause_scope(RecordThreadState *rts) : rts(rts), tmp(rts->pause()) {
        }
        ~pause_scope() {
            rts->paused = tmp;
        }
    };

    pause_scope scoped_pause() {
        return pause_scope(this);
    }

    bool paused = false;

    ThreadState *internal;

    Recording recording;

  private:
    // Mapping from data pointer of a variable to a index into the slot of the
    // recording.
    PtrToSlot ptr_to_slot;

    /**
     * Add information about a variable, deduplicating it and returning the slot
     * in the `variables` field of the recording.
     * Information is combined when the variable has already been added.
     * This is used by the input variables of a kernel.
     */
    uint32_t add_variable(void *ptr, RecordVariable rv) {

        auto it = this->ptr_to_slot.find(ptr);

        if (it == this->ptr_to_slot.end()) {
            uint32_t slot = this->recording.record_variables.size();

            this->recording.record_variables.push_back(rv);

            this->ptr_to_slot.insert({ptr, slot});

            return slot;
        } else {
            uint32_t slot = it.value();

            RecordVariable &old = this->recording.record_variables[slot];
            if (!old.compatible(rv)) {
                // If two record variables are not compatible, we have to create
                // a new one. Otherwise information about the new/old one might
                // get lost.
                slot = this->recording.record_variables.size();
                jitc_log(LogLevel::Debug,
                         "record(): adding new variable at slot %u", slot);
                this->recording.record_variables.push_back(rv);
                it.value() = slot;
            } else {
                this->recording.record_variables[slot] |= rv;
            }

            return slot;
        }
    }

    // Return the slot index given the data pointer of a variable.
    // This fails if the variable has not been added.
    uint32_t get_variable(const void *ptr) {
        auto it = this->ptr_to_slot.find(ptr);

        jitc_assert(it != this->ptr_to_slot.end(),
                    "Failed to find the slot corresponding to the variable "
                    "with data at %p",
                    ptr);

        return it.value();
    }

    bool has_variable(const void *ptr) {
        auto it = this->ptr_to_slot.find(ptr);

        return it != this->ptr_to_slot.end();
    }
};

void jitc_record_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs);

Recording *jitc_record_stop(JitBackend backend, const uint32_t *outputs,
                            uint32_t n_outputs);

void jitc_record_abort(JitBackend backend);

void jitc_record_destroy(Recording *recording);

bool jitc_record_pause(JitBackend backend);

bool jitc_record_resume(JitBackend backend);
