#include "drjit-core/hash.h"
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"
#include "var.h"
#include <cstdint>

// HashMap used to deduplicate variables
using PtrToSlot = tsl::robin_map<const void *, uint32_t, PointerHasher>;

enum class OpType {
    Barrier,
    KernelLaunch,
    MemsetAsync,
    Reduce,
    PrefixSum,
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
    };
    size_t size;
};

struct RecordVariable {
    VarType type;
    uint32_t size;
    uint32_t input_index;
    bool is_input = false;
    bool is_literal = false;
    // Nubmer of operations that reference this variable
    // used to deallocate unused variables during replay.
    uint32_t rc = 0;

    RecordVariable() {
    }
    RecordVariable(VarType type, uint32_t size) : type(type), size(size) {
    }
    RecordVariable(VarType type, uint32_t size, bool is_literal, bool is_input,
                   uint32_t input_index)
        : type(type), size(size), input_index(input_index), is_input(is_input),
          is_literal(is_literal) {
    }
};

struct ParamInfo {
    uint32_t index;
    ParamType type;

    ParamInfo(uint32_t index) : index(index) {
    }
    ParamInfo(uint32_t index, ParamType type) : index(index), type(type) {
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
    void compute_rc();
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
        uint32_t start = this->recording.dependencies.size();

        Operation op;
        op.type = OpType::Barrier;
        op.dependency_range = std::pair(start, start);
        this->recording.operations.push_back(op);

        return this->internal->barrier();
    }

    Task *launch(Kernel kernel, uint32_t size,
                 std::vector<void *> *kernel_params,
                 const std::vector<uint32_t> *kernel_param_ids) override {
        jitc_log(LogLevel::Info, "record(): recording kernel");

        uint32_t start = this->recording.dependencies.size();

        uint32_t kernel_param_offset =
            this->backend == JitBackend::CUDA ? 1 : 3;

        for (uint32_t param_index = 0; param_index < kernel_param_ids->size();
             param_index++) {
            Variable *v = jitc_var(kernel_param_ids->at(param_index));
            
            RecordVariable rv;
            rv.type = (VarType)v->type;
            rv.size = v->size;
            rv.is_literal = v->is_literal();

            uint32_t id = this->get_or_insert_variable(
                kernel_params->at(kernel_param_offset + param_index), rv);
            
            jitc_log(LogLevel::Info,
                     "  -> recording param %u = variable %u at slot %u",
                     param_index, kernel_param_ids->at(param_index), id);
            this->recording.dependencies.push_back(ParamInfo{
                /*index=*/id,
                /*type=*/(ParamType)v->param_type,
            });
        }

        uint32_t end = this->recording.dependencies.size();

        Operation op;
        op.type = OpType::KernelLaunch;
        op.dependency_range = std::pair(start, end);
        op.kernel = kernel;
        op.size = size;
        this->recording.operations.push_back(op);

        // Re-assign optix specific variables to internal thread state since
        // they might have changed
#if defined(DRJIT_ENABLE_OPTIX)
        this->internal->optix_pipeline = this->optix_pipeline;
        this->internal->optix_sbt = this->optix_sbt;
#endif
        return this->internal->launch(kernel, size, kernel_params,
                                      kernel_param_ids);
    }

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override {

        jitc_fail("RecordThreadState::memset_async(): unsupported function recording!");

        uint32_t start = this->recording.dependencies.size();

        // TODO: Insert if missing
        uint32_t ptr_id = this->get_variable(ptr);
        uint32_t src_id = this->get_variable(src);

        this->recording.dependencies.push_back(ptr_id);
        this->recording.dependencies.push_back(src_id);

        uint32_t end = this->recording.dependencies.size();

        Operation op;
        op.type = OpType::MemsetAsync;
        op.dependency_range = std::pair(start, end);
        op.size = size;
        this->recording.operations.push_back(op);

        return this->internal->memset_async(ptr, size, isize, src);
    }

    /// Reduce the given array to a single value
    void reduce(VarType type, ReduceOp rtype, const void *ptr, uint32_t size,
                void *out) override {

        uint32_t start = this->recording.dependencies.size();

        uint32_t ptr_id = this->get_variable(ptr);
        uint32_t out_id = this->get_or_insert_variable(out, RecordVariable{
                                                                /*type=*/type,
                                                                /*size=*/1,
                                                            });

        this->recording.dependencies.push_back(ptr_id);
        this->recording.dependencies.push_back(out_id);

        uint32_t end = this->recording.dependencies.size();

        Operation op;
        op.type = OpType::Reduce;
        op.dependency_range = std::pair(start, end);
        op.rtype = rtype;
        op.size = size;
        this->recording.operations.push_back(op);

        return this->internal->reduce(type, rtype, ptr, size, out);
    }

    /// 'All' reduction for boolean arrays
    bool all(uint8_t *values, uint32_t size) override {
        jitc_fail("RecordThreadState::all(): unsupported function recording!");
        return this->internal->all(values, size);
    }

    /// 'Any' reduction for boolean arrays
    bool any(uint8_t *values, uint32_t size) override {
        jitc_fail("RecordThreadState::any(): unsupported function recording!");
        return this->internal->any(values, size);
    }

    /// Exclusive prefix sum
    void prefix_sum(VarType vt, bool exclusive, const void *in, uint32_t size,
                    void *out) override {

        uint32_t start = this->recording.dependencies.size();

        uint32_t in_id = this->get_variable(in);
        uint32_t out_id = this->get_or_insert_variable(out, RecordVariable{
                                                                /*type=*/vt,
                                                                /*size=*/size,
                                                            });

        this->recording.dependencies.push_back(in_id);
        this->recording.dependencies.push_back(out_id);

        uint32_t end = this->recording.dependencies.size();

        Operation op;
        op.type = OpType::PrefixSum;
        op.dependency_range = std::pair(start, end);
        op.exclusive = exclusive;
        op.size = size;
        this->recording.operations.push_back(op);

        return this->internal->prefix_sum(vt, exclusive, in, size, out);
    }

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size,
                      uint32_t *out) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::compress(): unsupported function recording!");
        return this->internal->compress(in, size, out);
    }

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t mkperm(const uint32_t *values, uint32_t size,
                    uint32_t bucket_count, uint32_t *perm,
                    uint32_t *offsets) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::mkperm(): unsupported function recording!");
        return this->internal->mkperm(values, size, bucket_count, perm,
                                      offsets);
    }

    /// Perform a synchronous copy operation
    void memcpy(void *dst, const void *src, size_t size) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::memcpy(): unsupported function recording!");
        return this->internal->memcpy(dst, src, size);
    }

    /// Perform an assynchronous copy operation
    void memcpy_async(void *dst, const void *src, size_t size) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::memcpy_async(): unsupported function recording!");
        return this->internal->memcpy_async(dst, src, size);
    }

    /// Sum over elements within blocks
    void block_reduce(VarType type, ReduceOp op, const void *in, uint32_t size,
                      uint32_t block_size, void *out) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::block_reduce(): unsupported function recording!");
        return this->internal->block_reduce(type, op, in, size, block_size,
                                            out);
    }

    /// Compute a dot product of two equal-sized arrays
    void reduce_dot(VarType type, const void *ptr_1, const void *ptr_2,
                    uint32_t size, void *out) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::reduce_dot(): unsupported function recording!");
        return this->internal->reduce_dot(type, ptr_1, ptr_2, size, out);
    }

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::poke(): unsupported function recording!");
        return this->internal->poke(dst, src, size);
    }

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::aggregate(): unsupported function recording!");
        return this->internal->aggregate(dst, agg, size);
    }

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override {
        return this->internal->enqueue_host_func(callback, payload);
    }

    /// LLVM: reduce a variable that was previously expanded due to
    /// dr.ReduceOp.Expand
    void reduce_expanded(VarType vt, ReduceOp op, void *data, uint32_t exp,
                         uint32_t size) override {
        jitc_log(LogLevel::Warn, "RecordThreadState::reduce_expanded(): unsupported function recording!");
        return this->internal->reduce_expanded(vt, op, data, exp, size);
    }

    ~RecordThreadState() {
    }

    void set_input(uint32_t input) {
        uint32_t input_index = this->recording.inputs.size();
        Variable *v = jitc_var(input);
        uint32_t slot = this->get_or_insert_variable(
            v->data, RecordVariable{
                         /*type=*/(VarType)v->type,
                         /*size=*/v->size,
                         /*is_literal=*/v->is_literal(),
                         /*is_input=*/true,
                         /*input_index=*/input_index,
                     });
        jitc_log(LogLevel::Info,
                 "record(): Adding variable %u input %u to slot %u", input,
                 input_index, slot);
        this->recording.inputs.push_back(slot);
    }
    void set_output(uint32_t output) {
        uint32_t output_index = this->recording.outputs.size();
        Variable *v = jitc_var(output);
        uint32_t slot = this->get_or_insert_variable(
            v->data, RecordVariable{
                         /*type=*/(VarType)v->type,
                         /*size=*/v->size,
                         /*is_literal=*/v->is_literal(),
                         /*is_input=*/false,
                         /*input_index=*/0,
                     });

        jitc_log(LogLevel::Info,
                 "record(): Adding variable %u output %u to slot %u", output,
                 output_index, slot);
        this->recording.outputs.push_back(slot);
    }

    ThreadState *internal;

    Recording recording;

  private:
    // Mapping from data pointer of a variable to a index into the slot of the
    // recording.
    PtrToSlot ptr_to_slot;

    // Insert the variable by pointer, deduplicating it and returning a index
    // into the `variables` field of the recording.
    uint32_t get_or_insert_variable(void *ptr, RecordVariable rv) {

        auto it = this->ptr_to_slot.find(ptr);

        if (it == this->ptr_to_slot.end()) {
            uint32_t slot = this->recording.record_variables.size();

            this->recording.record_variables.push_back(rv);

            this->ptr_to_slot.insert({ptr, slot});

            return slot;
        } else {
            uint32_t slot = it.value();

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
};

void jitc_record_start(JitBackend backend, const uint32_t *inputs,
                       uint32_t n_inputs);

Recording *jitc_record_stop(JitBackend backend, const uint32_t *outputs,
                            uint32_t n_outputs);
