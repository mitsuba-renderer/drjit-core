
#include "drjit-core/hash.h"
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"
#include <cstdint>

// HashMap used to deduplicate variables
using VariableCache = tsl::robin_map<void *, uint32_t, PointerHasher>;

enum class OpType{
    KernelLaunch,
};

struct Operation{
    OpType type;
    // Indices into the dependencies vector
    std::pair<uint32_t, uint32_t> dependency_range;
};


struct RecordThreadState: ThreadState{

    RecordThreadState(ThreadState *internal){
        this->context = internal->context;
        this->stream = internal->stream;
        this->event = internal->event;
        this->sync_stream_event = internal->sync_stream_event;
        this->device = internal->device;
        this->compute_capability = compute_capability;
        this->ptx_version = internal->ptx_version;
        this->memory_pool = internal->memory_pool;

#if defined(DRJIT_ENABLE_OPTIX)
        this->optix_pipeline = internal->optix_pipeline;
        this->optix_sbt = internal->optix_sbt;
#endif

    };
    
    Task *launch(Kernel kernel, uint32_t size,
                 std::vector<void *> *kernel_params) override{
        
        uint32_t start = this->dependencies.size();

        uint32_t param_index = this->backend == JitBackend::CUDA ? 1 : 3;
        for (; param_index < kernel_params->size(); ++param_index){
            uint32_t id = this->get_or_insert_variable(kernel_params->at(param_index));
            this->dependencies.push_back(id);
        }

        uint32_t end = this->dependencies.size();

        this->operations.push_back(Operation{
            /*type=*/OpType::KernelLaunch,
            /*dependency_range=*/std::pair(start, end)
        });
        
        return this->internal->launch(kernel, size, kernel_params);
    }

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override{
        return this->internal->memset_async(ptr, size, isize, src);
    }

    /// Reduce the given array to a single value
    void reduce(VarType type, ReduceOp rtype, const void *ptr, uint32_t size,
                void *out) override{
        return this->internal->reduce(type, rtype, ptr, size, out);
    }

    /// 'All' reduction for boolean arrays
    bool all(uint8_t *values, uint32_t size) override{
        return this->internal->all(values, size);
    }

    /// 'Any' reduction for boolean arrays
    bool any(uint8_t *values, uint32_t size) override{
        return this->internal->any(values, size);
    }

    /// Exclusive prefix sum
    void prefix_sum(VarType vt, bool exclusive, const void *in, uint32_t size,
                    void *out) override{
        return this->internal->prefix_sum(vt, exclusive, in ,size, out);
    }

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size, uint32_t *out) override{
        return this->internal->compress(in, size, out);
    }

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t mkperm(const uint32_t *values, uint32_t size, uint32_t bucket_count,
                    uint32_t *perm, uint32_t *offsets) override{
        return this->internal->mkperm(values, size, bucket_count, perm, offsets);
    }

    /// Perform a synchronous copy operation
    void memcpy(void *dst, const void *src, size_t size) override{
        return this->internal->memcpy(dst, src, size);
    }

    /// Perform an assynchronous copy operation
    void memcpy_async(void *dst, const void *src, size_t size) override{
        return this->internal->memcpy_async(dst, src, size);
    }

    /// Sum over elements within blocks
    void block_sum(enum VarType type, const void *in, void *out, uint32_t size,
                   uint32_t block_size) override{
        return this->internal->block_sum(type, in, out, size, block_size);
    }

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override{
        return this->internal->poke(dst, src, size);
    }

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override{
        return this->internal->aggregate(dst, agg, size);
    }

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override{
        return this->internal->enqueue_host_func(callback, payload);
    }

    /// LLVM: reduce a variable that was previously expanded due to
    /// dr.ReduceOp.Expand
    void reduce_expanded(VarType vt, ReduceOp op, void *data, uint32_t exp, uint32_t size) override {
        return this->internal->reduce_expanded(vt, op, data, exp, size);
    }


    ~RecordThreadState(){}

    void replay(std::vector<void *> &input){
        
    }

private:

    ThreadState *internal;
    
    uint32_t variable_count;

    std::vector<uint32_t> input;

    std::vector<Operation> operations;
    std::vector<uint32_t> dependencies;
    // Mapping from variable index in State to variable in this struct.
    VariableCache variable_cache;

    // Insert the variable index, deduplicating it and returning a index into the
    // `variables` field.
    uint32_t get_or_insert_variable(void *ptr) {
        
        auto it = this->variable_cache.find(ptr);

        if (it == this->variable_cache.end()) {
            uint32_t id = this->variable_count++;
            this->variable_cache.insert({ptr, id});
            return id;
        } else {
            return it.value();
        }
    }

    void set_input(std::vector<void *> &inputs){
        for (void * input : inputs){
            this->input.push_back(this->get_or_insert_variable(input));
        }
    }

};
