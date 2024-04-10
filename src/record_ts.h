
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"
#include <cstdint>

// HashMap used to deduplicate variables
using VariableCache = tsl::robin_map<uintptr_t, uint32_t>;

enum class OpType{
    KernelLaunch,
};

struct Operation{
    OpType type;
    // Indices into the dependencies vector
    std::pair<uint32_t, uint32_t> dependency_range;
};


template<typename TS>
struct RecordThreadState: TS{

    RecordThreadState(){
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
            .type = OpType::KernelLaunch,
            .dependency_range = std::pair(start, end)
        });
        
        return TS::launch(kernel, size, kernel_params);
    }

    /// Fill a device memory region with constants of a given type
    void memset_async(void *ptr, uint32_t size, uint32_t isize,
                      const void *src) override{
        return TS::memset_async(ptr, size, isize, src);
    }

    /// Reduce the given array to a single value
    void reduce(VarType type, ReduceOp rtype, const void *ptr, uint32_t size,
                void *out) override{
        return TS::reduce(type, rtype, ptr, size, out);
    }

    /// 'All' reduction for boolean arrays
    bool all(uint8_t *values, uint32_t size) override{
        return TS::all(values, size);
    }

    /// 'Any' reduction for boolean arrays
    bool any(uint8_t *values, uint32_t size) override{
        return TS::any(values, size);
    }

    /// Exclusive prefix sum
    void prefix_sum(VarType vt, bool exclusive, const void *in, uint32_t size,
                    void *out) override{
        return TS::prefix_sum(vt, exclusive, in ,size, out);
    }

    /// Mask compression
    uint32_t compress(const uint8_t *in, uint32_t size, uint32_t *out) override{
        return TS::compress(in, size, out);
    }

    /// Compute a permutation to reorder an integer array into discrete groups
    uint32_t mkperm(const uint32_t *values, uint32_t size, uint32_t bucket_count,
                    uint32_t *perm, uint32_t *offsets) override{
        return TS::mkperm(values, size, bucket_count, perm, offsets);
    }

    /// Perform a synchronous copy operation
    void memcpy(void *dst, const void *src, size_t size) override{
        return TS::memcpy(dst, src, size);
    }

    /// Perform an assynchronous copy operation
    void memcpy_async(void *dst, const void *src, size_t size) override{
        return TS::memcpy_async(dst, src, size);
    }

    /// Sum over elements within blocks
    void block_sum(enum VarType type, const void *in, void *out, uint32_t size,
                   uint32_t block_size) override{
        return TS::block_sum(type, in, out, size, block_size);
    }

    /// Asynchronously update a single element in memory
    void poke(void *dst, const void *src, uint32_t size) override{
        return TS::poke(dst, src, size);
    }

    void aggregate(void *dst, AggregationEntry *agg, uint32_t size) override{
        return TS::aggregate(dst, agg, size);
    }

    // Enqueue a function to be run on the host once backend computation is done
    void enqueue_host_func(void (*callback)(void *), void *payload) override{
        return TS::enqueue_host_func(callback, payload);
    }

    /// LLVM: reduce a variable that was previously expanded due to
    /// dr.ReduceOp.Expand
    void reduce_expanded(VarType vt, ReduceOp op, void *data, uint32_t exp, uint32_t size) override {
        return TS::reduce_expanded(vt, op, data, exp, size);
    }


    ~RecordThreadState(){}

private:
    uint32_t variable_count;
    std::vector<uint32_t> dependencies;
    // Mapping from variable index in State to variable in this struct.
    VariableCache variable_cache;

    std::vector<Operation> operations;

    // Insert the variable index, deduplicating it and returning a index into the
    // `variables` field.
    uint32_t get_or_insert_variable(void *ptr) {
        uintptr_t hash_index = (uintptr_t)ptr;
        
        auto it = this->variable_cache.find(hash_index);

        if (it == this->variable_cache.end()) {
            uint32_t id = this->variable_count++;
            this->variable_cache.insert({hash_index, id});
            return id;
        } else {
            return it.value();
        }
    }

};
