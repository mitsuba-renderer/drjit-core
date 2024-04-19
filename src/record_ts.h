
#include "drjit-core/hash.h"
#include "drjit-core/jit.h"
#include "internal.h"
#include "log.h"
#include "var.h"
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
    // Kernel hash if a kernel was launched
    Kernel kernel;
    size_t size;
};

struct RecordVariable{
    VarType type;
};

struct RecordThreadState: ThreadState{

    RecordThreadState(ThreadState *internal){
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
            /*dependency_range=*/std::pair(start, end),
            /*kernel=*/kernel,
            /*size=*/size,
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

    std::vector<uint32_t> replay(uint32_t *replay_input, uint32_t n_inputs){

        // This struct holds the data and tracks the size of varaibles, 
        // used during replay.
        struct ReplayVariable{
            void *data;
            uint32_t size;
        };
        
        std::vector<ReplayVariable> variables(this->variables.size());
        std::vector<void *> kernel_params;
        std::vector<Task*> scheduled_tasks;

        // Populate with input variables
        for (uint32_t i = 0; i < n_inputs; ++i){
            Variable* input_variable = jitc_var(replay_input[i]);
            ReplayVariable &replay_variable = variables[this->inputs[i]];
            replay_variable.size = input_variable->size;
            replay_variable.data = input_variable->data;
        }

        // Execute kernels and allocate missing output variables

        for (uint32_t i = 0; i < this->operations.size(); ++i){
            Operation& op = this->operations[i];

            switch (op.type) {
                case OpType::KernelLaunch:
                    kernel_params.clear();
                    
                    if (backend == JitBackend::CUDA) {
                        uintptr_t size = 0;
                        memcpy(&size, &op.size, sizeof(uint32_t));
                        kernel_params.push_back((void *) size);
                    } else {
                        // First 3 parameters reserved for: kernel ptr, size, ITT identifier
                        for (int i = 0; i < 3; ++i)
                            kernel_params.push_back(nullptr);
                    }
                    
                    // Allocate Missing variables for kernel launch.
                    // The assumption here is that for every kernel launch, the inputs are already allocated.
                    // Therefore we only allocate output variables, which have the same size as the kernel.
                    // TODO: deallocate unused memory.
                    for (uint32_t j = op.dependency_range.first; j < op.dependency_range.second; ++j){
                        ReplayVariable &replay_variable = variables[this->dependencies[j]];
                        if (replay_variable.data == nullptr){
                            jitc_log(LogLevel::Info, "Allocating output variable of size %zu.", op.size);
                            
                            RecordVariable &rv = this->variables[this->dependencies[j]];
                            uint32_t dsize = op.size * type_size[(int) rv.type];
                            
                            replay_variable.data = jitc_malloc(AllocType::Device, dsize);
                            replay_variable.size = op.size;
                        }
                        kernel_params.push_back(replay_variable.data);
                    }

                    scheduled_tasks.push_back(this->internal->launch(op.kernel, op.size, &kernel_params));
                    
                    break;
                default:
                    jitc_fail("An operation has been recorded, that is not known to the replay functionality!");
                    break;
            }
        }
        
        if (this->backend == JitBackend::LLVM) {
            if (scheduled_tasks.size() == 1) {
                task_release(jitc_task);
                jitc_task = scheduled_tasks[0];
            } else {
                jitc_assert(!scheduled_tasks.empty(),
                            "jit_eval(): no tasks generated!");

                // Insert a barrier task
                Task *new_task = task_submit_dep(nullptr, scheduled_tasks.data(),
                                                 (uint32_t) scheduled_tasks.size());
                task_release(jitc_task);
                for (Task *t : scheduled_tasks)
                task_release(t);
                jitc_task = new_task;
            }
        }

        std::vector<uint32_t> output_variables;

        for(uint32_t i = 0; i < this->outputs.size(); ++i){
            uint32_t index = this->outputs[i];
            ReplayVariable &rv = variables[index];
            Variable v;
            v.kind = VarKind::Evaluated;
            v.type = (uint32_t) VarType::UInt32;
            v.size = rv.size;
            v.data = rv.data;
            v.backend = (uint32_t) this->backend;
            output_variables.push_back(jitc_var_new(v));
        }
        return output_variables;
    }

    void set_input(void *input){
        this->inputs.push_back(this->get_or_insert_variable(input));
    }
    void set_output(void *output){
        this->outputs.push_back(this->get_or_insert_variable(output));
    }


    ThreadState *internal;
    
    std::vector<RecordVariable> variables;

    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;

    std::vector<Operation> operations;
    std::vector<uint32_t> dependencies;
    
private:
    
    // Mapping from variable index in State to variable in this struct.
    VariableCache variable_cache;

    // Insert the variable index, deduplicating it and returning a index into the
    // `variables` field.
    uint32_t get_or_insert_variable(void *ptr) {
        
        auto it = this->variable_cache.find(ptr);

        if (it == this->variable_cache.end()) {
            uint32_t id = this->variables.size();
            this->variables.push_back(RecordVariable{
                /*type=*/VarType::UInt32,
            });
            this->variable_cache.insert({ptr, id});
            return id;
        } else {
            return it.value();
        }
    }

};


void jitc_record_start(JitBackend backend, uint32_t *inputs, size_t n_inputs);
RecordThreadState *jitc_record_stop(JitBackend backend, uint32_t *outputs, size_t n_outputs);

void jitc_test_record(JitBackend backend);
