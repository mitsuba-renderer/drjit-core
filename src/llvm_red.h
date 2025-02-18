/*
    src/llvm_red.h -- Helper functions for various reduction operations

    Copyright (c) 2024 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

template <typename T> struct RedAdd {
    using Value = std::conditional_t<std::is_same_v<drjit::half, T>, float, T>;

    static Value init() { return Value(0); }

    static Value reduce(Value a, Value b) { return a + b; }
};

template <typename T> struct RedMul {
    using Value = std::conditional_t<std::is_same_v<drjit::half, T>, float, T>;

    static Value init() { return Value(1); }

    static Value reduce(Value a, Value b) { return a * b; }
};

template <typename T> struct RedMin {
    using Value = T;

    static Value init() {
        if constexpr (std::is_integral_v<Value>)
            return std::numeric_limits<Value>::max();
        else
            return std::numeric_limits<Value>::infinity();
    }
    static Value reduce(Value a, Value b) { return std::min(a, b); }
};

template <typename T> struct RedMax {
    using Value = T;

    static Value init() {
        if constexpr (std::is_integral_v<Value>)
            return std::numeric_limits<Value>::min();
        else // the next line generates a warning if not performed in an 'if constexpr' block
            return -std::numeric_limits<Value>::infinity();
    }

    static Value reduce(Value a, Value b) { return std::max(a, b); }
};

template <typename T> struct RedOr {
    using Value = T;

    static Value init() { return Value(0); }

    static Value reduce(Value a, Value b) {
        if constexpr (std::is_integral_v<Value>) {
            return a | b;
        } else {
            (void) a; (void) b;
            return Value(0);
        }
    }
};

template <typename T> struct RedAnd {

    using Value = T;

    static Value init() { return Value(-1); }

    static Value reduce(Value a, Value b) {
        if constexpr (std::is_integral_v<Value>) {
            return a & b;
        } else {
            (void) a; (void) b;
            return Value(0);
        }
    }
};

using BlockReduction = void (*)(uint32_t index, uint32_t work_unit_size,
                                uint32_t size, uint32_t block_size, uint32_t chunk_size,
                                uint32_t chunk_count, uint32_t chunks_per_block,
                                const void *in, void *out);

template <typename T, typename Red>
static BlockReduction create_block_reduction_2() {
    return [](uint32_t index, uint32_t work_unit_size, uint32_t size,
              uint32_t block_size, uint32_t chunk_size, uint32_t chunk_count,
              uint32_t chunks_per_block, const void *in_, void *out_) {
        using Value = typename Red::Value;

        const T *in = (const T *) in_;
        T *out = (T *) out_;

        uint32_t chunk_start = index * work_unit_size,
                 chunk_end   = std::min(chunk_start + work_unit_size, chunk_count);

        for (uint32_t i = chunk_start; i < chunk_end; ++i) {
            uint32_t block_id = i / chunks_per_block,
                     chunk_within_block = i % chunks_per_block;

            uint32_t block_rel_start = chunk_within_block * chunk_size,
                     block_rel_end = std::min(block_rel_start + chunk_size, block_size);

            uint32_t block_offset = block_id * block_size,
                     start = block_rel_start + block_offset,
                     end = std::min(block_rel_end + block_offset, size);

            Value accum = Red::init();
            for (uint32_t j = start; j < end; ++j)
                accum = Red::reduce(accum, (Value) in[j]);

            out[i] = (T) accum;
        }
    };
}

using BlockPrefixReduction = void (*)(uint32_t index, uint32_t work_unit_size,
                                      uint32_t size, uint32_t block_size,
                                      uint32_t chunk_size, uint32_t chunk_count,
                                      uint32_t chunks_per_block, bool exclusive,
                                      bool reverse, const void *in,
                                      const void *scratch, void *out);

template <typename T, typename Red>
static BlockPrefixReduction create_block_prefix_reduction_2() {
    return [](uint32_t index, uint32_t work_unit_size, uint32_t size,
              uint32_t block_size, uint32_t chunk_size, uint32_t chunk_count,
              uint32_t chunks_per_block, bool exclusive, bool reverse,
              const void *in_, const void *scratch_, void *out_) {
        using Value = typename Red::Value;

        const T *in = (const T *) in_;
        const T *scratch = (const T *) scratch_;
        T *out = (T *) out_;

        uint32_t chunk_start = index * work_unit_size,
                 chunk_end   = std::min(chunk_start + work_unit_size, chunk_count);

        for (uint32_t i = chunk_start; i < chunk_end; ++i) {
            uint32_t block_id = i / chunks_per_block,
                     chunk_within_block = i % chunks_per_block;

            uint32_t block_rel_start = chunk_within_block * chunk_size,
                     block_rel_end = std::min(block_rel_start + chunk_size, block_size);

            uint32_t block_offset = block_id * block_size,
                     start = block_rel_start + block_offset,
                     end = std::min(block_rel_end + block_offset, size);

            Value accum;
            if (scratch)
                accum = scratch[i];
            else
                accum = Red::init();

            if (!reverse) {
                for (uint32_t j = start; j < end; ++j) {
                    Value value = (Value) in[j],
                          prev  = accum;

                    accum = Red::reduce(accum, value);
                    out[j] = exclusive ? prev : accum;
                }
            } else {
                for (uint32_t j = end; j > start; --j) {
                    uint32_t k = j - 1;
                    Value value = (Value) in[k],
                          prev  = accum;

                    accum = Red::reduce(accum, value);
                    out[k] = exclusive ? prev : accum;
                }
            }
        }
    };
}

#define DRJIT_ALL_REDUCTIONS(Name)                                             \
    template <typename T> static auto Name##_1(ReduceOp op) {                  \
        switch (op) {                                                          \
            case ReduceOp::Add:                                                \
                return Name##_2<T, RedAdd<T>>();                               \
            case ReduceOp::Mul:                                                \
                return Name##_2<T, RedMul<T>>();                               \
            case ReduceOp::Min:                                                \
                return Name##_2<T, RedMin<T>>();                               \
            case ReduceOp::Max:                                                \
                return Name##_2<T, RedMax<T>>();                               \
            case ReduceOp::Or:                                                 \
                return Name##_2<T, RedOr<T>>();                                \
            case ReduceOp::And:                                                \
                return Name##_2<T, RedAnd<T>>();                               \
            default:                                                           \
                jitc_raise(#Name "(): unsupported reduction type!");           \
        }                                                                      \
    }                                                                          \
                                                                               \
    template <bool... Bs> static auto Name(VarType vt, ReduceOp op) {          \
        switch (vt) {                                                          \
            case VarType::UInt8:                                               \
                return Name##_1<uint8_t>(op);                                  \
            case VarType::Int32:                                               \
                return Name##_1<int32_t>(op);                                  \
            case VarType::UInt32:                                              \
                return Name##_1<uint32_t>(op);                                 \
            case VarType::Int64:                                               \
                return Name##_1<int64_t>(op);                                  \
            case VarType::UInt64:                                              \
                return Name##_1<uint64_t>(op);                                 \
            case VarType::Float16:                                             \
                return Name##_1<drjit::half>(op);                              \
            case VarType::Float32:                                             \
                return Name##_1<float>(op);                                    \
            case VarType::Float64:                                             \
                return Name##_1<double>(op);                                   \
            default:                                                           \
                jitc_raise(#Name "(): unsupported data type!");                \
        }                                                                      \
    }

DRJIT_ALL_REDUCTIONS(create_block_reduction)
DRJIT_ALL_REDUCTIONS(create_block_prefix_reduction)

using Reduction2 = void (*) (const void *ptr_1, const void *ptr_2, uint32_t start, uint32_t end, void *out);
template <typename Value>
static Reduction2 reduce_dot_create() {
    return [](const void *ptr_1_, const void *ptr_2_,
              uint32_t start, uint32_t end, void *out) JIT_NO_UBSAN {
        const Value *ptr_1 = (const Value *) ptr_1_;
        const Value *ptr_2 = (const Value *) ptr_2_;
        Value result = 0;
        for (uint32_t i = start; i != end; ++i) {
            result = std::fma(ptr_1[i], ptr_2[i], result);
        }
        *((Value *) out) = result;
    };
}


static Reduction2 jitc_reduce_dot_create(VarType type) {
    using half = drjit::half;
    switch (type) {
        case VarType::Float16: return reduce_dot_create<half  >();
        case VarType::Float32: return reduce_dot_create<float >();
        case VarType::Float64: return reduce_dot_create<double>();
        default: jitc_raise("jit_reduce_dot_create(): unsupported data type!");
    }
}


