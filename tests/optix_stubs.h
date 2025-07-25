#include <drjit-core/optix.h>
// =====================================================
//       Various opaque handles and enumerations
// =====================================================

using CUdeviceptr            = void*;
using CUstream               = void*;
using OptixPipeline          = void *;
using OptixModule            = void *;
using OptixProgramGroup      = void *;
using OptixResult            = int;
using OptixTraversableHandle = unsigned long long;
using OptixBuildOperation    = int;
using OptixBuildInputType    = int;
using OptixVertexFormat      = int;
using OptixIndicesFormat     = int;
using OptixTransformFormat   = int;
using OptixAccelPropertyType = int;
using OptixProgramGroupKind  = int;

using OptixDisplacementMicromapArrayIndexingMode  = int;
using OptixDisplacementMicromapDirectionFormat    = int;
using OptixDisplacementMicromapBiasAndScaleFormat = int;
using OptixDisplacementMicromapFormat             = int;
using OptixOpacityMicromapFormat                  = int;
using OptixOpacityMicromapArrayIndexingMode       = int;

// =====================================================
//            Commonly used OptiX constants
// =====================================================

#define OPTIX_BUILD_INPUT_TYPE_TRIANGLES    0x2141
#define OPTIX_BUILD_OPERATION_BUILD         0x2161
#define OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT  1
#define OPTIX_VERTEX_FORMAT_FLOAT3          0x2121
#define OPTIX_SBT_RECORD_HEADER_SIZE        32

#define OPTIX_COMPILE_DEBUG_LEVEL_NONE      0x2350
#define OPTIX_COMPILE_DEBUG_LEVEL_FULL      0x2352
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_0  0x2340
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_3  0x2343

#define OPTIX_BUILD_FLAG_ALLOW_COMPACTION   2
#define OPTIX_BUILD_FLAG_PREFER_FAST_TRACE  4
#define OPTIX_PROPERTY_TYPE_COMPACTED_SIZE  0x2181

#define OPTIX_EXCEPTION_FLAG_NONE           0
#define OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW 1
#define OPTIX_EXCEPTION_FLAG_TRACE_DEPTH    2

#define OPTIX_PROGRAM_GROUP_KIND_MISS      0x2422
#define OPTIX_PROGRAM_GROUP_KIND_EXCEPTION 0x2423
#define OPTIX_PROGRAM_GROUP_KIND_HITGROUP  0x2424

#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS 1
#define OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE           (1 << 31)

// =====================================================
//          Commonly used OptiX data structures
// =====================================================

struct OptixMotionOptions {
    unsigned short numKeys;
    unsigned short flags;
    float timeBegin;
    float timeEnd;
};

struct OptixAccelBuildOptions {
    unsigned int buildFlags;
    OptixBuildOperation operation;
    OptixMotionOptions motionOptions;
};

struct OptixAccelBufferSizes {
    size_t outputSizeInBytes;
    size_t tempSizeInBytes;
    size_t tempUpdateSizeInBytes;
};

struct OptixOpacityMicromapUsageCount {
    unsigned int count;
    unsigned int subdivisionLevel;
    OptixOpacityMicromapFormat format;
};

struct OptixBuildInputOpacityMicromap {
    OptixOpacityMicromapArrayIndexingMode indexingMode;
    CUdeviceptr opacityMicromapArray;
    CUdeviceptr indexBuffer;
    unsigned int indexSizeInBytes;
    unsigned int indexStrideInBytes;
    unsigned int indexOffset;
    unsigned int numMicromapUsageCounts;
    const OptixOpacityMicromapUsageCount* micromapUsageCounts;
};

struct OptixDisplacementMicromapUsageCount {
    unsigned int count;
    unsigned int subdivisionLevel;
    OptixDisplacementMicromapFormat format;
};

struct OptixBuildInputDisplacementMicromap {
    OptixDisplacementMicromapArrayIndexingMode indexingMode;
    CUdeviceptr displacementMicromapArray;
    CUdeviceptr displacementMicromapIndexBuffer;
    CUdeviceptr vertexDirectionsBuffer;
    CUdeviceptr vertexBiasAndScaleBuffer;
    CUdeviceptr triangleFlagsBuffer;
    unsigned int displacementMicromapIndexOffset;
    unsigned int displacementMicromapIndexStrideInBytes;
    unsigned int displacementMicromapIndexSizeInBytes;
    OptixDisplacementMicromapDirectionFormat vertexDirectionFormat;
    unsigned int vertexDirectionStrideInBytes;
    OptixDisplacementMicromapBiasAndScaleFormat vertexBiasAndScaleFormat;
    unsigned int vertexBiasAndScaleStrideInBytes;
    unsigned int triangleFlagsStrideInBytes;
    unsigned int numDisplacementMicromapUsageCounts;
    const OptixDisplacementMicromapUsageCount* displacementMicromapUsageCounts;
};

struct OptixBuildInputTriangleArray {
    const CUdeviceptr* vertexBuffers;
    unsigned int numVertices;
    OptixVertexFormat vertexFormat;
    unsigned int vertexStrideInBytes;
    CUdeviceptr indexBuffer;
    unsigned int numIndexTriplets;
    OptixIndicesFormat indexFormat;
    unsigned int indexStrideInBytes;
    CUdeviceptr preTransform;
    const unsigned int* flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
    OptixTransformFormat transformFormat;
    OptixBuildInputOpacityMicromap opacityMicromap;
    OptixBuildInputDisplacementMicromap displacementMicromap;
};

struct OptixBuildInput {
    OptixBuildInputType type;
    union {
        OptixBuildInputTriangleArray triangleArray;
        char pad[1024];
    };
};

struct OptixPayloadType {
    unsigned int numPayloadValues;
    const unsigned int *payloadSemantics;
};

struct OptixModuleCompileOptions {
    int maxRegisterCount;
    int optLevel;
    int debugLevel;
    const void *boundValues;
    unsigned int numBoundValues;
    unsigned int numPayloadTypes;
    OptixPayloadType *payloadTypes;
};

struct OptixPipelineCompileOptions {
    int usesMotionBlur;
    unsigned int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    unsigned int exceptionFlags;
    const char* pipelineLaunchParamsVariableName;
    unsigned int usesPrimitiveTypeFlags;
    int allowOpacityMicromaps;
    int allowClusteredGeometry; // OptiX 9.0 ABI
};

struct OptixAccelEmitDesc {
    CUdeviceptr result;
    OptixAccelPropertyType type;
};

struct OptixProgramGroupSingleModule {
    OptixModule module;
    const char* entryFunctionName;
};

struct OptixProgramGroupHitgroup {
    OptixModule moduleCH;
    const char* entryFunctionNameCH;
    OptixModule moduleAH;
    const char* entryFunctionNameAH;
    OptixModule moduleIS;
    const char* entryFunctionNameIS;
};

struct OptixProgramGroupDesc {
    OptixProgramGroupKind kind;
    unsigned int flags;

    union {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupHitgroup hitgroup;
    };
};

struct OptixProgramGroupOptions {
    OptixPayloadType *payloadType;
};

struct OptixShaderBindingTable {
    CUdeviceptr raygenRecord;
    CUdeviceptr exceptionRecord;
    CUdeviceptr  missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    CUdeviceptr  hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    CUdeviceptr  callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
};

// =====================================================
//             Commonly used OptiX functions
// =====================================================

#if defined(OPTIX_STUBS_IMPL)
#  define D(name, ...) OptixResult (*name)(__VA_ARGS__) = nullptr;
#else
#  define D(name, ...) extern OptixResult (*name)(__VA_ARGS__)
#endif

D(optixAccelComputeMemoryUsage, OptixDeviceContext,
  const OptixAccelBuildOptions *, const OptixBuildInput *, unsigned int,
  OptixAccelBufferSizes *);
D(optixAccelBuild, OptixDeviceContext, CUstream, const OptixAccelBuildOptions *,
  const OptixBuildInput *, unsigned int, CUdeviceptr, size_t, CUdeviceptr,
  size_t, OptixTraversableHandle *, const OptixAccelEmitDesc *, unsigned int);
D(optixModuleCreate, OptixDeviceContext,
  const OptixModuleCompileOptions *, const OptixPipelineCompileOptions *,
  const char *, size_t, char *, size_t *, OptixModule *);
D(optixModuleDestroy, OptixModule);
D(optixProgramGroupCreate, OptixDeviceContext, const OptixProgramGroupDesc *,
  unsigned int, const OptixProgramGroupOptions *, char *, size_t *,
  OptixProgramGroup *);
D(optixProgramGroupDestroy, OptixProgramGroup);
D(optixSbtRecordPackHeader, OptixProgramGroup, void*);
D(optixAccelCompact, OptixDeviceContext, CUstream, OptixTraversableHandle,
  CUdeviceptr, size_t, OptixTraversableHandle *);

#undef D

extern void init_optix_api();

