#pragma once

#if defined(ENOKI_CUDA)
#  define CUDA_API_PER_THREAD_DEFAULT_STREAM
#  include <cuda.h>
#  include <cuda_runtime_api.h>
#endif
