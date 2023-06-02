#pragma once

#if defined(DRJIT_ENABLE_LLVM)
#  include "llvm.h"
#endif

#if defined(DRJIT_ENABLE_CUDA)
#  include "cuda.h"
#endif

#if defined(DRJIT_ENABLE_OPTIX)
#  include "optix.h"
#endif

#if defined(DRJIT_ENABLE_METAL)
#  include "metal.h"
#endif
