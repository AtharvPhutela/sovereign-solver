# GPU toolkit detection -- Build Map ticket #3, gate M0.
#
# Sets SOVEREIGN_WITH_CUDA / SOVEREIGN_WITH_HIP, which src/CMakeLists.txt uses
# to decide which backends get compiled.
#
# The rule that matters: absence is not an error, but it is never silent. A
# build that quietly produced a CPU-only library while a benchmark table said
# "GPU" would invalidate every number above it, so the configure summary always
# states what was really compiled and why.

include(CheckLanguage)

set(SOVEREIGN_WITH_CUDA OFF)
set(SOVEREIGN_WITH_HIP OFF)
set(_sov_cuda_why "")
set(_sov_hip_why "")

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
if(SOVEREIGN_CUDA STREQUAL "OFF")
  set(_sov_cuda_why "disabled by -DSOVEREIGN_CUDA=OFF")
else()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(SOVEREIGN_WITH_CUDA ON)
      set(_sov_cuda_why "CUDA ${CUDAToolkit_VERSION}, nvcc at ${CMAKE_CUDA_COMPILER}")
    else()
      set(_sov_cuda_why "nvcc found but the CUDA toolkit libraries were not")
    endif()
  else()
    set(_sov_cuda_why "no CUDA compiler found")
  endif()

  if(SOVEREIGN_CUDA STREQUAL "ON" AND NOT SOVEREIGN_WITH_CUDA)
    message(FATAL_ERROR
      "-DSOVEREIGN_CUDA=ON was requested but CUDA is unavailable: ${_sov_cuda_why}")
  endif()
endif()

# ---------------------------------------------------------------------------
# ROCm / HIP
# ---------------------------------------------------------------------------
if(SOVEREIGN_HIP STREQUAL "OFF")
  set(_sov_hip_why "disabled by -DSOVEREIGN_HIP=OFF")
else()
  find_package(hip QUIET)
  find_package(rocblas QUIET)
  if(hip_FOUND AND rocblas_FOUND)
    set(SOVEREIGN_WITH_HIP ON)
    set(_sov_hip_why "ROCm hip ${hip_VERSION}, rocBLAS ${rocblas_VERSION}")
  elseif(hip_FOUND)
    set(_sov_hip_why "hip found but rocBLAS was not")
  else()
    set(_sov_hip_why "no ROCm installation found")
  endif()

  if(SOVEREIGN_HIP STREQUAL "ON" AND NOT SOVEREIGN_WITH_HIP)
    message(FATAL_ERROR
      "-DSOVEREIGN_HIP=ON was requested but ROCm is unavailable: ${_sov_hip_why}")
  endif()
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "L0 backends:")
message(STATUS "  host : always compiled (reference implementation and oracle)")
if(SOVEREIGN_WITH_CUDA)
  message(STATUS "  cuda : ENABLED  -- ${_sov_cuda_why}")
else()
  message(STATUS "  cuda : disabled -- ${_sov_cuda_why}")
endif()
if(SOVEREIGN_WITH_HIP)
  message(STATUS "  hip  : ENABLED  -- ${_sov_hip_why}")
else()
  message(STATUS "  hip  : disabled -- ${_sov_hip_why}")
endif()

if(NOT SOVEREIGN_WITH_CUDA AND NOT SOVEREIGN_WITH_HIP)
  message(STATUS
    "  NOTE: no GPU backend compiled. The L0 abstraction and its tests still "
    "run in full on the host backend, but no GPU code path in this build has "
    "been compiled or executed -- treat backend_cuda.cu and backend_hip.cpp as "
    "unverified until a build with a toolkit says otherwise.")
endif()

if(SOVEREIGN_INDEX64)
  message(STATUS "L0 index width: int64 (SOVEREIGN_INDEX64=ON)")
else()
  message(STATUS "L0 index width: int32 (default; -DSOVEREIGN_INDEX64=ON for huge instances)")
endif()
