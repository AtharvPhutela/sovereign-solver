# L0 numerical substrate -- Build Map ticket #3, gate M0.
#
# One library, `sovereign_l0`, with the GPU backends compiled in only when
# their toolkit is actually present. The detection is deliberately quiet about
# absence and loud about presence: a build that silently produced a CPU-only
# library while a benchmark claimed "GPU" would invalidate every number above
# it, so `backend_report()` and the configure summary always state what was
# really compiled.

add_library(sovereign_l0
  l0/sparse.cpp
  l0/backend.cpp
  l0/backend_host.cpp
  # Ticket #5 -- model readers (L6 interface layer, Bible S4.7). Kept in the
  # same library for now; they split out when the layering earns its own
  # targets.
  io/problem.cpp
  io/mps_reader.cpp
  io/lp_reader.cpp
  # Ticket #4 -- the from-scratch simplex, our own correctness oracle.
  l1/simplex.cpp
  # Ticket #6 -- Ruiz + Pock-Chambolle preconditioning.
  l1/scaling.cpp
  # Ticket #7 -- Farkas/dual-ray & duals plumbing.
  l1/duals.cpp
  # Ticket #8 -- PDHG/PDLP engine.
  l1/pdhg.cpp
  # Ticket #9 -- regularized pivoting-free IPM.
  l1/ipm.cpp
  # Ticket #10 -- the concurrent engine race.
  l1/race.cpp
  # Ticket #11 -- concurrent checkpoint crossover.
  l1/crossover.cpp
  # Ticket #12 -- spiral-axis vertex jump.
  l1/spiral_jump.cpp
)

# Ticket #10 needs real threads (std::thread, not just <thread>'s declarations).
find_package(Threads REQUIRED)
target_link_libraries(sovereign_l0 PUBLIC Threads::Threads)

target_include_directories(sovereign_l0
  PUBLIC "${CMAKE_SOURCE_DIR}/include")

target_compile_features(sovereign_l0 PUBLIC cxx_std_20)

if(SOVEREIGN_INDEX64)
  target_compile_definitions(sovereign_l0 PUBLIC SOVEREIGN_INDEX64)
endif()

if(MSVC)
  target_compile_options(sovereign_l0 PRIVATE /W4)
else()
  target_compile_options(sovereign_l0 PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------
if(SOVEREIGN_WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  target_sources(sovereign_l0 PRIVATE l0/backend_cuda.cu)
  target_compile_definitions(sovereign_l0 PUBLIC SOVEREIGN_WITH_CUDA)
  # cuSPARSE / cuBLAS are numerical primitives, PERMITTED in sovereignty.toml.
  target_link_libraries(sovereign_l0 PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)

  set_target_properties(sovereign_l0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    # sovereign_l0 is STATIC, and CMake does not device-link a static library's
    # separable-compilation objects by default -- it expects whatever consumes
    # the library to do that. But most consumers here (sovereign-cli, the test
    # binaries) are plain C++ targets with no .cu sources, so CMake never
    # inserts that step for them and the final link fails on undefined
    # __cudaRegisterLinkedBinary_*/cudaGetDeviceProperties_v2 symbols. Making
    # the library resolve its own device symbols keeps every CUDA-aware
    # decision inside this one CUDA-gated block, instead of pushing
    # CUDA_RESOLVE_DEVICE_SYMBOLS onto every current and future consumer.
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # `native` targets the card in the build machine; override with
    # -DCMAKE_CUDA_ARCHITECTURES=... when cross-building for a cluster.
    set_target_properties(sovereign_l0 PROPERTIES CUDA_ARCHITECTURES native)
  endif()
endif()

# ---------------------------------------------------------------------------
# ROCm / HIP backend
# ---------------------------------------------------------------------------
if(SOVEREIGN_WITH_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(rocblas REQUIRED)

  set_source_files_properties(l0/backend_hip.cpp PROPERTIES LANGUAGE HIP)
  target_sources(sovereign_l0 PRIVATE l0/backend_hip.cpp)
  target_compile_definitions(sovereign_l0 PUBLIC SOVEREIGN_WITH_HIP)
  target_link_libraries(sovereign_l0 PRIVATE hip::device roc::rocblas)
endif()

add_library(sovereign::l0 ALIAS sovereign_l0)
