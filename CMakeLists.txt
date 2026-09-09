# Sovereign GPU-Accelerated Optimization Solver
#
# Build Map Phase 0. At this point the project deliberately contains no solver
# code: ticket #1 (sovereignty ledger) is complete, tickets #2 (benchmark corpus
# and oracle harness) and #3 (L0 numerical substrate) are next, and #3 is where
# the sparse containers and the CUDA/ROCm backend abstraction land.
#
# What exists here now is the scaffolding the sovereignty check needs in order
# to have a build manifest and a dependency graph to inspect.

cmake_minimum_required(VERSION 3.24)

project(sovereign_solver
  VERSION 0.0.1
  DESCRIPTION "From-scratch GPU-native LP / MILP / QP solver core"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
endif()

# --- L0 build policy (ticket #3) --------------------------------------------
# int32 indices by default: they halve the bandwidth of the index arrays in a
# memory-bound SpMV and are what cuSPARSE wants natively. Industrial instances
# can exceed 2^31 nonzeros, so the width is a switch rather than an assumption.
option(SOVEREIGN_INDEX64 "Use 64-bit indices (for instances beyond 2^31 nonzeros)" OFF)

# GPU backends are compiled only when their toolkit is actually found. AUTO
# detects; ON demands and fails the configure if absent; OFF forces them out.
set(SOVEREIGN_CUDA "AUTO" CACHE STRING "CUDA backend: AUTO, ON, or OFF")
set(SOVEREIGN_HIP  "AUTO" CACHE STRING "ROCm/HIP backend: AUTO, ON, or OFF")
set_property(CACHE SOVEREIGN_CUDA PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE SOVEREIGN_HIP PROPERTY STRINGS AUTO ON OFF)
include(DetectAccelerators)

# Ticket #1. Included early and unconditionally: the sovereignty constraint is
# the project's disqualifying condition, so it gates everything below it.
include(Sovereignty)

# Ticket #2 -- benchmark corpus + black-box oracle. Adds `oracle`, `corpus`,
# `corpus-verify` targets and registers the ticket #2 test suites with CTest.
# Links nothing.
include(Oracle)

# Ticket #3 -- L0 numerical substrate: sparse containers, numeric policy, and
# the CUDA/ROCm backend abstraction.
add_subdirectory(src)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# --- placeholders for the phases that follow --------------------------------
# ticket #4  -- from-scratch CPU revised simplex (the correctness oracle)
# ticket #5  -- MPS / LP parser
# ticket #6  -- Ruiz + Pock-Chambolle preconditioning

message(STATUS "sovereign_solver ${PROJECT_VERSION} configured "
               "(Phase 0 complete: L0 substrate + corpus + oracle)")
