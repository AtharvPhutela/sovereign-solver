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

# Ticket #1. Included early and unconditionally: the sovereignty constraint is
# the project's disqualifying condition, so it gates everything below it.
include(Sovereignty)

# Ticket #2 -- benchmark corpus + black-box oracle. Adds `oracle`, `corpus`,
# `corpus-verify` targets and registers the ticket #2 test suites with CTest.
# Links nothing.
include(Oracle)

# --- placeholders for the phases that follow --------------------------------
# add_subdirectory(src)      # ticket #3 onward -- L0 numerical substrate
# add_subdirectory(tests)    # ticket #4 onward -- against the oracle

message(STATUS "sovereign_solver ${PROJECT_VERSION} configured "
               "(Phase 0: scaffolding only, no solver sources yet)")
