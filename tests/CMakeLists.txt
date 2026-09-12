# L0 tests -- Build Map ticket #3.
#
# Each executable is self-contained (test_support.hpp provides the harness, no
# third-party test framework) and registered with CTest, so `ctest` runs the
# C++ suites alongside the ticket #1/#2 Python ones.

function(sovereign_add_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE sovereign::l0)
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_features(${name} PRIVATE cxx_std_20)
  if(NOT MSVC)
    target_compile_options(${name} PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "m0;l0")
endfunction()

sovereign_add_test(test_l0_sparse  test_l0_sparse.cpp)
sovereign_add_test(test_l0_backend test_l0_backend.cpp)

# Ticket #5 -- model readers.
sovereign_add_test(test_mps_reader test_mps_reader.cpp)
sovereign_add_test(test_lp_reader  test_lp_reader.cpp)

# Ticket #4 -- the from-scratch simplex.
sovereign_add_test(test_simplex    test_simplex.cpp)

# Ticket #6 -- preconditioning.
sovereign_add_test(test_scaling    test_scaling.cpp)

# Ticket #7 -- Farkas/dual-ray & duals plumbing.
sovereign_add_test(test_duals      test_duals.cpp)

# Ticket #8 -- PDHG/PDLP engine.
sovereign_add_test(test_pdhg       test_pdhg.cpp)

# Ticket #9 -- regularized pivoting-free IPM.
sovereign_add_test(test_ipm        test_ipm.cpp)

# Interface parity. On a machine with no CUDA/ROCm toolkit the GPU backends are
# never compiled, so a change to the Backend interface would leave them behind
# silently. This is the part of their correctness checkable without a GPU.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME l0_backend_parity
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_backend_parity.py")
set_tests_properties(l0_backend_parity PROPERTIES LABELS "m0;l0")
