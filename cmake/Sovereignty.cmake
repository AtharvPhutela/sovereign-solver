# Sovereignty enforcement -- Build Map ticket #1, gate M0.
#
# Wires tools/sovereignty_check.py into the build in two places, because the two
# catch different things:
#
#   configure time -- scans manifests, source, submodules and vendored trees.
#                     Fails fast, before anything is compiled.
#   build time     -- additionally scans the RESOLVED dependency graph
#                     (CMakeCache.txt, build.ninja, link.txt), which only exists
#                     once the project is configured. A manifest can be clean
#                     while a transitively-pulled target puts a forbidden library
#                     on the link line; this is the pass that sees that.
#
# The build-time target is in ALL deliberately: a forbidden dependency must break
# an ordinary local `cmake --build`, not just CI. A check you only meet on a pull
# request is a check you discover too late.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(SOVEREIGNTY_CHECK_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tools/sovereignty_check.py")
set(SOVEREIGNTY_POLICY "${CMAKE_CURRENT_SOURCE_DIR}/sovereignty.toml")

if(NOT EXISTS "${SOVEREIGNTY_CHECK_SCRIPT}")
  message(FATAL_ERROR
    "sovereignty check script is missing: ${SOVEREIGNTY_CHECK_SCRIPT}\n"
    "This is the disqualifying constraint of the project (Bible S1.2); the build "
    "must not proceed without it.")
endif()

option(SOVEREIGNTY_CHECK_AT_CONFIGURE "Run the sovereignty check during CMake configure" ON)
option(SOVEREIGNTY_CHECK_AT_BUILD "Run the sovereignty check as part of the default build target" ON)

# -- configure-time pass ------------------------------------------------------
if(SOVEREIGNTY_CHECK_AT_CONFIGURE)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${SOVEREIGNTY_CHECK_SCRIPT}"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --check-ledger
    RESULT_VARIABLE _sov_result
    OUTPUT_VARIABLE _sov_output
    ERROR_VARIABLE _sov_error)

  if(NOT _sov_result EQUAL 0)
    message(FATAL_ERROR
      "\n${_sov_output}${_sov_error}\n"
      "Sovereignty check failed at configure time (exit ${_sov_result}).\n"
      "See DEPENDENCY_LEDGER.md.")
  endif()
  message(STATUS "Sovereignty check: clean (configure-time pass)")
endif()

# -- build-time pass, including the resolved dependency graph -----------------
if(SOVEREIGNTY_CHECK_AT_BUILD)
  add_custom_target(sovereignty_check ALL
    COMMAND "${Python3_EXECUTABLE}" "${SOVEREIGNTY_CHECK_SCRIPT}"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --cmake-build-dir "${CMAKE_BINARY_DIR}"
            --check-ledger
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Checking dependency sovereignty (ticket #1)"
    VERBATIM)
endif()

# Self-test for the checker. Run with `ctest -R sovereignty`.
if(BUILD_TESTING)
  add_test(NAME sovereignty_check_selftest
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/test_sovereignty_check.py")
endif()
