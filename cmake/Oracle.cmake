# Benchmark oracle + corpus wiring -- Build Map ticket #2, gate M0.
#
# This module adds convenience targets and registers the ticket #2 test suites
# with CTest. It does NOT link anything -- the oracle is a subprocess tool
# (Oracle rule; DEPENDENCY_LEDGER.md S5.5). The HiGHS CLI it drives is built
# out-of-tree by tools/oracle/install_highs.sh into build-oracle/, a path the
# sovereignty check and this project both ignore.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_ORACLE_HIGHS "${CMAKE_SOURCE_DIR}/build-oracle/bin/highs")

# `make oracle` -- build the reference solver if it is not already present.
add_custom_target(oracle
  COMMAND "${CMAKE_COMMAND}" -E echo "Building the benchmark oracle (out-of-tree)..."
  COMMAND sh "${CMAKE_SOURCE_DIR}/tools/oracle/install_highs.sh"
  USES_TERMINAL
  COMMENT "Building HiGHS CLI as a black-box oracle into build-oracle/")

# `make corpus` -- pull the default corpora (Netlib LP).
add_custom_target(corpus
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/benchmarks/fetch_corpus.py"
  USES_TERMINAL
  COMMENT "Fetching default benchmark corpora into benchmarks/data/")

# `make corpus-verify` -- re-hash whatever is downloaded against the lock files.
add_custom_target(corpus-verify
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/benchmarks/fetch_corpus.py" --verify
  COMMENT "Verifying downloaded corpora against their lock files")

if(BUILD_TESTING)
  add_test(NAME oracle_wrapper
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_oracle.py")
  add_test(NAME corpus_manifest
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_corpus.py")

  # A labelled probe of the actual pass condition for ticket #2: solve one real
  # Netlib instance through the CLI oracle and check it against the parsed
  # reference objective. Skips cleanly (exit 0) when the oracle or corpus is
  # absent, so a fresh checkout's `ctest` stays green.
  add_test(NAME oracle_end_to_end
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/oracle/check_reference.py")
  set_tests_properties(oracle_end_to_end PROPERTIES LABELS "m0;oracle")

  # Ticket #5 pass condition: the MPS/LP reader reproduces every downloaded
  # instance's rows/columns/nonzeros identically to the oracle's own parse.
  add_test(NAME parser_vs_oracle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_parser_vs_oracle.py")
  set_tests_properties(parser_vs_oracle PROPERTIES LABELS "m0;io")

  # Ticket #4 pass condition: the from-scratch simplex agrees with the oracle
  # on Netlib and with the hand-verified smoke answers. Row-capped so ctest
  # stays quick; the dense basis makes larger instances a performance study,
  # not a correctness one. Both scripts skip cleanly (exit 0) when the oracle
  # binary or the corpus is absent.
  add_test(NAME simplex_vs_oracle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_simplex_vs_oracle.py"
            --max-rows 250 --timeout 60)
  set_tests_properties(simplex_vs_oracle PROPERTIES LABELS "m0;simplex" TIMEOUT 900)

  if(EXISTS "${_ORACLE_HIGHS}")
    message(STATUS "Benchmark oracle: ${_ORACLE_HIGHS}")
  else()
    message(STATUS "Benchmark oracle: not built yet (run `cmake --build . --target oracle`)")
  endif()
endif()
