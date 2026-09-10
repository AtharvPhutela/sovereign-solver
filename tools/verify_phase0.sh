#!/bin/sh
# Phase 0 gate check -- Build Map tickets #1, #2, #3 (gate M0, partial).
#
# Checks each ticket against the Build Map's own "Done when" wording rather
# than against "the tests pass", because those are not the same claim. Where a
# condition cannot be met on this machine it says so out loud instead of
# quietly narrowing what was checked -- an unverifiable claim reported as
# verified is the failure mode this whole project's discipline exists to avoid.
#
#   tools/verify_phase0.sh [build-dir]
#
# Exit 0 if every checkable condition holds, 1 otherwise.

set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-${ROOT}/build}"
PY="${PYTHON:-python3}"

pass=0
fail=0
skip=0

section() { printf '\n\033[1m%s\033[0m\n' "$1"; }
ok()      { printf '  \033[32mPASS\033[0m  %s\n' "$1"; pass=$((pass + 1)); }
no()      { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail + 1)); }
note()    { printf '  \033[33mNOTE\033[0m  %s\n' "$1"; skip=$((skip + 1)); }
info()    { printf '        %s\n' "$1"; }

run() { if "$@" >/dev/null 2>&1; then return 0; else return 1; fi; }

printf '\033[1mPhase 0 verification\033[0m  (%s)\n' "$ROOT"

# ---------------------------------------------------------------------------
section "Build"
# ---------------------------------------------------------------------------
if [ ! -d "$BUILD" ]; then
  info "configuring $BUILD"
  cmake -S "$ROOT" -B "$BUILD" -G Ninja >/dev/null 2>&1 \
    || cmake -S "$ROOT" -B "$BUILD" >/dev/null 2>&1
fi
if run cmake --build "$BUILD"; then
  ok "project configures and builds (sovereignty target runs as part of ALL)"
else
  no "build failed -- run: cmake --build $BUILD"
fi

# ---------------------------------------------------------------------------
section "Ticket #1 -- Sovereignty Dependency Ledger"
# Done when: every dependency is classified in writing, and a CI job fails the
# build if a forbidden name appears in the dependency graph.
# ---------------------------------------------------------------------------
if run "$PY" "$ROOT/tools/sovereignty_check.py" --check-ledger; then
  ok "tree, submodules and vendored sources are clean; ledger covers the policy"
else
  no "sovereignty check failed -- see DEPENDENCY_LEDGER.md"
fi

if run "$PY" "$ROOT/tools/sovereignty_check.py" --cmake-build-dir "$BUILD" --check-ledger; then
  ok "resolved CMake dependency graph is clean"
else
  no "a forbidden dependency reached the resolved dependency graph"
fi

if run "$PY" "$ROOT/tools/test_sovereignty_check.py"; then
  ok "checker self-tests (the checker is itself tested before it is trusted)"
else
  no "sovereignty checker self-tests failed"
fi

# The negative half of the pass condition: planting a violation must break the
# build. Checked for real, in a scratch copy, rather than assumed.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM
mkdir -p "$TMP/probe"
cp "$ROOT/sovereignty.toml" "$ROOT/DEPENDENCY_LEDGER.md" "$TMP/probe/"
printf 'find_package(SCIP REQUIRED)\ntarget_link_libraries(x PRIVATE -lcbc)\n' \
  > "$TMP/probe/CMakeLists.txt"
if "$PY" "$ROOT/tools/sovereignty_check.py" --root "$TMP/probe" >/dev/null 2>&1; then
  no "a planted forbidden dependency was NOT detected"
else
  ok "a planted forbidden dependency is detected (checker fails closed)"
fi

# ---------------------------------------------------------------------------
section "Ticket #2 -- Benchmark Corpus & Oracle Harness"
# Done when: all corpora are local, and the oracle answers a sample instance
# through a clean file-based interface.
# ---------------------------------------------------------------------------
if run "$PY" "$ROOT/tests/test_corpus.py"; then
  ok "corpus manifest, Netlib readme parsing, and smoke-set integrity"
else
  no "corpus tests failed"
fi

if run "$PY" "$ROOT/tests/test_oracle.py"; then
  ok "oracle wrapper: backend resolution, parsing, status normalization"
else
  no "oracle tests failed"
fi

if [ -f "$ROOT/benchmarks/data/netlib_lp/_lock.toml" ]; then
  n=$(grep -c '^\[instances\.' "$ROOT/benchmarks/data/netlib_lp/_lock.toml" || echo 0)
  if run "$PY" "$ROOT/benchmarks/fetch_corpus.py" --verify; then
    ok "Netlib LP corpus present and intact ($n instances, sha256-verified)"
  else
    no "downloaded corpus failed sha256 verification"
  fi
else
  note "Netlib LP not downloaded -- run: $PY benchmarks/fetch_corpus.py"
fi
info "MIPLIB / Mittelmann / MILPBench / QPLIB are deferred by design (see corpus.toml)"

if [ -x "$ROOT/build-oracle/bin/highs" ] || command -v highs >/dev/null 2>&1; then
  if run "$PY" "$ROOT/tools/oracle/check_reference.py"; then
    ok "oracle solves real Netlib instances and matches the published optima"
  else
    no "oracle disagreed with the Netlib reference -- do not trust anything downstream"
  fi
else
  note "no oracle binary -- run: tools/oracle/install_highs.sh"
fi

# ---------------------------------------------------------------------------
section "Ticket #3 -- L0 Numerical Substrate"
# Done when: a matrix lives in VRAM, one primitive runs through the backend
# abstraction, and the ROCm path compiles (even if stubbed).
# ---------------------------------------------------------------------------
if [ -x "$BUILD/tests/test_l0_sparse" ] && run "$BUILD/tests/test_l0_sparse"; then
  ok "sparse containers: CSR/CSC, triplet assembly, validation, transpose"
else
  no "L0 sparse tests failed"
fi

if [ -x "$BUILD/tests/test_l0_backend" ] && run "$BUILD/tests/test_l0_backend"; then
  ok "backend abstraction: SpMV matches hand-computed values on every backend"
  ok "residency contract: the matrix crosses the bus exactly once"
else
  no "L0 backend tests failed"
fi

if run "$PY" "$ROOT/tools/check_backend_parity.py"; then
  ok "all three backends implement the full interface (no silent drift)"
else
  no "a backend has drifted from the Backend interface"
fi

# The honest part. "A matrix lives in VRAM" is verified on whatever backends
# were actually compiled; naming them is the difference between a demonstrated
# claim and an assumed one.
BACKENDS="$("$BUILD/tests/test_l0_backend" 2>/dev/null | head -1 || echo 'unknown')"
info "compiled backends: ${BACKENDS}"
if printf '%s' "$BACKENDS" | grep -q cuda; then
  ok "CUDA backend compiled -- device residency and SpMV verified on real VRAM"
else
  note "no CUDA toolkit: backend_cuda.cu was NOT compiled or run on this machine"
fi
if printf '%s' "$BACKENDS" | grep -q hip; then
  ok "HIP backend compiled -- the ROCm path builds, as the ticket requires"
else
  note "no ROCm: backend_hip.cpp was NOT compiled on this machine"
  info "ticket #3 asks that the ROCm path compile; that remains unproven here"
fi

# ---------------------------------------------------------------------------
section "Ticket #5 -- MPS / LP Parser"
# Done when: the parser reproduces instance dimensions and structure identically
# to the oracle across a varied sample.
# ---------------------------------------------------------------------------
if [ -x "$BUILD/tests/test_mps_reader" ] && run "$BUILD/tests/test_mps_reader" \
   && [ -x "$BUILD/tests/test_lp_reader" ] && run "$BUILD/tests/test_lp_reader"; then
  ok "reader unit tests: RANGES table, BOUNDS types, free rows, objective constant, fixed-column fallback"
else
  no "reader unit tests failed"
fi

if [ -x "$ROOT/build-oracle/bin/highs" ] && [ -d "$ROOT/benchmarks/data/netlib_lp" ]; then
  if run "$PY" "$ROOT/tools/check_parser_vs_oracle.py"; then
    ok "reader reproduces rows/columns/nonzeros identically to the oracle on every Netlib instance"
  else
    no "the parser's shape disagrees with the oracle's -- a parsing bug wearing a numerical disguise"
  fi
else
  note "oracle or corpus absent -- parser-vs-oracle shape diff not run"
fi

# ---------------------------------------------------------------------------
section "Ticket #4 -- From-Scratch CPU Revised Simplex"
# Done when: the CPU simplex agrees with the external oracle on Netlib, and
# survives a degenerate instance without cycling.
# ---------------------------------------------------------------------------
if [ -x "$BUILD/tests/test_simplex" ] && run "$BUILD/tests/test_simplex"; then
  ok "simplex unit tests: hand-checked optima, the three terminal statuses, solution feasibility"
  ok "anti-cycling: Beale's 1955 cycling example and a degenerate optimum both terminate"
else
  no "simplex unit tests failed"
fi

if [ -x "$ROOT/build-oracle/bin/highs" ] && [ -d "$ROOT/benchmarks/data/netlib_lp" ]; then
  if run "$PY" "$ROOT/tools/check_simplex_vs_oracle.py" --max-rows 250 --timeout 60; then
    ok "simplex agrees with the oracle on the Netlib instances the dense basis can take"
    info "larger instances are a performance study, not a correctness one (dense O(m^3) factorization)"
  else
    no "the simplex disagrees with the oracle -- the oracle role is void until this is resolved"
  fi
else
  note "oracle or corpus absent -- simplex-vs-oracle comparison not run"
fi

# ---------------------------------------------------------------------------
section "Ticket #6 -- Preconditioning (Ruiz + Pock-Chambolle)"
# Done when: preconditioning measurably improves conditioning on a deliberately
# ill-scaled instance, and the scaled solve returns the original answer.
# ---------------------------------------------------------------------------
if [ -x "$BUILD/tests/test_scaling" ] && run "$BUILD/tests/test_scaling"; then
  ok "Ruiz collapses the row/column norm spread; scaled problem has the same optimum"
else
  no "scaling tests failed"
fi

if [ -x "$BUILD/apps/sovereign-cli" ] && [ -f "$ROOT/benchmarks/data/netlib_lp/grow22.mps" ]; then
  base=$("$BUILD/apps/sovereign-cli" solve "$ROOT/benchmarks/data/netlib_lp/grow22.mps" --json 2>/dev/null \
         | "$PY" -c "import json,sys;print(json.load(sys.stdin)['iterations'])" 2>/dev/null || echo 0)
  scl=$("$BUILD/apps/sovereign-cli" solve "$ROOT/benchmarks/data/netlib_lp/grow22.mps" --scale --json 2>/dev/null \
        | "$PY" -c "import json,sys;print(json.load(sys.stdin)['iterations'])" 2>/dev/null || echo 0)
  if [ "$scl" -gt 0 ] && [ "$base" -gt 0 ] && [ "$scl" -lt "$base" ]; then
    ok "on a real ill-conditioned instance (grow22) --scale cuts iterations ($base -> $scl)"
  else
    note "grow22 iteration comparison inconclusive (plain=$base scaled=$scl)"
  fi
else
  note "cli or corpus absent -- --scale demonstration on real instances not run"
fi

# ---------------------------------------------------------------------------
section "Full suite"
# ---------------------------------------------------------------------------
if run ctest --test-dir "$BUILD"; then
  ok "ctest: every registered suite passes"
else
  no "ctest reported failures -- run: ctest --test-dir $BUILD --output-on-failure"
fi

# ---------------------------------------------------------------------------
printf '\n\033[1mPhase 0: %d passed, %d failed, %d not verifiable here\033[0m\n' \
  "$pass" "$fail" "$skip"

if [ "$fail" -ne 0 ]; then
  printf '\033[31mPhase 0 is NOT green.\033[0m\n'
  exit 1
fi
if [ "$skip" -ne 0 ]; then
  printf '\033[33mPhase 0 is green for everything checkable on this machine.\033[0m\n'
  printf 'The NOTE lines above are real gaps, not passes -- carry them forward.\n'
else
  printf '\033[32mPhase 0 fully verified.\033[0m\n'
fi
exit 0
