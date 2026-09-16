# Handoff — Sovereign Solver, Phase 0 complete

**Last commit:** `efd3a44` — *Ticket #3: L0 numerical substrate and the CUDA/ROCm backend abstraction*
**Phase 0 (#1, #2, #3): complete.** Gate M0 is *not* closed — it also needs #4
(CPU simplex oracle) and #5 (MPS/LP parser), both in Phase 1.
**Phase 0 gate:** `tools/verify_phase0.sh` → 14 passed, 0 failed, 2 not
verifiable on this machine (the CUDA and HIP backends — see §11.3).

This document covers **only what was written in this repo**. The Bible and Build
Map explain the *why* and the *what-next* and are uploaded alongside — this fills
the gap between them and the actual code on disk.

---

## 1. Where the project is

Build Map Phase 0 (tickets #1–#3 of 55) is complete. The repo was a bare
directory with three `.md` files; it is now a CMake project with the sovereignty
machinery, a benchmark corpus and oracle, and the L0 numerical substrate.

**There is still no *solver* yet** — no simplex, no parser, no LP is solved by
our own code. L0 gives the algorithms their containers and their device
abstraction; the first thing that actually solves something is ticket #4.

Sections §2–§9 below describe ticket #1, §10 ticket #2, §11 ticket #3, §12 the
Phase 0 gate, §13 what is next.

### Files added in ticket #1

```
CMakeLists.txt                     top-level, ~40 lines, scaffolding only
sovereignty.toml                   machine-readable dependency policy (the source of truth)
DEPENDENCY_LEDGER.md               human-readable ledger (ticket #1's actual deliverable)
tools/sovereignty_check.py         the enforcement scanner (~555 lines, stdlib only)
tools/test_sovereignty_check.py    38 tests for the scanner (~400 lines)
cmake/Sovereignty.cmake            wires the check into configure-time + build
.github/workflows/sovereignty.yml  CI job
.gitignore
```

Directories `src/`, `include/` exist but are empty (git won't track them; they'll
appear when #3 lands).

---

## 2. The sovereignty check — how it actually works

The Build Map says "grep the build manifest against the FORBIDDEN list". That was
taken as the intent, not the implementation. A literal substring grep is wrong in
both directions: it fires on `scipy` (contains `scip`) and it mangles `-lcbc`
into a match on `lcbc`. The first kind of error is the dangerous one — **a check
that cries wolf on a legitimate dependency gets switched off within two weeks**,
and a disabled check protects nothing.

### 2.1 Three tiers, not two

`sovereignty.toml` classifies every dependency as `permitted`, `restricted`, or
`forbidden`.

The **restricted** tier is an addition beyond what the ticket text implies, and
it is the most important design decision in this ticket. Two real cases have no
honest binary answer:

1. **The benchmark oracle (ticket #2).** An established solver is *forbidden as a
   library* and *required as a validation tool*. Resolution: it is reachable only
   as a black-box CLI over files on disk — never linked, never imported.
2. **METIS.** As a nested-dissection fill-reducing ordering inside sparse
   factorization it is pure linear algebra. As block-structure detection it would
   be doing **ticket #14's** job — a named differentiator (Bet 6). Same library,
   opposite verdicts, distinguishable *only by where it is used*.

A `restricted` entry declares `allowed_scopes` (path globs). A reference inside
those paths is fine; a reference outside them **fails the build**. This makes the
ordering-vs-structure-detection boundary mechanical instead of a code-review
argument. METIS is scoped to `src/l0/**` and `cmake/**`.

The checker refuses to load a policy where a `restricted` entry has no
`allowed_scopes` (that's just a `permitted` entry), or where any entry has an
empty `reason` (ticket #1's pass condition is "classified **in writing**", so an
unjustified entry is a hard policy error — exit code 2, not a warning).

### 2.2 Token-aware matching

`name_candidates(token)` turns a raw source token into the set of dependency
names it could plausibly be naming:

- splits on path separators (`/ \ . : ,`) and word separators (`_ -`)
- strips `lib` prefixes → `libscip.so.8` yields `scip`
- resolves linker flags → `-lscip` yields `scip`, `-lcbc` yields `cbc`
- drops file-extension noise (`so`, `a`, `h`, `cpp`, `cmake`, …)

Result: `scipy` never resolves to `scip`, but `libscip.so.8`,
`coinor-libcbc-dev`, `SCIP_DIR`, and `--with-scip` all resolve correctly.

Confirmed behaviour (these are in the test suite):

| Text | Verdict |
|---|---|
| `import scipy` | ignored (correct) |
| `-lscip` / `-lcbc` | flagged |
| `libscip.so.8` | flagged |
| `coinor-libcbc-dev` | flagged |
| `SCIP_DIR`, `CBC_ROOT` | flagged |
| `AES_CBC_MODE` | flagged until a justified `[[exceptions]]` entry is added |

### 2.3 What gets scanned

1. **Build manifests** — `CMakeLists.txt`, `*.cmake`, `pyproject.toml`,
   `requirements*.txt`, `vcpkg.json`, `conanfile.*`, `meson.build`, `Makefile`,
   `.gitmodules`, `*.pc`
2. **Source** — C/C++/CUDA/HIP + Python, catching `#include` and `import`
3. **Shell scripts** (`*.sh`, `*.bash`, added in #3) — where a `git clone` or
   `apt install` of a forbidden library would otherwise enter untouched by any
   manifest
4. **Vendored directories** — any subdir *or file* of `third_party/`, `extern/`,
   `vendor/`, `deps/`, `subprojects/`, … whose *name* matches a forbidden entry.
   Catches a raw source-tree copy that no manifest mentions.
5. **Resolved dependency graph** — with `--cmake-build-dir <dir>`, reads
   `CMakeCache.txt`, `build.ninja`, `rules.ninja`, `link.txt`, `*.cmake`. This is
   the pass that catches a forbidden library arriving **transitively** — a
   manifest can be perfectly clean while a pulled-in target puts something on the
   link line. Declared intent ≠ resolved reality.

### 2.4 Comment handling — subtle, and the source of a bug that was caught

Comments are skipped because this codebase *deliberately* discusses forbidden
libraries in prose ("studied, never linked"). But the comment marker is
language-specific and getting it wrong is dangerous in exactly one direction:

- **C-family files** (`.c/.cc/.cpp/.cu/.h/.hpp/.cuh/.hip`): comment prefixes are
  `//`, `/*`, `*`. **`#` is NOT a comment** — it opens a preprocessor directive.
  An early version treated `#` as a comment everywhere, which silently blinded
  the check to `#include <scip/scip.h>` — the single most likely real leak
  vector in a C++ project. Fixed; regression tests lock it in both directions.
- **`CMakeCache.txt`**: uses `//` for the doc comment above each entry, so `#`
  and `//` are both comment prefixes there.
- **Everything else**: `#`.

### 2.5 CMake private-cache filtering — the second bug that was caught

`find_package(Python3)` writes bookkeeping into `CMakeCache.txt` like
`_Python3_NumPy_REASON_FAILURE:INTERNAL=` (this records that NumPy was *not*
found). The depgraph scanner read that as a NumPy dependency declaration and
failed a build that has **no dependencies at all**.

Fix: `strip_cmake_private_cache()` blanks lines matching
`^_[A-Za-z0-9_]*:(INTERNAL|STATIC)=` before scanning `CMakeCache.txt` (line
numbering preserved). A genuine `SCIP_LIBRARY:FILEPATH=...` entry (no leading
underscore) is still caught — test `test_real_cache_entry_still_caught` guards
that the filter didn't become a blanket amnesty.

### 2.6 Exceptions

`[[exceptions]]` entries in `sovereignty.toml` suppress confirmed false
positives. Each needs a `token`, a mandatory `reason`, and an optional `paths`
scope. **The checker refuses to run on an exception with no reason** — an
undocumented suppression is indistinguishable from a breach.

Ticket #1 shipped with zero exceptions (both bugs above were fixed properly
rather than suppressed). Tickets #2 and #3 added scoped ones, each justified in
`sovereignty.toml`: the oracle wrapper must name the solver executables it runs
(`tools/oracle/**`), `fetch_corpus.py` parses a CPLEX *citation* out of the
Netlib readme, and `verify_phase0.sh` plants violations deliberately to prove
the checker fails closed. An exception matches by resolved policy entry, so one
keyed `glpk` also covers a hit on its alias `glpsol`.

### 2.7 Exit codes

`0` clean · `1` violation · `2` policy/usage error. A malformed `sovereignty.toml`
exits `2`, never `0` — a broken policy must never read as a pass.

### 2.8 CLI surface

```
tools/sovereignty_check.py                          # scan the tree
tools/sovereignty_check.py --check-ledger           # + fail on ledger/policy drift
tools/sovereignty_check.py --cmake-build-dir build  # + scan resolved dep graph
tools/sovereignty_check.py --explain nauty          # print one entry's classification
tools/sovereignty_check.py -v                       # list permitted hits + in-scope restricted
tools/test_sovereignty_check.py                     # test the checker
```

---

## 3. `sovereignty.toml` — contents and the non-obvious classifications

`schema_version = 1`. The checker rejects any other value. If the schema changes,
bump this and update `load_policy()`.

**76 entries: 37 forbidden, 7 restricted, 32 permitted.** The Bible §1.2 list is
the starting point; the following were added by applying the same logic and are
worth knowing about because they are **not** in the Bible or Build Map:

### Forbidden, added beyond the Bible's explicit list

- **Hypergraph partitioners — KaHyPar / Mt-KaHyPar, PaToH, hMETIS.** Same logic
  as the nauty ban: ticket #14 must *own* multi-level hypergraph partitioning
  (Bet 6). hMETIS is called out as distinct from METIS — hypergraph partitioning
  has exactly one use here and it is ours; graph partitioning for fill-reduction
  is linear algebra.
- **Conic / first-order solvers — Clarabel, ECOS, SCS.** SCS especially: it's the
  same algorithmic family (operator splitting) as our PDHG engine.
- **qpOASES, MOSEK, PySP, DSP** — rounded out the solver categories.
- **Traces** — ships inside the nauty distribution, banned with it.
- **PermLib** — Schreier-Sims / BSGS library; ticket #42 builds the stabilizer
  chain itself.
- **GLOP and PDLP are listed as separate entries** from the OR-Tools umbrella,
  because they can be vendored independently.

### Permitted, with caveats recorded in the entry `reason`

- **SuiteSparse, MUMPS** — sparse *direct factorization*. Permitted as linear
  algebra: they factor a matrix you hand them and have no notion of an objective,
  a bound, or a basis. MUMPS is flagged explicitly because it's commonly bundled
  *with* Ipopt — take the factorization, never the solver around it.
- **Intel oneMKL** — permitted as BLAS/LAPACK, but flagged as an x86 vendor lock
  that weakens *hardware* sovereignty. Prefer OpenBLAS/BLIS.
- **oneTBB** — permitted for threads, but the entry warns: do **not** use its
  task scheduler, because ticket #46's Chase-Lev deques and deterministic
  barriers must be ours or determinism becomes untestable.
- **cuDSS** — permitted (factorization only). Its pivot-free stability is a
  Bible §6.5 *numerical* risk, not a sovereignty one; tracked separately.

### Restricted (the full list of 7)

| Entry | Scopes | Why restricted |
|---|---|---|
| `metis` / ParMETIS | `src/l0/**`, `cmake/**` | ordering = LA; structure detection = ticket #14 |
| `scipy` | `tools/`, `tests/`, `benchmarks/` | ships HiGHS-backed `linprog`; fine for harness I/O only |
| `numpy` | + `python/**` | array container; never a core runtime dep |
| `pyomo` | `python/`, `tests/`, `benchmarks/` | modeling frontend we're a backend *for* (#51) |
| `pulp` | same | **the PuLP wheel vendors a CBC executable** — presence tolerated, invocation is an Oracle-rule breach |
| `cvxpy` | same | pulls ECOS/SCS/Clarabel transitively |
| `oracle` | `tools/`, `tests/`, `benchmarks/` | placeholder naming the policy: oracle binaries are CLI-only |

**If you add a dependency:** add its entry to `sovereignty.toml` *in the same
commit*, with a real `reason`. Unclassified ≠ permitted. Then update
`DEPENDENCY_LEDGER.md` prose to mention it (the `--check-ledger` flag fails the
build if the policy names something the ledger never mentions — that's what keeps
the two from drifting).

---

## 4. `DEPENDENCY_LEDGER.md` — what it is

The human-readable half of ticket #1 and its actual deliverable. ~325 lines.
Structure: the constraint and the vanishing test → the three-tier model with the
two motivating cases → forbidden list grouped by category with a one-line verdict
each → restricted table → permitted list → how the check works and why not grep →
a maintenance section with an "Open items" list.

**Open items recorded in the ledger** (carry these forward):
- Transitive *native* deps are only checked when a configured build tree exists.
  Once real third-party targets appear (Phase 2+), CI should configure the build
  before checking so the graph scan runs every commit.
- Python transitive deps aren't lock-file-verified yet. When a `requirements.lock`
  exists, scan it — the PuLP-vendors-CBC case is exactly what a lock file exposes.
- A few forbidden entries are short collision-prone tokens (`dip`, `scs`, `dsp`,
  `clp`). Matched loosely on purpose: a false positive costs one exception, a
  false negative costs the project. Tighten per-entry if noise becomes a burden.

---

## 5. CMake wiring (`cmake/Sovereignty.cmake` + `CMakeLists.txt`)

The check runs in **two** places because they catch different things:

1. **Configure time** (`execute_process` in `Sovereignty.cmake`) — scans
   manifests/source/vendored trees + `--check-ledger`. Fails fast via
   `FATAL_ERROR` before anything compiles.
2. **Build time** (`add_custom_target(sovereignty_check ALL ...)`) — additionally
   passes `--cmake-build-dir` to scan the resolved dependency graph, which only
   exists after configure. **In `ALL` deliberately**: a forbidden dependency must
   break an ordinary local `cmake --build`, not just CI.

Two options to disable if ever needed:
`-DSOVEREIGNTY_CHECK_AT_CONFIGURE=OFF`, `-DSOVEREIGNTY_CHECK_AT_BUILD=OFF`.

`ctest -R sovereignty` runs `test_sovereignty_check.py` as
`sovereignty_check_selftest` (registered under `if(BUILD_TESTING)`).

`CMakeLists.txt` itself: C++20, `CMAKE_EXPORT_COMPILE_COMMANDS ON`, includes
`CTest` at top level, `list(APPEND CMAKE_MODULE_PATH .../cmake)`, then
`include(Sovereignty)`. Nothing else. `cmake_minimum_required(VERSION 3.24)`.

---

## 6. CI (`.github/workflows/sovereignty.yml`)

Triggers on push, PR, manual. Steps, in order:

1. checkout with `submodules: recursive` (a forbidden lib could hide in one)
2. setup-python 3.12 (`tomllib` needs ≥ 3.11 — **the check will not run on
   3.10 or earlier**)
3. **self-test the checker** before trusting it
4. scan tree + `--check-ledger`
5. `cmake -S . -B build -G Ninja`
6. scan resolved dependency graph
7. `cmake --build build` (belt-and-braces: the in-`ALL` target fails here too)

---

## 7. Test suite (`tools/test_sovereignty_check.py`)

38 tests, all green. `python3 tools/test_sovereignty_check.py`. Mostly **negative
tests** — the discipline is that a compliance check must be shown to go *red* on
a real leak, not just green on a clean tree. Groups:

- **TokenisationTests** — `scipy` ≠ `scip`, versioned sonames, `-l` flags, distro
  package names, CMake vars, configure flags, extensions-aren't-names
- **GlobTests** — `**` matches nested + root, `*` doesn't cross `/`, prefix scopes
- **PolicyValidationTests** — empty reason rejected, restricted-without-scope
  rejected, alias claimed by two tiers rejected, wrong schema version rejected,
  unjustified exception rejected
- **EndToEndTests** — clean tree passes; `scipy` import doesn't trip; forbidden in
  CMake manifest / `-l` flags / Python import / C++ `#include` / `<>` include all
  fail; C++ comment discussing a forbidden lib is allowed; Python `#` comment
  still a comment; vendored source drop fails; restricted in-scope passes /
  out-of-scope fails; justified exception suppresses + is path-scoped; depgraph
  catches what the manifest hides; missing build dir is not an error; CMake
  private cache entries ignored but real cache entry still caught; ledger drift
  caught
- **RealRepositoryTests** — this repo is clean with `--check-ledger`; the policy
  classifies the known traps in the right tiers

If you touch `sovereignty_check.py`, run this first. If you add a policy entry,
`test_real_policy_classifies_the_known_traps` may need the new name added.

---

## 8. Verification performed (not just asserted)

Planted real violations in the repo, confirmed the **build** breaks (exit 1),
reverted:

- `find_package(SCIP REQUIRED)` + `-lcbc` in `CMakeLists.txt` → 3 findings, build fails
- `third_party/bliss/graph.cc` with nothing referencing it → vendored-drop finding, fails
- `metis_PartGraphKway()` in `src/l3/structure.cpp` → out-of-scope finding, fails
- **same call** moved to `src/l0/ordering.cpp` → passes (this is the tier working)

`ctest --test-dir build` → 1/1 pass. `--explain nauty` → prints classification.

---

## 9. Environment on the build machine (blocks Phase 2, not #2/#3)

- **No GPU toolchain.** `nvidia-smi` absent, no CUDA toolkit, no ROCm. There is a
  physical GTX 1650 Mobile (Turing) but no usable driver stack.
- Even with drivers, the 1650 is **1/32-rate fp64**. This project's baseline
  correctness is fp64 (Bible §4.1). This machine could validate GPU *correctness*
  but cannot produce meaningful GPU *performance* numbers.
- **No BLAS/LAPACK** in the default lib path (no OpenBLAS, no reference LAPACK).
- Present and fine: gcc 15.2, CMake 4.3.2, Ninja 1.13, Python 3.13.13, git 2.54,
  12-core i5-12450H.

Implication: tickets **#2** (benchmark corpus + CLI oracle) and **#3** (L0
substrate — the CUDA/ROCm abstraction can be stubbed per the ticket) are both
doable here. Anything that needs to *run* a GPU kernel or link BLAS needs either
this machine provisioned or a different one.

---

## 10. Ticket #2 — benchmark corpus + black-box oracle (commit `3178a54`)

### 10.1 What was built

```
benchmarks/
  corpus.toml            manifest: sets, URLs, licences, unpack method
  fetch_corpus.py        stdlib-only downloader + verifier
  README.md
  smoke/                 COMMITTED — 4 tiny hand-verified instances
    tiny_lp.mps          optimal, obj -10/3, unique fractional optimum
    tiny_infeasible.mps  infeasible
    tiny_unbounded.mps   unbounded
    tiny_degenerate.mps  optimal, obj -2, primal-degenerate (anti-cycling target for #4)
    expected.toml        reference answers, each derived by hand (see the .mps headers)
  data/                  gitignored — 90 Netlib LP instances fetched here
  .tools/                gitignored — the compiled `emps` decoder + cached indexes
tools/oracle/
  run_oracle.py          subprocess-only solver wrapper → normalized JSON
  install_highs.sh       builds HiGHS CLI into build-oracle/ (out of tree)
  check_reference.py     solves real instances, checks obj vs Netlib published value
  README.md
cmake/Oracle.cmake       `oracle` / `corpus` / `corpus-verify` targets + CTest wiring
tests/
  test_oracle.py         14 tests (fake-solver parsing + real-HiGHS smoke)
  test_corpus.py         20 tests (manifest, readme parsing, smoke set, lock verify)
```

### 10.2 The corpus

- **Netlib LP is fetched and local**: 90 of ~92 instances in
  `benchmarks/data/netlib_lp/`, each expanded from Netlib's packed format by
  `emps` (a ~250-line C decompressor built into `benchmarks/.tools/`; **no
  optimization logic**, classified `permitted` in the policy).
- **`data/netlib_lp/_reference.toml`** — published optimal objective values
  parsed from the Netlib `readme` PROBLEM SUMMARY TABLE (MINOS 5.3, plus CPLEX
  cross-check values where the readme records a discrepancy). This is the
  oracle-rule *independent answer*: a citation, not a number we computed.
- **`data/netlib_lp/_lock.toml`** — sha256 of every expanded `.mps`.
  `fetch_corpus.py --verify` re-hashes against it (currently 90/90 intact).
- **Known gap:** `stocfor3`, `truss` and any other shell-archive-bundled
  instances are skipped for M0 (they need an `sh` unpack step before `emps`).
- **Deferred sets** (`miplib2017` + isolated `miplib2017_hard`, `mittelmann`,
  `milpbench`, `qplib`): manifest entries with URLs/licence notes exist; the
  fetchers are stubs that print provenance. Implement per milestone
  (MIPLIB→M5, Mittelmann→M5/M9, MILPBench→M7). The `miplib2017_hard` subset is
  `keep_separate = true` — never merge it into MIPLIB stats; it's the M6 evidence.

### 10.3 The oracle

- **`run_oracle.py`** resolves a backend in this order: `--backend` / explicit
  path → `$SOLVER_ORACLE` → `build-oracle/bin/highs` → `highs`/`cbc`/`glpsol`/
  `scip` on PATH. It **only ever `subprocess.run`s** the binary. No solver is
  imported. If none is found → `status: "error"`, never an in-process fallback.
- **Normalized schema `oracle_schema = 1`** (full shape in `tools/oracle/
  README.md`): `status` ∈ {optimal, infeasible, unbounded, infeasible_or_
  unbounded, time_limit, iteration_limit, error}; `objective` is `null` unless
  the solver has a point in hand; `backend.linked = false` is an asserted
  invariant; `threads` defaults to **1** for reproducible references.
- **HiGHS adapter specifics** (pinned `v1.11.0`): that CLI has **no `--threads`
  or `--write_solution_style` flags** — they go through an options file that
  `run_oracle.py` writes. `--parallel` and `--time_limit` are real flags. The
  solution parser targets `write_solution_style = 1` (labelled columnar table:
  `Index Status Lower Upper Primal Dual Name`), tolerating version drift by
  skipping unparseable rows.
- **`build-oracle/`** holds the HiGHS source + build + installed binary. It is
  gitignored **and** in `sovereignty.toml` `exclude_globs`. Delete it and the
  sovereignty check + every test still pass — that's the proof it's scaffolding.
  Rebuild with `tools/oracle/install_highs.sh` or `cmake --build build --target
  oracle`.
- **Verified pass condition:** `check_reference.py` solved 10 Netlib instances
  (afiro, sc50a/b, adlittle, blend, stocfor1, degen2, bandm, share2b, beaconfd)
  and all 10 objectives matched the published reference to tolerance.

### 10.4 Sovereignty-check changes in this ticket (all with regression tests)

| Change | Why |
|---|---|
| `exclude_globs` now prunes whole subtrees (`build-oracle/**`, `benchmarks/data/**`, `benchmarks/.tools/**`) in both the file walk and the vendored-dir walk | a prefix like `build-oracle/**` must exclude the dir itself, not only its contents; `dir_is_excluded()` handles that |
| Exceptions match by **resolved policy entry**, not raw string | an exception keyed `token = "glpk"` now also covers a hit that matched via the alias `glpsol` |
| Scoped exceptions for `highs`/`cbc`/`glpsol`/`scip` in `tools/oracle/**` + `tests/test_oracle.py` | the oracle wrapper's job is to *name and run* those executables; the names stay hard violations everywhere else |
| Depgraph scan strips first-party paths + CMake `DESC =`/`COMMENT =` prose before tokenizing | our own `oracle` custom target (which runs `install_highs.sh` and has a "Building HiGHS CLI" comment) was self-flagging; a real external `libscip.so` on a link line still fails (tested) |
| Removed the placeholder restricted `oracle` token | the Oracle rule is now enforced concretely (excluded path + scoped exceptions + ledger §5.5); the fake token only false-positived on the English word "oracle" |
| `emps` classified `permitted`; `cplex` excepted in `fetch_corpus.py` | `emps` is a format decompressor; the `cplex` hit is parsing a *citation* in the Netlib readme, not a dependency |

Policy totals now: **76 dependencies — 37 forbidden, 6 restricted, 33 permitted**
(was 7 restricted / 32 permitted before the `oracle`→removed, `emps`→added swap).

### 10.5 How to test ticket #2 (see §12)

`ctest --test-dir build` runs everything (7 suites). Individual:
`python3 tests/test_oracle.py`, `python3 tests/test_corpus.py`,
`python3 tools/oracle/check_reference.py`.

---

## 11. Ticket #3 — L0 numerical substrate (commit `efd3a44`)

### 11.1 What was built

```
include/sovereign/
  numeric.hpp     scalar + index policy, tolerances, close()
  status.hpp      Status enum, Error, check()
  sparse.hpp      CsrMatrix, CscMatrix, Triplet, VbcsrMatrix/BsrMatrix stubs
  backend.hpp     Backend interface, DeviceBuffer, DeviceCsr, factory
src/l0/
  sparse.cpp        containers, validation, CSR<->CSC
  backend.cpp       factory + owning device handles
  backend_host.cpp  full reference implementation (always compiled)
  backend_cuda.cu   cuSPARSE/cuBLAS   -- compiled only if CUDA found
  backend_hip.cpp   rocBLAS skeleton  -- compiled only if ROCm found
cmake/DetectAccelerators.cmake   toolkit detection + configure summary
tests/
  test_support.hpp     ~120-line harness (no GoogleTest: avoids a CI network dep)
  test_l0_sparse.cpp   18 cases
  test_l0_backend.cpp  21 cases, run over EVERY compiled backend
tools/
  check_backend_parity.py  interface-drift guard for the uncompiled GPU backends
  verify_phase0.sh         Phase 0 gate against the Build Map's "Done when"
```

### 11.2 Design decisions worth knowing

**Three backends, not two.** Host is a complete reference implementation, not a
placeholder. It plays the same role for the GPU kernels that the from-scratch
CPU simplex (#4) plays for the GPU solvers: the independent answer. It also
makes the abstraction falsifiable — an interface with one implementation is a
fiction shaped around that implementation, and every call site is now exercised
by more than one.

**VRAM residency is structural.** `DeviceCsr` and `DeviceBuffer` are move-only,
so a deep copy of the constraint matrix does not compile. Every backend counts
its host↔device transfers, so `backend.matrix_crosses_the_bus_exactly_once`
*asserts* residency rather than assuming it (3 uploads for the 3 CSR arrays,
then 32 SpMVs with zero further traffic).

**CSR and CSC are peers.** PDHG needs `Ax` and `Aᵀy` every iteration, so
`spmv_transpose` is a primitive, not `spmv` on a materialized transpose — that
would double the resident footprint of the largest object in the solver.

**`beta == 0` overwrites, never scales.** `0 * NaN` is `NaN`. A `y` buffer
holding a NaN from a previous failed solve would otherwise poison every
subsequent iteration, surfacing far from its cause. Tested explicitly.

**Index width is a switch** (`-DSOVEREIGN_INDEX64=ON`). int32 halves index
bandwidth in a memory-bound SpMV and is what cuSPARSE wants; the PS asks for
instances that can exceed 2³¹ nonzeros. CI builds and tests both.

**No GoogleTest.** Permitted, but pulling it in means a FetchContent download
into the build tree and a network dependency in CI to replace ~120 lines. Swap
when the suite outgrows `test_support.hpp`; nothing depends on the harness
beyond its macros.

### 11.3 The GPU gap — read this before trusting anything GPU-shaped

**`backend_cuda.cu` and `backend_hip.cpp` have never been compiled or run.**
This machine has no CUDA toolkit and no ROCm (§9). They are written against the
documented cuSPARSE/cuBLAS/rocBLAS APIs and must be treated as **unverified**
until a build with a toolkit says otherwise. Expect them not to compile first
time.

Two things mitigate it, neither of which is a substitute:

1. **`DetectAccelerators.cmake` never lies about what was built.** The configure
   summary states which backends were compiled and why not, and
   `-DSOVEREIGN_CUDA=ON` with no toolkit is a hard `FATAL_ERROR` rather than a
   silent host fallback — a benchmark claiming "GPU" while running on the CPU
   would invalidate every number above it.
2. **`tools/check_backend_parity.py`** catches the one defect class that *is*
   detectable without a toolkit: interface drift. Add a method to `Backend` and
   the compiler flags Host while saying nothing about the two GPU backends. This
   parses `backend.hpp` for pure virtuals and confirms each backend overrides
   them (declining with `Status::Unsupported` is fine; being absent is not).
   Registered as the `l0_backend_parity` ctest.

**When you get a GPU machine:** `cmake -S . -B build -DSOVEREIGN_CUDA=ON`, fix
the compile errors, then run `./build/tests/test_l0_backend` — the existing 21
cases become the host-vs-device differential test with no edit, because
`for_each_backend` already loops over everything compiled.

### 11.4 Tests can actually fail (verified)

39 C++ cases passing on the first run is suspicious for numerical code, so three
deliberate sabotages were run and reverted:

| Sabotage | Caught by |
|---|---|
| off-by-one in the SpMV row loop (`k + 1 < end`) | 4 cases red |
| `beta == 0` scales instead of overwriting | the NaN case red |
| backend pretends to re-upload per SpMV | residency case red, with its own message |

### 11.5 Sovereignty change in this ticket

**Shell scripts are now scanned** (`**/*.sh`, `**/*.bash`). A setup script is
exactly where a forbidden dependency enters without touching any manifest — a
`git clone`, an `apt install`, a downloaded tarball. Verified: a `git clone
.../Cbc.git` in a `.sh` is now caught. `install_highs.sh` stays silent (covered
by the `tools/oracle/**` scope); `verify_phase0.sh` needed exceptions because it
plants violations on purpose.

---

## 12. Phase 0 gate

`tools/verify_phase0.sh` checks tickets #1/#2/#3 against the Build Map's own
"Done when" wording rather than against "the tests pass" — those are different
claims. It re-plants a forbidden dependency in a temp directory to confirm the
checker still fails closed, and reports anything unverifiable as **NOTE** rather
than folding it into a pass.

Current result on this machine:

```
Phase 0: 14 passed, 0 failed, 2 not verifiable here
```

The two NOTEs are the uncompiled CUDA and HIP backends. They are real gaps
carried forward, not passes.

---

## 13. Tickets #5, #4, #6 — parser, simplex, preconditioning (Phase 1, M0)

Built in dependency order: #5 first (the simplex test *is* "solve Netlib", which
needs a reader), then #4, then #6.

### 13.1 What was built

```
include/sovereign/
  problem.hpp     canonical internal model: minimize/maximize, two-sided row
                  bounds (lo <= Ax <= hi), column bounds, integrality, names
  io.hpp          read_mps / read_lp / read_model, ReaderWarning, MpsOptions
  simplex.hpp     Simplex, SimplexOptions, SimplexResult, SolveStatus
  scaling.hpp     Scaling (Ruiz + Pock-Chambolle), ScalingOptions, Conditioning
src/io/
  problem.cpp       model container + name-addressed Builder
  mps_reader.cpp    free-form MPS, with a fixed-column fallback
  lp_reader.cpp     CPLEX LP subset
src/l1/
  simplex.cpp       revised primal simplex, bounded-variable, two-phase, EXPAND
  scaling.cpp       equilibration + solution mapping
apps/
  sovereign_cli.cpp   `sovereign-cli info|solve [--scale] [--json] <model>`
tests/
  test_mps_reader.cpp  24 cases   test_lp_reader.cpp  16 cases
  test_simplex.cpp     20 cases   test_scaling.cpp     6 cases
tools/
  check_parser_vs_oracle.py    rows/cols/nnz vs the oracle's own parse
  check_simplex_vs_oracle.py   objective vs the oracle (not the stale readme)
```

### 13.2 Ticket #5 — the parser

**Canonical form is two-sided row bounds** (`lo <= Ax <= hi`). MPS row senses
(`L`/`G`/`E`), the `RANGES` section, and free rows all become `(lo, hi)` pairs
with infinities marking the one-sided cases — so `RANGES` is an assignment, not
a special case, and it is the form the presolve (#13) and both continuous
engines want anyway.

Quirks handled explicitly, each with a test that fails on the wrong behaviour:

| Quirk | Rule implemented |
|---|---|
| First `N` row is the objective; **later `N` rows are free rows** | objective is never a row of `A`; free rows are kept so counts match the file |
| RHS entry **on the objective row** | it is the **negated** objective constant (a sign error shifts every reported objective by `2d`) |
| `RANGES` on `E` rows | **sign-dependent**: `R>=0 -> [b, b+R]`, `R<0 -> [b+R, b]`; `L`/`G` use `|R|` |
| `BOUNDS` `UP` with a negative value on a default-`[0,inf)` column | lower bound goes to `-inf` (the established-solver convention); **recorded as a warning**, and `MpsOptions` can select the literal reading |
| Duplicate `(row, col)` coefficients | summed |
| `MARKER`/`INTORG`/`INTEND` | integral, but bounds stay `[0, inf)` — not silently `[0,1]`, which would cut off the optimum of every general-integer model |
| Semi-continuous (`SC`), `QSECTION`, SOS | **refused or warned, never silently reinterpreted** |

**The fixed-column lesson.** The header comment first claimed no Netlib
instance uses embedded spaces in names. `forplan` does (`DEDO3 1R`), and
free-form tokenization split it into a duplicate `DEDO3`. Fix: a free-form
parse failure triggers a retry under fixed-column field offsets (2-3, 5-12,
15-22, 25-36, 40-47, 50-61), and the fallback is recorded as a warning. This is
exactly the "silently builds a different problem" failure the ticket warns about
— had the split names stayed unique it would have surfaced much later as a
numerical mystery.

**Pass condition met on the whole corpus, not a sample:**
`check_parser_vs_oracle.py` → **90/90 Netlib instances match the oracle's rows,
columns and nonzeros exactly.**

### 13.3 Ticket #4 — the from-scratch simplex

**A revised _primal_ simplex, not the dual the Build Map names.** The reasoning
is the oracle role: dual simplex needs a manufactured dual-feasible starting
basis whose bugs would be indistinguishable from the bugs it is meant to catch.
The primal two-phase method starts from the slack basis, which always exists, so
correctness is checkable end to end. The dual variant belongs with ticket #38,
where warm-starting is what actually needs it.

- **Computational form** `[A -I][x; s] = 0` with a bound pair on every variable
  — a starting basis (`-I`) always exists, and ranges/equalities/free rows are
  just bound pairs.
- **Dense LU of the basis** with partial pivoting, refactorized every 100
  pivots, product-form (eta) updates between. Deliberately capped at
  `kMaxDenseRows = 3000`: sparse LU with Markowitz pivoting is the real answer
  and is named as follow-up, not half-attempted. Larger instances are refused
  with a clear message, not ground through.
- **Anti-cycling is EXPAND** (Gill, Murray, Saunders & Wright 1989), added
  during this work after `brandy` cycled forever. A working feasibility
  tolerance grows every pivot and resets at each refactorization; because it is
  strictly increasing between resets, the relaxed feasible region differs every
  iteration and a basis cannot recur — the ratio test always has room for a
  strictly positive step, even at a fully degenerate vertex. Bland's rule is
  kept as a cheap secondary net. The earlier bug: an absolute-plus-relative
  pivot floor big enough to be numerically safe also excluded small-but-real
  pivots from the ratio test, which quietly voided Bland's guarantee.
- **EXPAND endgame.** The relaxed ratio test can declare optimality a few `1e-7`
  outside a true bound; left alone that leaked `~1e-5` into the reported
  objective (`scsd1`, `wood1p`). So after Phase 2 converges, expansion is
  switched off, the basis is rebuilt exactly, and a bounded burst of exact
  pivots refines the point to a true vertex.
- **Progress-stall net.** If the merit function (infeasibility in Phase 1,
  objective in Phase 2) does not improve for `~40*(m+n)` pivots, refactorize
  once and, failing that, stop with `NumericalFailure` — never a wrong
  `Optimal`.

**Verification (both halves of the ticket's pass condition):**
- `test_simplex` — 20 hand-checked cases, including **Beale's 1955 cycling
  example** (the canonical anti-cycling test) and a primal-degenerate optimum,
  both terminating; plus a test that the reported *point* is feasible, not just
  that the objective number matches.
- `check_simplex_vs_oracle.py --max-rows 250` → **30/30 agree with the oracle**
  (26 Netlib + 4 hand-verified smoke), `brandy` among them.
- **The 1988 Netlib readme is stale for two instances** (`e226`, `scagr7`):
  our answer and the oracle's agree with each other and differ from the printed
  table. The harness flags this rather than picking a winner — the running
  oracle is the reference (Oracle rule), a 37-year-old table is not.

### 13.4 Ticket #6 — preconditioning

`Scaling::equilibrate` runs **Ruiz** (iterative row/column inf-norm
equilibration toward 1) then **Pock-Chambolle** (the diagonal step-size
preconditioner PDHG needs) on top. Works on a `Problem` — rescaling the matrix,
the objective, and the row/column bounds consistently — and carries the factors
so a scaled solve maps back to the original answer. The arithmetic is
row/column reductions and an elementwise divide, i.e. the L0 backend
primitives; this runs on the host `CsrMatrix` (the host is a backend), and the
device version is the same formulas through the backend calls.

**Pass condition met:**
- `test_scaling` — on a matrix with coefficients spanning `1e-6` to `1e+6`, the
  row- and column-norm spread collapses to `< 1.5`; and the scaled problem
  solved through the simplex returns the original optimum (objective and point)
  to `1e-6`.
- End to end via the CLI: `sovereign-cli solve grow22.mps --scale` cuts
  iterations **1263 -> 606** for the same optimum; `scfxm1` 590 -> 433;
  `pilot4` 3941 -> 3535.

### 13.5 The CLI (seed of #50)

`sovereign-cli info <model> [--json] [--warnings]` — shape, row-type histogram,
coefficient spread, reader warnings.
`sovereign-cli solve <model> [--json] [--scale] [--bland] [--refactor N]
[--time-limit S] [--max-iterations N]` — solve the LP relaxation, report status
/ objective / iterations / anti-cycling activity.

---

## 14. M0 status

**M0 is complete.** Its benchmark pass condition — *"from-scratch CPU simplex
solves Netlib to tolerance vs. oracle"* — is green, and every M0-tagged ticket
(#1-#6) is built and verified.

`tools/verify_phase0.sh` → **21 passed, 0 failed, 2 not verifiable here.** The
two NOTEs are the uncompiled CUDA and HIP backends (no toolkit on this machine);
they are carried gaps, not passes. `ctest` → **13/13**.

Remaining Phase-0-adjacent gap, unchanged: `stocfor3` and `truss` are not in the
local corpus (shell-archive packed format, skipped for M0).

---

## 15. Immediate next steps

Phase 1 is done. Phase 2 — the GPU continuous core (M1) — has #7 (§17), #8
(§18), #9 (§19), and #10 (§20) built: three engines solve LPs, race each
other, and agree with the oracle on this machine's Host backend. **M1's own
gate is not yet closed** — it also names "beat CPU wall-clock on a
large-network instance," which needs a real GPU run to mean anything (§9,
§17.2); everything checkable without one is green. Phase 3 (crossover,
#11/#12) is next.

**This is where the missing toolchain finally bites.** #8 and #9 are GPU
engines; there is no CUDA/ROCm and no BLAS on this machine (§9, §11.3). Options:
provision this box, use a different one, or build the PDHG/IPM math against the
**host backend** first (it is a real backend, and `Backend` already has `spmv`,
`axpy`, `dot`, `project_box`, `norm2`) and treat the GPU port as a later step —
the abstraction was built precisely so that is possible.

**When adding a dependency** (OpenBLAS, cuSPARSE, …): classify it in
`sovereignty.toml` **and** `DEPENDENCY_LEDGER.md` in the same commit, and extend
`test_sovereignty_check.py`'s `RealRepositoryTests` list.

---

## 16. What you can test right now

Green on this machine as of the tickets #4/#5/#6 commit.

```sh
cd /home/arch_btw/Documents/SIH
rm -rf build && cmake -S . -B build -G Ninja
cmake --build build            # in-ALL sovereignty check runs here
ctest --test-dir build         # 13 suites

./tools/verify_phase0.sh       # 21 passed, 0 failed, 2 not verifiable

# the parser and simplex, against the oracle
python3 tools/check_parser_vs_oracle.py            # PASS: 90/90
python3 tools/check_simplex_vs_oracle.py --max-rows 250   # PASS: 30/30

# individual C++ suites
./build/tests/test_mps_reader ./build/tests/test_lp_reader
./build/tests/test_simplex ./build/tests/test_scaling

# the CLI
./build/apps/sovereign-cli info  benchmarks/data/netlib_lp/forplan.mps --warnings
./build/apps/sovereign-cli solve benchmarks/data/netlib_lp/brandy.mps
./build/apps/sovereign-cli solve benchmarks/data/netlib_lp/grow22.mps --scale

# ticket #7 — Farkas certificate + complementary slackness
./build/tests/test_duals

# ticket #8 — PDHG, on the Host backend (no GPU needed to validate the engine)
./build/tests/test_pdhg
./build/apps/sovereign-cli solve benchmarks/data/netlib_lp/afiro.mps --engine pdhg
python3 tools/check_pdhg_vs_oracle.py --max-rows 100 --timeout 30

# ticket #9 — the pivoting-free IPM, dense SQD LDL^T
./build/tests/test_ipm
./build/apps/sovereign-cli solve benchmarks/data/netlib_lp/afiro.mps --engine ipm
python3 tools/check_ipm_vs_oracle.py --timeout 60

# ticket #10 — the concurrent engine race
./build/tests/test_race
./build/apps/sovereign-cli solve benchmarks/data/netlib_lp/afiro.mps --engine race
```

**Not yet testable on this machine:** the GPU continuous core (#8, #9) — needs
a toolchain. (A CUDA 12.4 toolkit + gcc-10 were provisioned, no root, on a
separate RTX 2080 Ti machine to unblock ticket #3's CUDA backend — see §17.2 —
but that machine is not this one and is not always available.)

---

## 17. Ticket #7 — Farkas/dual-ray & duals plumbing (Phase 2, M1 begins)

### 17.1 What was built

```
include/sovereign/duals.hpp   FarkasCertificate, DualSolution
src/l1/duals.cpp              the range-arithmetic certificate checker
tests/test_duals.cpp          7 cases
```

`SimplexResult` gained a `FarkasCertificate farkas` field, populated whenever
`solve()` returns `Infeasible` from a real Phase I search (not the trivial
crossed-bounds-before-any-solve path, which has no basis to take a dual from).

**The contract, not just the simplex's use of it.** The point of this ticket
is a contract every continuous engine implements the same way — PDHG (#8) and
the IPM (#9) populate the same `FarkasCertificate`/`DualSolution` shapes later,
so Benders (#26) and conflict-cut derivation (#43) consume one interface
regardless of which engine solved the subproblem.

**The certificate is derived from scratch, not asserted.** For row multipliers
`y`, the identity `sum_i y_i (Ax)_i == sum_j (A^T y)_j x_j` holds for *any* x.
Bounding each side independently through the row bounds and the column bounds
gives two ranges for the same quantity; if they don't overlap, no feasible x
can exist. `FarkasCertificate::certifies_infeasibility()` computes both ranges
and checks disjointness — an arithmetic proof, checkable without re-solving
and without trusting whichever engine produced `y`. Worked by hand in
`duals.hpp`'s own comment against the `tiny_infeasible` smoke fixture
(`x >= 2, x <= 1`): `y = (1, -1)` forces row range `[1, ∞)` against column
range `{0}` — disjoint, contradiction confirmed.

**Where the simplex gets `y`.** At the point Phase I terminates with residual
infeasibility, the Phase I dual (`price()`'s `duals_` side effect against the
*live*, unfrozen Phase I cost — verified frozen-Bland state has already
thawed by the time Phase I can return Optimal) is exactly this certificate.
No new algorithm — this reads a value the simplex already computes internally
and had never exposed.

**`DualSolution::complementary_slackness_violation`** is the reusable form of
the check `SimplexImpl::run()` already did informally: a row dual must be
zero unless that row's activity sits at a bound, a reduced cost must be zero
unless the column does. Takes the primal point, the duals, and the problem —
engine-agnostic on purpose.

### 17.2 GPU box provisioning (unblocks ticket #3's CUDA backend, no code change)

Not part of ticket #7's own scope, but done in the same session: got SSH
access to a machine with an RTX 2080 Ti and compiled `backend_cuda.cu` for
the first time ever. Two real, non-obvious fixes, both committed:

- `src/CMakeLists.txt`: `CUDA_RESOLVE_DEVICE_SYMBOLS ON` on `sovereign_l0`.
  Needed because the library is `STATIC` with `CUDA_SEPARABLE_COMPILATION
  ON`, and CMake does not device-link a static library's separable-compiled
  objects unless something tells it to — a plain C++ executable (no `.cu`
  sources of its own, like `sovereign-cli` or any test binary) linking
  against it otherwise fails at the final link with undefined
  `__cudaRegisterLinkedBinary_*` references.
- `sovereignty.toml`: a scoped exception for `CUDA_cusolver_metis_static_LIBRARY`
  in the resolved dependency graph. `find_package(CUDAToolkit)` caches a
  variable for every optional library of every component it *can* find,
  including cuSOLVER's optional METIS dependency, regardless of whether the
  project asked for that component — this project links only `CUDA::cudart`,
  `CUDA::cublas`, `CUDA::cusparse`. Confirmed by the actual link line before
  adding the exception, scoped strictly to the synthetic `<depgraph>/` path.

Both the CUDA toolkit (12.4) and a working host compiler (gcc-10, needed for
`<span>` — the system default gcc 9.4's libstdc++ predates it) were installed
to that machine's home directory with **no root**, via `apt-get download` +
`dpkg-deb -x` for the compiler and the official runfile installer's
`--toolkit`-only mode for CUDA. Verified there: `test_l0_backend` 21/21,
`ctest` 13/13, sovereignty check clean including the resolved dependency
graph. That machine is Turing (RTX 2080 Ti) — same 1/32-rate fp64 caveat as
the GTX 1650 noted in §9: good for GPU *correctness* validation, not GPU
*performance* claims, exactly as documented there.

---

## 18. Ticket #8 — PDHG / PDLP engine (Phase 2, M1)

### 18.1 What was built

```
include/sovereign/pdhg.hpp   PdhgOptions, PdhgResult, Pdhg
src/l1/pdhg.cpp               the iteration, step-size estimate, restarts
tests/test_pdhg.cpp           7 cases
tools/check_pdhg_vs_oracle.py oracle comparison at PDHG's own tolerance
apps/sovereign_cli.cpp        `solve --engine pdhg`
```

Runs entirely through the L0 `Backend` abstraction (`DeviceCsr`/`DeviceBuffer`,
`spmv`/`spmv_transpose`/`axpy`/`scale`/`project_box`/`dot`/`norm2`) — on this
machine that means the Host backend, but nothing in the algorithm is
Host-specific; the same code runs on Cuda once §17.2's toolkit is available
to build with.

**The derivation (pdhg.hpp carries the full version).** Introduce a slack
`s := Ax` with the row bounds moved onto it, dualize the resulting equality
with multiplier `y`, and apply Chambolle-Pock to the saddle point. The result
is exactly "SpMV + elementwise clamp" per iteration:
```
x <- Proj_X( x - tau (c - A^T y) )
s <- Proj_S( s - tau y )
y <- y + sigma ( (2s-s_old) - A(2x-x_old) )
```
The sign convention (`c - A^T y`, not `c + A^T y`) was chosen deliberately to
match ticket #7's `DualSolution` exactly, so PDHG's `y`/reduced costs need no
translation before feeding #10's engine race or #26's Benders later — verified
by a dedicated test (`reduced_costs_use_the_same_sign_convention_as_the_simplex`).

**Step size is measured, not assumed.** Ruiz + Pock-Chambolle (#6) runs as
the mandatory front-end Bible §6.2 requires, but Scaling's diagonal formula
was derived for `A` alone and doesn't account for the slack block's identity
column in the actual saddle-point operator `K = [-A | I]` — using it directly
as a tau/sigma bound would be a hair optimistic. Instead `estimate_operator_norm`
runs ~30 steps of power iteration on `K` itself, using the same `spmv`/
`spmv_transpose` primitives as the main loop (no new Backend primitive
needed), and `tau = sigma = 0.9 / ‖K‖_est`. Honest, and it costs a handful of
extra SpMVs once at setup.

**Restarts are real, not a stub.** A Cesaro average of `(x, y)` since the
last restart is tracked; every `restart_check_period` iterations the average's
KKT residual is compared against the residual at the last restart, and a
sufficient decrease (default: half) triggers a restart to the average. Fires
in practice — 6 restarts on `blend`, 1 on `sc50a`/`sc50b`, 6 on `share2b` —
and in each case the run went on to actually converge, which is what "the
restarts are what rescue it" (Bible §4.2 Engine A) has to mean in practice,
not just that a counter increments.

**Two real bugs caught before this shipped, both worth knowing about:**
- *Scaled-space convergence checking was not enough.* Unscaling divides
  column `j`'s reduced cost by `col_scale_j`; a residual safely under
  tolerance in scaled space can land anywhere after unscaling depending on
  how aggressive that column's scale factor was. Fixed by adding
  `original_space_residual`, which unscales the *candidate* point and checks
  it against the real problem before declaring convergence — scaled-space
  `kkt_residual` is now used only for the restart decision, where a relative
  comparison in a consistent space is all that's needed.
- *The dual residual only ever checked columns.* A non-binding row's dual
  must be ~0 at any true fixed point (a one-line argument from the `s`
  update: an interior `s` forces `tau*y = 0`), but nothing was checking it —
  so `blend`/`adlittle` could report "optimal" while `y` was still very wrong
  on rows that weren't binding. Both residual functions now check the row
  side too. `tests/test_duals.cpp` didn't need to change (ticket #7's
  `DualSolution::complementary_slackness_violation` already checked rows
  correctly) — this was specifically PDHG's own internal convergence check
  that was incomplete, not the shared contract.

### 18.2 Verification

- `test_pdhg` — 7 hand-checked cases (mirroring `test_simplex.cpp`'s
  discipline): a two-variable optimum, a lower-bound-forced optimum, an
  equality row, a column-bound-forced optimum, a degenerate case, a
  restart-firing case, and the sign-convention check above.
- `check_pdhg_vs_oracle.py`: **10/13 Netlib instances agree with the oracle**
  to 5e-3 relative / 1e-3 absolute (afiro, adlittle, blend, sc50a, sc50b,
  scsd1, share2b, recipe, plus both feasible smoke instances). `adlittle`
  needed ~394k iterations to actually reach tolerance, `share2b` ~774k —
  both verified to get there, which is why `PdhgOptions::max_iterations`
  defaults to 1,000,000, not the original 200,000.
- **Two honest, tracked gaps, not hidden:** `kb2` still hasn't converged at
  1,000,000 iterations (needs more — not yet quantified how many), and
  `fit1d`/`fit2d` (25 rows, 1026/10500 columns — extremely wide) time out at
  the check script's default 30s budget. Both are instances of PDHG's
  documented slow tail (Bible §4.2 Engine A, §10.1), not a correctness
  defect — every instance that *does* converge matches the oracle. Widening
  `--timeout`/`--max-iterations` on these specific instances is a reasonable
  next probe, not yet done.
- **PDHG has no infeasibility or unbounded detection.** Both are caught at
  the crossed-bounds level only (same as Simplex's trivial case); a genuinely
  infeasible or unbounded LP handed to PDHG will exhaust its iteration budget
  rather than terminate early with the right status. Named explicitly rather
  than silently returned as `iteration_limit` with no explanation — tracked
  as a real gap for whenever PDHG needs to stand fully on its own rather than
  racing the simplex (#10).
- `ctest`: 15/15. Sovereignty check: clean, including the resolved dependency
  graph (`check_pdhg_vs_oracle.py` needed the same scoped `highs` exception
  the other oracle-comparison scripts already have — added to its `paths`).

---

## 19. Ticket #9 — regularized pivoting-free IPM (Phase 2, M1)

### 19.1 What was built

```
include/sovereign/ipm.hpp   IpmOptions, IpmResult, Ipm
src/l1/ipm.cpp               Mehrotra predictor-corrector + dense SQD LDL^T
tests/test_ipm.cpp           9 cases
tools/check_ipm_vs_oracle.py oracle comparison at tight tolerance
apps/sovereign_cli.cpp       `solve --engine ipm`
```

**The derivation (ipm.hpp carries the full version).** Same `z = (x, s)`,
`M = [A, -I]` convention as simplex/PDHG. For each `z_j` with a finite lower
bound, track `g_j = z_j - lo_j >= 0` and multiplier `lambda_j`; for a finite
upper bound, `h_j = hi_j - z_j >= 0` and multiplier `zeta_j`. Linearizing the
barrier-perturbed KKT conditions and eliminating `dlambda`, `dzeta` in favor
of `dz` (textbook bounded-variable IPM reduction, re-derived from scratch,
not lifted from any specific solver) collapses to one symmetric system:
```
[ -(Theta^-1 + delta_x I)      M^T          ] [dz]   [rhs_z]
[  M                            delta_y I    ] [dy] = [rhs_y]
```
where `Theta_j^-1 = lambda_j/g_j + zeta_j/h_j`. Without `delta_x, delta_y`
this is the classical augmented IPM system — negative semi-definite (1,1)
block, zero (2,2) block, exactly where a numerical pivot search would
normally be needed. Adding those two regularization terms makes both blocks
strictly definite: this **is** the Symmetric Quasi-Definite (SQD) system the
ticket asks for, which Vanderbei's 1995 result says factors via `LDLᵀ` in
**any** diagonal order — including the natural, unpermuted one, which is
what "no pivoting on the critical path" means concretely here. Full Mehrotra
predictor-corrector (affine step → centering parameter from `(mu_aff/mu)^3`
→ corrector with the second-order `dz_aff·dlambda_aff` cross term) reuses one
factorization per iteration for both solves.

**Scope, mirroring ticket #4's own precedent exactly.** Dense, from-scratch,
no-pivoting `LDLᵀ` of the `(n+2m)×(n+2m)` augmented matrix, refactorized
every iteration (Theta changes every step; there's no eta-file analogue
here), capped at the same `kMaxDenseSize` discipline as simplex's
`kMaxDenseRows` — refuses cleanly past the cap rather than grinding. Sparse
`LDLᵀ` (cuDSS on GPU, or a from-scratch Markowitz-ordered CPU version) is
named follow-up work, not attempted here, exactly as sparse LU was ticket
#4's named follow-up.

### 19.2 Four real bugs caught before this shipped

The ticket's own "Watch out" calls the regularization schedule "the single
most likely place to compute a silently-slightly-wrong answer" — that
warning earned its keep. All four were caught by testing against the oracle
on real Netlib instances, not by inspection:

1. **Initial-point inconsistency for zero/narrow-gap bounds.** `g_j` and
   `h_j` are not independent — `dg_j = dz_j = -dh_j`, so their *sum* is fixed
   at `hi_j - lo_j` for the whole solve. An earlier version floored each
   independently to at least 1.0 without checking that sum, which silently
   pushed `z_j` outside its own box for any gap narrower than 2 — equality
   rows (gap = 0) included. The convergence check only ever watched `Mz = 0`
   and complementarity, never "is z itself still in its box," so this
   corrupted `afiro`, `blend`, `sc50a/b`, `scsd1`, `recipe`, `share2b` (every
   real instance with an equality row) while converging to tiny residuals at
   a **wrong** point. Fixed by splitting the actual gap in half, floored only
   when the gap itself is near zero — `tests/test_ipm.cpp`'s `equality_row`
   and `fixed_variable` cases are the regression tests for this specifically.
2. **No Ruiz/Pock-Chambolle front-end.** Bible §6.2's "mandatory front-end to
   L1" was applied to PDHG (#8) but an earlier version of this engine never
   scaled the problem at all. `adlittle` — a known badly-scaled Netlib
   instance — responded by sending `mu` past `1e9` instead of converging.
   Fixed by calling `Scaling::equilibrate` exactly as PDHG does, unscaling
   the result at the end.
3. **Factorization breakdown deep into convergence, with no recovery.**
   `recipe` hit a near-zero pivot at iteration 14, with `mu` already down to
   `~7e-9` — Theta⁻¹ entries that large make the elimination's floating-point
   cancellation genuinely fragile even though the SQD guarantee holds in
   exact arithmetic. Fixed with a backoff: on a factorization failure,
   multiply `delta` by 10 and retry (up to 6 times) before giving up — the
   direct, standard answer to the ticket's own "too little regularization"
   framing.
4. **No best-iterate tracking.** On `adlittle` specifically, pushing the
   iteration budget past ~200 showed `mu` overshooting to `~1e-32` — deep
   into floating-point noise relative to the O(1) quantities `Theta^-1` is
   built from — after which the *next* Newton step was unreliable and moved
   measurably away from the optimum, even though the point at iteration
   ~199 was already correct. Fixed by tracking whichever iterate had the
   best combined residual across the whole run and reporting that, not
   whatever the last step happened to produce — standard, robust IPM
   practice for exactly this late-stage fragility.

### 19.3 Verification

- `test_ipm` — 9 hand-checked cases, including dedicated regressions for the
  equality-row and fixed-variable bugs above, and a complementary-slackness
  check reusing ticket #7's `DualSolution` directly.
- Hand-verified models converge to `~1e-12`–`1e-15` residuals in 5-9
  iterations — dramatically tighter than PDHG, which is the entire point of
  this engine.
- `check_ipm_vs_oracle.py` on the full Netlib corpus at the (initial) 200-
  iteration default: **40/60 checkable instances agreed with the oracle**
  (31 correctly refused via the documented dense-size cap, not a
  disagreement). Of the 20 remaining, spot-checking `agg`, `agg2`, `bandm`,
  `beaconfd`, `lotfi`, `tuff` at a higher iteration cap (`/tmp/ipm_more`,
  ad hoc — not a committed tool) confirmed they converge to the **correct**
  answer given enough iterations (`bandm` ~400, `lotfi` ~1300) — genuinely
  budget-limited, not a correctness bug. `max_iterations` default raised
  from 200 to 1500 on that evidence.
- **One honest, tracked gap:** `forplan` plateaus with `primal_res ≈ 94` even
  at 2000 iterations and does not reach the correct objective. Checked for
  the obvious suspect (free/`MI`/`PL` columns interacting badly with the
  regularization floor) and ruled it out — `forplan` has no free columns,
  just 21 `UP` and 3 `FX` bounds beyond the MPS default. The actual cause is
  not yet identified; carried forward as a real gap rather than hidden,
  same discipline as PDHG's `kb2`/`fit1d`/`fit2d` notes in §18.
- `ctest`: 16/16. Sovereignty check clean, including the resolved dependency
  graph (`check_ipm_vs_oracle.py` added to the `highs` exception's `paths`,
  same pattern as the other oracle-comparison scripts).

---

## 20. Ticket #10 — the concurrent engine race (Phase 2, M1)

### 20.1 What was built

```
include/sovereign/cancellation.hpp   CancellationToken (shared by all 3 engines)
include/sovereign/race.hpp           RaceOptions, RaceResult, race_solve
src/l1/race.cpp                      the harness itself
tests/test_race.cpp                  7 cases
apps/sovereign_cli.cpp               `solve --engine race` (+ --no-simplex/-pdhg/-ipm)
```

**Cooperative cancellation, added to all three existing engines.** Each of
`SolveStatus`, `PdhgStatus`, `IpmStatus` gained a `Cancelled` value, and
`Simplex::solve` / `Pdhg::solve` / `Ipm::solve` all gained an optional
`const CancellationToken*` checked once per pivot/iteration — a single
atomic load, negligible next to an SpMV or a dense factorization. This is
deliberately minimal surgery on three already-tested engines rather than a
parallel "cancellable" code path: same algorithm, same tests, one extra
early-exit check.

**The harness (race.hpp carries the full design rationale).** Launches every
enabled engine on the same `Problem` in its own thread, each with its own
`CancellationToken`. The first engine to reach a status this harness
actually trusts — "valid" is checked per engine against what it can promise
(`Optimal`/`Infeasible`/`Unbounded` for Simplex, `Optimal` only for
PDHG/IPM, since neither has infeasibility or unbounded detection yet) —
cancels every other entrant's token and becomes the winner. Every launched
thread is joined before `race_solve` returns, win or lose, so "clean
cancellation" (Build Map #10's own "Watch out") means what it says: no
dangling threads, no leaked state, by construction rather than by
convention.

**Built general on purpose**, per the ticket's own explicit ask ("reused
verbatim by the decomposition race #25"): the harness only needs an engine
to accept a `CancellationToken`, report a status this harness can check for
trustworthiness, and expose objective/primal/dual in a common shape.
Nothing here assumes the three contestants are simplex/PDHG/IPM specifically.

### 20.2 Two real bugs caught by testing this concurrently, not by inspection

1. **A genuine data race, caught by ThreadSanitizer.** `shared.remaining`
   and `shared.launched` were incremented in the main thread right before
   launching each worker, unlocked — race.cpp's very first version. If an
   already-launched worker finished and called `report()` (which decrements
   `shared.remaining` *under* the mutex) before the main thread reached the
   next iteration of its launch loop, the two writes raced. Building and
   running `test_race` under `-fsanitize=thread` caught this immediately
   (two independent data races reported, both at this exact spot); a plain
   build and even five repeated normal runs showed nothing, because the
   race window is narrow and machine-dependent. Fixed by computing both
   counts once, in full, before any thread exists — there is no longer
   anything for the main thread to touch after launch. Re-verified clean
   under TSan across three fresh runs after the fix.
2. **A dimension-mismatch crash on a real large instance, unrelated to
   concurrency.** Racing `maros-r7` (3136 rows, 9408 cols — over both
   Simplex's and IPM's dense-size caps, see below) with a tight time budget
   surfaced a genuine bug in PDHG (#8) itself: `best_x_scaled`/`best_y_scaled`
   (the best-iterate tracking added to fix ticket #8's own `adlittle`
   overshoot, HANDOFF §18) were only ever populated inside the "ran out of
   iterations" branch. A `TimeLimit` or `Cancelled` status that fires before
   iteration 0 reaches its first restart-check point (plausible on a big
   instance: setup costs like the power-iteration operator-norm estimate can
   themselves consume the whole budget) skipped that branch entirely, left
   both vectors empty, and the final `Scaling::unscale_primal` call threw a
   dimension-mismatch exception instead of returning a legitimate partial
   result. Fixed by decoupling "populate the reported point" from which
   status the loop exited with — it now always runs, and the status
   (`Optimal` / `IterationLimit` / whatever the loop already set) is decided
   separately. Regression test: `pdhg.a_time_limit_before_the_first_restart_
   check_still_reports_a_point`, using a `1e-9` second time limit to
   reproduce the exact zero-iterations edge case deterministically rather
   than depending on a slow machine or a large instance.

### 20.3 Verification

- `test_race` — 7 cases: a hand-verified optimum, an infeasible model
  (checking that Simplex's `Infeasible` counts as a *win*, not a loss, since
  PDHG/IPM can never produce one), an unbounded model likewise, a degenerate
  small instance, disabling every engine (reported cleanly, not hung),
  disabling two engines, and 20 repeated races back-to-back as practical
  evidence nothing leaks between calls.
- Clean under `-fsanitize=thread` (3 fresh runs, 0 warnings) after the fix
  in §20.2.1 above.
- `sovereign-cli solve <model> --engine race` on `afiro`: **Simplex wins**
  (small, well-conditioned — matches the Build Map's own "the degenerate
  small one may be won by simplex").
- **The "huge instance" half of the Build Map's Test step is mechanically
  demonstrated, not fully demonstrated end to end.** On `dfl001` (6071 rows)
  and `maros-r7` (3136 rows, 9408 cols), Simplex and IPM both correctly and
  immediately refuse via their own documented dense-size guards (`iterations:
  0`, a clear message) — the architectural claim ("the engine built for
  scale is the one left standing") holds exactly as designed. But PDHG,
  running single-threaded on the Host backend with no GPU on this machine,
  did not reach `Optimal` on either within a 45-120s budget — so the race
  correctly reports "no valid winner" rather than a wrong one, but does not
  positively demonstrate a PDHG *win* on a huge instance here. This is the
  same gap already named in §9/§17.2: the o9 Solutions precedent this
  project's whole strategic bet rests on (Bible Part II) is specifically
  about GPU wall-clock, and a CPU simulation of PDHG's iteration is not
  that. Revisit once GPU access is available again.
- `ctest`: 17/17. Sovereignty check clean.

---

## 21. Ticket #11 -- concurrent checkpoint crossover (Phase 3, M2)

### 21.1 What was built

```
include/sovereign/crossover.hpp   CrossoverOptions, CrossoverResult, Crossover,
                                   guess_basis_from_point, and a WarmStart addition
                                   to simplex.hpp
src/l1/crossover.cpp              the checkpoint harness
tests/test_crossover.cpp          6 cases
tests/test_simplex.cpp            +3 cases (WarmStart contract)
tools/check_crossover_vs_oracle.py  oracle comparison at simplex-grade (exact) tolerance
apps/sovereign_cli.cpp            `solve --engine crossover`
```

Two smaller, surgical additions this ticket needed elsewhere, both documented
inline where they live:

- **`Simplex::solve` gained an optional `WarmStart`** (simplex.hpp/.cpp): a
  basis guess (`m` variable indices) plus a point (used only to decide which
  bound a nonbasic guess starts at). `setup()` was split so the all-logical
  basis construction is now `reset_to_slack_basis()`, callable standalone;
  `apply_warm_start()` overwrites it from the guess and `run()` falls back to
  `reset_to_slack_basis()` if the guessed basis turns out singular. Nothing
  about a cold-start solve (`warm_start == nullptr`) changed -- every
  existing ticket #4 test still exercises the exact same code path it always
  did.
- **`PdhgOptions` gained an optional `checkpoint` callback** (pdhg.hpp/.cpp):
  invoked at the exact point the main loop already unscales a candidate
  point to check convergence (every `restart_check_period` iterations), with
  that same unscaled `(x, y)` and its original-space residual. The unscaling
  this needed was already being done for the final result, so it was
  factored into one `unscale_to_original()` helper used by both call sites
  rather than duplicated.

### 21.2 Why concurrent, not "PDHG then crossover"

The ticket's own "Watch out" is direct: "classical crossover is essentially
simplex again -- if you run it once, serially, after PDHG fully converges,
you delete the GPU win." So `Crossover::solve` never waits for PDHG to
finish. It launches PDHG on `backend` in its own thread with the checkpoint
callback installed; every checkpoint PDHG reaches -- whether or not PDHG
itself would call it converged -- spawns a fresh crossover attempt
(`guess_basis_from_point` + a warm-started `Simplex::solve`) in its own
thread, running concurrently with PDHG's continuing iteration and every
earlier attempt. The first attempt to reach a status Simplex is actually
entitled to assert (`Optimal`, `Infeasible`, or `Unbounded` -- Phase I/II are
exact regardless of how good the guess was, so an infeasibility proof from a
bad guess is just as trustworthy as one from a good one) wins: it cancels
PDHG and every other in-flight attempt and the harness returns as soon as
everything is joined. This is deliberately the same shape as ticket #10's
`race_solve` (own `CancellationToken` per entrant, "first to a *valid*
answer" wins, every thread joined before returning) -- ticket #11 is
racing an unbounded, dynamically-growing set of entrants instead of three
fixed ones, which is the one structural difference (`std::deque
<CancellationToken>` for stable addresses across growth, since
`CancellationToken` wraps a `std::atomic<bool>` and is neither copyable nor
movable, so it cannot live in a `std::vector` that might reallocate).

**Graceful degradation when nothing overlaps.** If no checkpoint ever fires
before PDHG itself stops (tested explicitly: `restart_check_period` set
larger than `max_iterations`), or every checkpoint attempt loses without
producing a valid answer, the harness makes exactly one more attempt
directly on PDHG's own final returned point before declaring failure --
ordinary, non-concurrent crossover, exactly the degenerate case concurrency
had nothing left to overlap with. `CrossoverResult::winning_checkpoint_
iteration == -1` marks this path so a caller can tell it apart from a real
concurrent win.

### 21.3 Basis construction (`guess_basis_from_point`)

For each of the `n + m` variables in the internal `[A -I]` indexing (ticket
#4's own form -- structural columns then row logicals), compute its margin
to the nearer of its own two bounds (`min(x - lo, hi - x)`; a variable with
no finite bound at all gets `+infinity`, because it MUST be basic for its
guessed value to be representable -- a nonbasic variable is pinned to a
bound, and a nonbasic free variable is pinned to 0, silently discarding
whatever value it actually had). The `m` variables with the largest margins
(furthest from their own bounds -- i.e., the ones the interior point treats
as truly interior) are guessed basic; everything else is nonbasic, snapped
to whichever bound its guessed value sits closer to.

This is a GUESS, not a proof, and the code never claims otherwise: `Simplex`
verifies and corrects it exactly as it would a cold start (Phase I restores
feasibility from whatever basis it is handed, Phase II then optimizes), and
falls back to the slack basis if the guess is singular. A bad guess costs
pivots; it cannot cost correctness -- this is what makes running dozens of
concurrent guesses safe rather than dozens of chances to return a wrong
answer.

### 21.4 Verification

- `test_simplex` -- 3 new cases: a correct warm-start basis reaches the same
  hand-checked optimum as a cold start; a structurally invalid guess (basis
  index out of range) is rejected by `apply_warm_start()` and falls back
  cleanly; a warm start on an infeasible model still proves infeasibility
  (the guess must never change what the solve is ALLOWED to conclude).
- `test_crossover` -- 6 cases: `guess_basis_from_point` picks the right
  basis on a hand-verified vertex; a feasible model reaches the exact
  optimum (`1e-9`, not PDHG's `1e-4`) via a real concurrent checkpoint win
  (`checkpoint_attempts > 0`); a degenerate small instance; an infeasible
  model caught by a crossover attempt's own Simplex Phase I even though PDHG
  itself has no infeasibility detection at all; the no-checkpoint-ever-fires
  degenerate case falls back correctly (`winning_checkpoint_iteration ==
  -1`); 10 repeated solves back to back as practical evidence of clean join
  discipline, same precedent as ticket #10's own test.
- **Clean under ThreadSanitizer**, 3 fresh runs, 0 warnings -- ticket #10's
  own race harness caught a real data race this way (HANDOFF S20.2.1) and
  this harness has the same shape (shared winner state, dynamically spawned
  threads reporting into it), so the same check was run here rather than
  trusted by inspection.
- `check_crossover_vs_oracle.py` at **exact tolerance** (`1e-6`, the same
  tight tolerance `check_simplex_vs_oracle.py` (#4) uses, deliberately not
  PDHG's looser `5e-3`/`1e-3`): smoke set + 7 Netlib instances (`adlittle`,
  `afiro`, `fit1d`, `fit2d`, `kb2`, `sc50a`, `sc50b`) all agree with the
  oracle. `fit2d` (25 rows, 10500 columns -- the same instance
  `check_pdhg_vs_oracle.py` already flags as PDHG's documented slow tail,
  HANDOFF S18.2) needs a wider timeout than the default 30s to let a
  checkpoint actually reach convergence; given 180s it converges and
  matches the oracle exactly.
- **Wall-clock, mechanically, on this machine's Host backend (not the GPU
  claim -- see S9/S17.2/S20.3 for why no number here can be that claim
  yet).** `fit1d`: crossover 207ms vs. plain simplex 515ms. `fit2d`:
  crossover 12.7s vs. plain simplex 50.5s -- roughly 4x, and specifically on
  the two instances wide/hard enough that simplex actually does a lot of
  work, which is the shape of evidence the ticket's own "beat CPU wall-clock"
  condition wants. On the small, easy instances (`afiro`, `sc50a`, `sc50b`,
  `kb2`) plain simplex is faster in absolute terms -- expected and
  unsurprising (concurrency overhead is not free, and simplex alone was
  already solving those in single-digit milliseconds), and not the case the
  ticket's own M2 pass condition is asking about.
- `ctest`: 18/18. Sovereignty check clean (`check_crossover_vs_oracle.py`
  added to the existing `highs` exception's `paths`, same pattern as the
  other oracle-comparison scripts).

### 21.5 M2 status and what is still open

**M2's own pass condition** ("an exact vertex is produced from a GPU
interior solution, and total time still beats CPU") is green on everything
checkable without a GPU: exact vertices, concurrent checkpoint wins,
clean cancellation under TSan, and a real (if Host-backend-only) wall-clock
win on the two hardest instances tried. What is NOT yet demonstrated is the
ticket's actual headline claim -- winning against CPU on a genuinely large
GPU-scale instance, which needs a real device (same carried gap as M1's own
close in S9/S17.2/S20.3).

**Ticket #12 (spiral-axis vertex jump)** is the explicit stretch goal layered
on top of this -- Build Map: "this is a moonshot layered on top... if it
doesn't work, it's a clean roadmap line," and #11 (this ticket) is the
committed fallback it must not be blocked by. Not started.

**Not addressed here, tracked forward:** the checkpoint callback currently
fires unconditionally at PDHG's existing restart-check cadence
(`restart_check_period`), not at a schedule tuned for crossover's own cost
(spawning `O(iterations / restart_check_period)` Simplex solves on a large
instance is not free); a real GPU run is what will make that tuning
question concrete rather than theoretical.

---

## 22. Ticket #12 -- spiral-axis vertex jump (Phase 3, M2 stretch, SEED/FRONTIER)

### 22.1 What was built

```
include/sovereign/spiral_jump.hpp   estimate_spiral_jump, SpiralJumpResult
src/l1/spiral_jump.cpp              the order-2 minimal polynomial extrapolation
tests/test_spiral_jump.cpp          7 cases, against a SYNTHETIC spiral
include/sovereign/crossover.hpp     +enable_spiral_jump, +spiral_jump_fit_residual_cutoff,
                                     +spiral_jump_attempts, +won_by_spiral_jump
src/l1/crossover.cpp                the extra race entrant
tests/test_crossover.cpp            +2 cases (additive, never required)
apps/sovereign_cli.cpp              `--no-spiral-jump`, spiral-jump fields in
                                     `solve --engine crossover` output
```

### 22.2 The math, and why it's trustworthy on its own terms

`estimate_spiral_jump` is Sidi's Minimal Polynomial Extrapolation at order 2,
self-derived in spiral_jump.hpp's own header comment from the assumption
PDLP's literature documents: near the optimal face, PDHG's error
`e_k = z_k - z*` often evolves as `e_{k+1} = M e_k` for a fixed operator `M`
whose dominant mode is a COMPLEX-CONJUGATE eigenvalue pair -- a rotating,
contracting spiral, not a straight-line approach (the reason ticket #8's
restarts help at all: averaging over a rotation cancels it). Given 4
consecutive iterates, the method fits the order-2 recurrence the resulting
difference vectors must satisfy and solves directly for the point that
recurrence converges to -- a weighted average of the first three iterates,
with negative weights allowed (extrapolation, not interpolation), no further
iteration run.

**Verified against a synthetic spiral first, deliberately separated from
real PDHG behavior.** `test_spiral_jump` constructs an exact
rotate-and-contract sequence with a known fixed point (`estimate_spiral_jump`
recovers it to `1e-6` for both a scalar and a joint primal+dual case) and
confirms the method correctly DECLINES rather than guesses when the
assumption's preconditions fail: a window with no rotational information
(already converged, every difference zero) or only a single real eigenvalue
(differences collinear, the 2x2 fit is exactly singular) both correctly
return `available = false`. This is what makes it safe to wire into a real
solve without knowing in advance whether any given instance's dynamics will
cooperate -- the math itself is correct and self-certifying about when it
applies, independent of the empirical question below.

### 22.3 Integration: a strictly additive race entrant, never a replacement

Per the ticket's own "Watch out" ("do NOT let it block M2 -- #11 is the
committed path"), the spiral jump changes nothing about ticket #11's own
correctness or of its own accord: `Crossover` maintains a 4-point window of
raw checkpoint iterates (touched only from the single-threaded PDHG
checkpoint callback, no lock needed for the window itself) and, whenever the
window is full and `estimate_spiral_jump` reports `available` with a
`fit_residual` below `spiral_jump_fit_residual_cutoff` (default `0.3`,
explicitly a generous "worth a free extra entrant" bar, not a tuned
constant), launches ONE MORE crossover attempt from the extrapolated point
-- racing alongside, never instead of, the checkpoint's own literal-point
attempt. `enable_spiral_jump = false` removes every such entrant and the
solve is provably unaffected in what it can conclude (`test_crossover`:
`disabling_the_spiral_jump_still_solves_correctly`), because every attempt,
spiral-jump-seeded or not, is just another `WarmStart` guess into the same
Simplex machinery ticket #11 already made safe against a bad guess.

Clean under ThreadSanitizer (3 fresh runs, 0 warnings) after adding the new
shared counters and the `launch_attempt` helper -- re-run for the same
reason ticket #11's own concurrency was re-checked rather than trusted by
inspection.

### 22.4 The empirical question, answered honestly

The ticket's own Done-when is explicit that a negative result is an
acceptable outcome here, carried as a documented roadmap line rather than
hidden. Tested against 9 real Netlib instances (`afiro`, `adlittle`,
`sc50a`, `sc50b`, `kb2`, `blend`, `brandy`, `fit1d`, plus repeated runs of
`blend`/`share2b`/`afiro` for race-timing variance) via `--engine crossover
--json`:

- The mechanism fires SELECTIVELY, as designed -- 0-1 spiral-jump attempts
  on well-conditioned small instances (`afiro`, `brandy`), 13-15 on
  instances with richer dynamics (`blend`, `sc50b`) -- confirming
  `fit_residual` is doing real filtering, not passing everything or
  nothing.
- **It has not won a single race in this test suite.** Every solve above
  reached the exact optimum correctly (never a regression -- ticket #11's
  own literal-checkpoint attempts, or the final fallback, always closed it
  out), but `won_by_spiral_jump` was `false` on every run tried.
- The likely reason, based on watching `checkpoint_attempts` vs.
  `winning_checkpoint_iteration`: dozens of attempts (both kinds) launch
  over a single PDHG run on this 12-core machine, and the race is won by
  whichever warm start needs the FEWEST Simplex pivots -- which tends to be
  a LATE, already near-converged literal checkpoint (a very easy Simplex
  finish), not an early spiral-jump extrapolation whose target itself is
  less accurate this early in the run. This is a genuine, previously-unclear
  empirical finding, not a defect in the extrapolation math (S22.2's
  synthetic tests confirm the math is right when its own preconditions hold)
  -- it may be an artifact of CPU-oversubscribed concurrent racing
  specifically, which could read differently on a real GPU where PDHG's own
  iteration cost (not Simplex pivot count) dominates the wall-clock and an
  early jump has more time to matter. Untested, because that needs the same
  GPU this project has lacked since S9.

**Per the Build Map's own accepted outcome for this ticket:** carried as
attempted, correctly implemented, empirically not a winner on the CPU-only
instances tested here, and left ENABLED by default because it never costs
correctness and never blocks M2 (already closed by #11) -- a clean roadmap
line, re-evaluate once a GPU is available (S9/S17.2/S20.3's same standing
gap) or a larger instance class surfaces a case where it does win.

### 22.5 Verification

- `test_spiral_jump` -- 7 cases (S22.2).
- `test_crossover` -- +2 cases: disabling the spiral jump still solves
  correctly; an unreachable cutoff (`-1.0`) disables every spiral-jump
  attempt, confirming the cutoff is actually wired to
  `estimate_spiral_jump`'s own output rather than ignored.
- Clean under ThreadSanitizer, 3 fresh runs.
- `ctest`: 19/19. Sovereignty check clean.

---

## 23. Ticket #13 -- lightweight dual-preserving presolve (Phase 4, M3)

### 23.1 What was built

```
include/sovereign/presolve.hpp   PresolveMove variant, PresolveResult, PostsolveResult,
                                  presolve(), postsolve()
src/l1/presolve.cpp              the reduction pipeline + trail replay
tests/test_presolve.cpp          10 cases, each checked via #7's own
                                  complementary_slackness_violation
apps/sovereign_cli.cpp           `presolve <model>` -- reduces, solves the
                                  reduced model, postsolves, and self-verifies
tools/check_presolve_vs_oracle.py   reduction size vs. HiGHS's own presolve log
```

### 23.2 Scope -- four move types, each with a PROVEN postsolve rule

The ticket's own "Watch out" ("dual-preservation is not optional -- break it
and you break Benders and conflict learning downstream") was taken
literally: every reduction implemented here removes a row or column via a
move whose postsolve dual-reconstruction rule was derived by hand (in
presolve.hpp's own file comment) from the original problem's KKT conditions,
not just asserted plausible.

- **FixedColumnMove** -- a column pinned to one value (originally `lo==hi`,
  or tightened to a point by a SingletonRowMove) is substituted out, folding
  its contribution into every row it still touched.
- **EmptyColumnMove** -- a column touching no active row is resolved
  directly from its own bounds and the objective's sign; it never had a
  row's dual to account for, so its postsolve reduced cost is just its
  objective coefficient.
- **RedundantRowMove** -- a row whose activity range (computed from bounds
  it did not itself produce) already sits inside its own bounds. Its dual
  is UNCONDITIONALLY 0 in postsolve: by construction it is never binding
  across the entire feasible region the check considered, and a constraint
  that never binds has zero shadow price in ANY optimal dual solution --
  no case split needed.
- **SingletonRowMove** -- a row with exactly one active nonzero coefficient
  tightens that column's bound and is removed. Its dual is NOT
  unconditionally 0 -- see S23.3.

**Deliberately out of scope, named as follow-up, not half-attempted (same
discipline ticket #4 applied to sparse LU):** general non-removing
activity-based bound tightening, and multi-variable "forcing row"
elimination (pinning several columns from one row's own extreme
simultaneously). Both need the SAME dual-redistribution reasoning
generalized across a longer dependency chain; getting it wrong silently is
exactly the failure mode the ticket warns about. This is also the honest
reason the reduction SIZE falls well short of "~90% of a commercial
presolve" (S23.5) -- most of a commercial presolve's power is in exactly
these two families.

### 23.3 The singleton-row dual formula, and why it needs one

Worked out by hand before writing any code (presolve.cpp's own header
comment carries the full derivation): the REDUCED problem's own solve
reports `rc_reduced[j] = c_j - sum_{rows still present} a_kj y_k` for the
tightened column `j` -- row `i` (the singleton row) is simply absent from
that sum. The TRUE original reduced cost is `rc_true[j] = rc_reduced[j] -
a_ij * y_i`.

- If `x_j` ends up at a bound STRICTLY TIGHTER than its pre-row-`i` bound
  (row `i` is the actual reason `x_j` sits there): relative to the
  ORIGINAL, wider bound, `x_j` is interior, so KKT requires `rc_true[j] = 0`
  exactly -- which pins `y_i = rc_reduced[j] / a_ij`.
- If `x_j` does not sit at the bound row `i` produced (interior even to the
  tightened bound, or the tightened bound happens to coincide with the
  original one): row `i` contributed nothing distinguishable, and `y_i = 0`
  is safe.

An earlier draft of this reasoning (worked through by hand against a small
`x + y <= 10, y in [2,3]` example before any code was written) initially
assumed `y_i = 0` was ALWAYS safe once a row becomes redundant on a later
pass -- exactly the case a hand-derived counterexample caught: if the row's
own tightening is what CAUSED its later redundancy, the column can end up
interior to its ORIGINAL bounds while the reduced solve reports a nonzero
reduced cost for it, which is only consistent if that reduced cost is
attributed to the row rather than reported as-is. This is why
SingletonRowMove and RedundantRowMove are two DIFFERENT move types with two
different dual rules, not one generic "row removed -> y=0": a row is only
ever safe at `y=0` unconditionally when its own removal did not itself
create the bound the final point might be sitting at.

### 23.4 Postsolve architecture: a trail, replayed in reverse (LIFO)

Every move is appended to `trail` in the order APPLIED. `postsolve` walks
`trail` in REVERSE. The invariant that makes each move's local dual rule
sufficient: by the time any move's undo runs, every row/column it
references was either never removed, or removed by a move LATER in forward
order -- which, in reverse (LIFO), has ALREADY been undone. Concretely: a
`FixedColumnMove.rows` list only ever contains rows that were still ACTIVE
at the moment that column was fixed, so any row in it was either never
removed (known from the start) or removed by a STRICTLY LATER move (already
restored by the time this one's reverse-undo runs). This is Andersen &
Andersen's own standard presolve/postsolve design, re-derived here rather
than copied, and it is what makes the two-step chain in
`chained_reductions_across_rounds_still_check_out`'s test (a SingletonRowMove
tightens a column to a point, a LATER FixedColumnMove substitutes it out)
compose correctly without either move needing to know about the other.

### 23.5 Verification

- `test_presolve` -- 10 hand-verified cases, one per move type plus a
  two-round chained case and two ordinary (non-reducible) models. Every
  single case runs through the SAME check: presolve -> solve the reduced
  model -> postsolve -> confirm the objective matches a direct
  (unpresolved) solve exactly AND `DualSolution::complementary_slackness_
  violation` (ticket #7's own checker, reused rather than reinvented) is
  near zero against the ORIGINAL problem. A move that is individually
  "tested" but produces a subtly wrong dual cannot pass this.
- **Against the real corpus, exact-tolerance:** every Netlib LP instance up
  to 500 rows (51 instances) presolves, solves, postsolves, and passes both
  the objective-match and complementary-slackness checks --
  `check_presolve_vs_oracle.py`: **51/51 verified**. Two instances
  (`d6cube`, `fit2d`) time out at the script's default 30s budget on this
  machine's sequential CPU presolve (see S23.6) and are skipped, not
  counted as failures.
- **Reduction size vs. HiGHS's own presolve** (parsed from its `Presolve :
  Reductions: rows R(-r); columns C(-c); elements E(-e)` log line, summed
  across every comparable instance): **rows 30.6%, columns 12.9%, elements
  10.6%** of what HiGHS's presolve removes on the same instances. This is
  the ticket's own "~90%" pass condition, reported honestly rather than
  rounded up -- the gap is exactly the two families named out of scope in
  S23.2 (general bound tightening and forcing rows are where most of a
  commercial presolve's reduction actually comes from). The ~30%/13%/11%
  achieved here is real, verified-correct reduction from a deliberately
  narrower move set, not a partial or unsound version of the full one.
- `ctest`: 20/20. Sovereignty check clean.

### 23.6 What is NOT done here, tracked forward

- **General activity-based bound tightening and multi-variable forcing-row
  elimination** (S23.2) -- the largest concrete gap against the "~90%"
  target. Needs the singleton-row dual-redistribution idea (S23.3)
  generalized to a bound that may have been tightened by SEVERAL rows in
  sequence, not just one -- a real extension, not a rewrite, but real work.
- **Sequential CPU only**, matching ticket #4's own precedent: this is the
  correctness baseline, not the GPU-native/tight-parallel version Bible
  S4.4 ultimately wants. `d6cube` and `fit2d` (the two timeouts above) are
  exactly the instances large/dense enough that this matters in practice.
- **No duplicate row/column detection, no coefficient strengthening, no
  parallel-row/column merging** -- all standard commercial-presolve
  reductions, all untouched here. Each would need its own dual-postsolve
  derivation with the same care as S23.3; none was attempted rather than
  guessed at.
