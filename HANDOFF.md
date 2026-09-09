# Handoff — Sovereign Solver, state after Ticket #1

**Last commit:** `b743a65` — *Ticket #1: sovereignty dependency ledger and CI enforcement*
**Gate:** M0 (in progress — #1 done, #2 and #3 remain before M0 closes)
**Model note:** #1 was built under Opus 5; this handoff written under Sonnet 5.

This document covers **only what was written in this repo**. The Bible and Build
Map explain the *why* and the *what-next* and are uploaded alongside — this fills
the gap between them and the actual code on disk.

---

## 1. Where the project is

Build Map Phase 0, ticket #1 of 55 is complete. The repo was a bare directory
with three `.md` files; it is now a git repository (`git init` was run — commit
author `Ashish Phutela <ashishphutela@gmail.com>`) with a CMake skeleton whose
**only** job right now is to give the sovereignty check something to inspect.

**There is no solver code yet.** No parser, no simplex, no L0 containers. That
starts at ticket #3. `CMakeLists.txt` has commented-out `add_subdirectory(src)`
and `add_subdirectory(tests)` lines marking where that plugs in.

### Files added in this ticket

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
3. **Vendored directories** — any subdir of `third_party/`, `extern/`, `vendor/`,
   `deps/`, `subprojects/`, … whose *name* matches a forbidden entry. Catches a
   raw source-tree copy that no manifest mentions.
4. **Resolved dependency graph** — with `--cmake-build-dir <dir>`, reads
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
undocumented suppression is indistinguishable from a breach. There are currently
zero exceptions (the two bugs above were fixed properly rather than suppressed).

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

`ctest --test-dir build` runs everything. Individual suites:
`python3 tests/test_oracle.py`, `python3 tests/test_corpus.py`,
`python3 tools/oracle/check_reference.py`.

---

## 11. Immediate next steps

M0 still needs **#3** before Phase 1's simplex oracle (#4).

- **#3** — L0 numerical substrate: CSR + CSC containers (VBCSR/BSR stubs),
  mixed-precision type policy (fp64 real, fp32/fp16 flagged unused), and the
  CUDA↔ROCm backend abstraction (CUDA impl + ROCm stub). When `src/` gets real
  content, uncomment `add_subdirectory(src)` in `CMakeLists.txt`; the
  sovereignty check picks it up automatically (globs already cover `**/*.cpp`,
  `**/*.cu`, …). **Blocker on this machine:** no CUDA toolkit, no ROCm, no
  BLAS/LAPACK — the abstraction layer and CSR containers can be built and
  unit-tested CPU-only, but any GPU kernel or BLAS call needs the toolchain
  installed first (see §9).
- When #3 lands, add any new permitted deps (OpenBLAS, GoogleTest, cuSPARSE…)
  to `sovereignty.toml` **and** `DEPENDENCY_LEDGER.md` in the same commit, and
  extend `test_sovereignty_check.py`'s `RealRepositoryTests` list.

---

## 12. What you can test right now

Everything below is green on this machine as of commit `3178a54`.

```sh
# one shot — configure, run the in-ALL sovereignty target, run every suite
cd /home/arch_btw/Documents/SIH
rm -rf build && cmake -S . -B build -G Ninja
cmake --build build          # fails the build on any sovereignty violation
ctest --test-dir build       # 4 tests: sovereignty self-test, oracle, corpus, e2e probe

# the sovereignty check directly
python3 tools/sovereignty_check.py --cmake-build-dir build --check-ledger -v
python3 tools/sovereignty_check.py --explain kahypar          # any dependency
python3 tools/test_sovereignty_check.py                       # 45 tests

# the oracle end to end
tools/oracle/run_oracle.py benchmarks/data/netlib_lp/afiro.mps --with-solution
tools/oracle/check_reference.py                               # 10 instances vs published optima
tools/oracle/check_reference.py --all                         # all 90 (slower)

# the corpus
python3 benchmarks/fetch_corpus.py --list
python3 benchmarks/fetch_corpus.py --verify                   # 90/90 sha256 intact
python3 tests/test_oracle.py                                  # 14 tests
python3 tests/test_corpus.py                                  # 20 tests
```

Expected: sovereignty `OK`, `ctest` 4/4, `check_reference.py` `PASS: 10/10`,
`--verify` `90/90 instances intact`, all Python suites `OK`.

To rebuild the oracle from scratch (e.g. on another machine):
`tools/oracle/install_highs.sh` then `python3 benchmarks/fetch_corpus.py`.

**Not yet testable** (needs ticket #3 + a GPU/BLAS toolchain): anything that
solves an LP with our *own* code. The oracle is the reference; we haven't
written the thing it's a reference *for* yet.
