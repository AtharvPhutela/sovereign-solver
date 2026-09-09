# Sovereignty Dependency Ledger

**Build Map ticket #1 · Gate M0 · Bible §1.2**

This is the living record of what this project is allowed to depend on, and why.
It is the written half of ticket #1; the machine-enforced half is
[`sovereignty.toml`](sovereignty.toml), checked in CI by
[`tools/sovereignty_check.py`](tools/sovereignty_check.py).

The two halves cannot drift: `sovereignty_check.py --check-ledger` fails the
build if anything classified in the policy is missing from this document.

---

## 1. The constraint

> *"It shall not be built upon any existing open source solver library but shall
> be built from scratch from mathematical foundation."*
> — Problem statement, Description

This single sentence governs the entire design (Bible §1.2). It does **not** mean
re-deriving BLAS. It means **we own the optimization algorithms**, and we
assemble them on top of standard numerical bricks.

### The test

Every dependency is judged by one question:

> **If this library vanished tomorrow, is what remains still our optimization
> engine?**

- **Yes** — the thing that vanished was a numerical primitive. *Permitted.*
- **No** — the thing that vanished was doing our job for us. *Forbidden.*

A useful sharpening: does the library know what an *objective*, a *bound*, or a
*basis* is? A sparse factorization package does not — you hand it a matrix and it
factors it. A simplex implementation does. That distinction is why SuiteSparse
and MUMPS are permitted while Clp is not, even though all three are "sparse
linear algebra" by casual description.

### Why the check is mechanical

Sovereignty is the project's disqualifying constraint: a single forbidden library
in the build tree invalidates the entire submission, however good the numbers
are. That is too important to rest on reviewer vigilance, and too easy to breach
by accident — a transitive dependency, a helpful `find_package`, a vendored
directory copied in "just to compare". So it is enforced by a CI job that fails
the build, not by a convention.

---

## 2. The three tiers

A binary permitted/forbidden split loses two cases this project actually has, so
the ledger uses three tiers.

| Tier | Meaning | Enforcement |
|---|---|---|
| **PERMITTED** | A numerical primitive, build tool, or test framework. Carries no optimization logic. | Free use. |
| **RESTRICTED** | Dual-use. Legitimate in one role, a sovereignty breach in another. | Permitted **only** inside declared path scopes; a reference outside them fails the build. |
| **FORBIDDEN** | An optimization solver, or an algorithm that is itself one of our differentiators. | Never linked, forked, vendored, or imported. Build fails on sight. |

The restricted tier exists because two real cases have no honest binary answer:

1. **The benchmark oracle (#2).** We are required to validate against an
   established solver. That solver is forbidden as a *library* and required as a
   *tool*. The resolution is the Oracle rule: it is reachable only as a black-box
   CLI over MPS/LP files on disk — never linked, never imported, never on our
   link line. The tier makes that boundary mechanical instead of aspirational.

2. **METIS.** As a nested-dissection fill-reducing ordering inside sparse
   factorization it is unambiguously linear algebra. As block-structure
   detection it would be doing ticket #14's job — a named differentiator
   (Bet 6). Same library, opposite verdicts, distinguishable only by *where it is
   used*. So it is scoped to `src/l0/**` and the check enforces that.

---

## 3. FORBIDDEN

Never linked, forked, vendored, or imported. Studying published algorithms and
papers is not only allowed but expected — Bible §4.2A.4 explicitly marks GCG,
mpi-sppy and PIPS-IPM++ as *studied, never linked*. Reading the paper is
research; linking the binary is disqualification.

### 3.1 LP / MILP solver libraries

| Library | Category | Verdict reason |
|---|---|---|
| **COIN-OR CBC** | MILP solver | A branch-and-cut MILP solver — it *is* the thing we are building at L4 (#38). |
| **COIN-OR Clp** | LP solver | Simplex LP solver; substitutes for the L1 core and the from-scratch simplex oracle (#4). CoinUtils is aliased to it: it is Clp's model substrate, not generic linear algebra. |
| **HiGHS** | LP + MILP solver | Dual simplex, IPM and branch-and-cut. The open-source bar the PS names — an oracle, never a component. |
| **SCIP** | MILP solver | Constraint-integer framework carrying the conflict analysis and symmetry handling we must build ourselves (#41–#45). |
| **SoPlex** | LP solver | SCIP's simplex engine. |
| **GLPK** | LP + MILP solver | Simplex, IPM and branch-and-cut — a complete substitute for our spine. |
| **lp_solve** | LP + MILP solver | Simplex plus branch-and-bound. |
| **Google OR-Tools** | Solver suite | Bundles GLOP, PDLP and CP-SAT. |
| **GLOP** | LP solver | OR-Tools' simplex; listed separately because it can be vendored alone. |
| **PDLP** | LP solver | The reference first-order LP solver. We reimplement its published algorithm at #8; linking it would hollow out the project's central bet. |
| **NVIDIA cuOpt** | GPU solver | Our closest architectural competitor (Bible Part VIII). Using it makes the GPU-native differentiation claim vacuous. |

### 3.2 Commercial solvers

| Library | Verdict reason |
|---|---|
| **Gurobi** | One of the three foreign solvers the PS exists to displace. Oracle-only, if licensed. |
| **IBM ILOG CPLEX** | Named in the PS title as what we are a sovereign alternative to. |
| **FICO Xpress** | Named in the PS. Its structural-determinism guarantee is a target we match ourselves at #46, not one we borrow. |
| **MOSEK** | Commercial conic/QP solver. |

### 3.3 QP / conic / NLP solvers

| Library | Verdict reason |
|---|---|
| **OSQP** | The ADMM operator-splitting QP solver. Bible §4.2 specifies an "OSQP-*style*" variant — the published algorithm, not the library. |
| **Ipopt** | Interior-point NLP solver; substitutes for our IPM (#9) and the future NLP extension. |
| **Bonmin** | MINLP branch-and-bound. |
| **Couenne** | Spatial branch-and-bound for nonconvex MINLP — precisely the Bible Part XI.B roadmap we intend to own. |
| **qpOASES** | Active-set QP solver. |
| **Clarabel**, **ECOS**, **SCS** | Interior-point and first-order conic solvers; SCS in particular is the same algorithmic family as our PDHG engine. |

### 3.4 Graph automorphism — the nauty trap

Bible §1.2 calls these out specifically, and they are the easiest gray zone to
get wrong: they look like graph utilities, but symmetry detection is
differentiator **Bet 7**.

| Library | Verdict reason |
|---|---|
| **nauty** | Individualization-Refinement automorphism engine. Ticket #41 requires IR be reimplemented from the mathematical foundation. |
| **Traces** | Ships inside the nauty distribution. |
| **bliss** | The automorphism tool SCIP links for symmetry breaking — we differentiate against exactly that arrangement (Bible Part VIII). |
| **saucy** | Sparse-symmetry automorphism tool. |
| **PermLib** | Schreier-Sims / BSGS permutation group library; ticket #42 builds the stabilizer chain itself. |

### 3.5 Decomposition frameworks

Bible §4.2A.4 lists these as *studied, never linked*.

| Library | Verdict reason |
|---|---|
| **GCG** | Generic Column Generation — automatic block detection plus Dantzig-Wolfe. We read its hypergraph detection design and reimplement it GPU-parallel at #14. |
| **COIN-OR DIP** | Decomposition in Integer Programming; substitutes for L1.5 (#25). |
| **mpi-sppy** | Progressive Hedging / scenario decomposition. Studied for its adaptive ρ initialization (#32). |
| **PIPS-IPM++** | Schur-complement structured IPM. Studied as validation for the cuDSS approach. |
| **PySP**, **DSP** | Stochastic / dual decomposition layers. |

### 3.6 Hypergraph partitioners

| Library | Verdict reason |
|---|---|
| **KaHyPar** / Mt-KaHyPar | Multi-level hypergraph partitioning — precisely the algorithm ticket #14 must implement GPU-parallel. Linking it forfeits Bet 6. |
| **PaToH** | Hypergraph partitioning toolkit. |
| **hMETIS** | Hypergraph partitioner. Distinct from METIS (graph, fill-reducing, restricted tier): hypergraph partitioning has exactly one use here, and it is ours to own. |

---

## 4. RESTRICTED

Permitted **only** inside the listed scopes. A reference outside them fails the
build — these are the boundaries that cannot be enforced by intent alone.

| Dependency | Allowed scope | The line being drawn |
|---|---|---|
| **METIS / ParMETIS** | `src/l0/**`, `cmake/**` | Fill-reducing ordering for sparse factorization is linear algebra. Block-structure detection is ticket #14. Same library, opposite verdicts. |
| **SciPy** | `tools/**`, `tests/**`, `benchmarks/**` | Ships `scipy.optimize.linprog` (HiGHS-backed) alongside `scipy.sparse`. Fine for harness matrix I/O; never in the core, and never an answer source except as a declared oracle. |
| **NumPy** | `tools/**`, `tests/**`, `benchmarks/**`, `python/**` | Array container only. Scoped so it never becomes a load-bearing runtime dependency of the C++ core. |
| **Pyomo** | `python/**`, `tests/**`, `benchmarks/**` | A modeling frontend we must be a drop-in backend *for* (#51), not a solver. Caveat: Pyomo dispatches to whatever solvers are installed — invisible at build time, so the Oracle rule governs at runtime. |
| **PuLP** | `python/**`, `tests/**`, `benchmarks/**` | Same role (#51). **Specific trap: the PuLP wheel vendors a CBC executable.** Its presence in the environment is tolerated; invoking that CBC from anything but a declared oracle path violates the Oracle rule. |
| **CVXPY** | `python/**`, `tests/**`, `benchmarks/**` | Modeling frontend (#51). Pulls ECOS/SCS/Clarabel transitively — tolerated in the harness environment, never linked into the core. |

The benchmark oracle (ticket #2) is handled by §5.5 below rather than a
restricted-tier row: the Oracle rule is enforced concretely (an excluded build
path plus scoped exceptions), not by a placeholder token.

---

## 5. PERMITTED

Numerical primitives, build tooling, test infrastructure. Free use.

### 5.1 Dense and sparse linear algebra

| Dependency | Note |
|---|---|
| **OpenBLAS**, **BLIS**, **reference LAPACK** | BLAS/LAPACK. Remove them and our algorithms remain, only slower. |
| **Intel oneMKL** | Permitted as linear algebra, but vendor-locked to x86 and so a *hardware*-sovereignty weakness. Prefer OpenBLAS/BLIS by default. |
| **Eigen** | Header-only dense/sparse templates. No optimization logic. |
| **SuiteSparse** (CHOLMOD, UMFPACK, KLU, AMD, COLAMD, SPQR) | Sparse direct factorization and fill-reducing orderings. It factors a matrix we hand it and has no notion of an objective, a bound, or a basis — recorded explicitly so nobody later assumes it carries optimization logic. |
| **MUMPS** | Multifrontal sparse direct solver. Same verdict. Named explicitly because it is commonly bundled *with* Ipopt: take the factorization, never the solver wrapped around it. |
| **AMD** (approximate minimum degree) | Fill-reducing ordering for the symbolic factorization in #9. |

### 5.2 CUDA

**cuBLAS**, **cuSPARSE**, **cuSOLVER**, **cuDSS**, **Thrust**, **CUB**, the CUDA
runtime and **NVTX**. Bible §1.2 names these permitted explicitly.

cuSPARSE deserves a note: it provides the SpMV that is the entire inner loop of
PDHG (#8). It multiplies a matrix by a vector — the primal-dual iteration around
it, the restarts, the preconditioning and the convergence theory are ours.

cuDSS likewise performs the LDLᵀ factorization for the pivoting-free IPM (#9) and
the Dantzig-Wolfe restricted master (#29). Bible §6.5 flags its pivot-free
stability as a Tier-1 *numerical* risk — that is a correctness concern, not a
sovereignty one, and is tracked separately.

### 5.3 ROCm

**rocBLAS**, **rocSPARSE**, **rocSOLVER**, **HIP** (hipBLAS/hipSPARSE/hipSOLVER).
Required for the CUDA↔ROCm backend abstraction in ticket #3, which is what buys
hardware sovereignty. Bible §4.1 is emphatic that this abstraction goes in from
the start — retrofitting it later is a rewrite.

### 5.4 Build, bindings, parallelism, and test infrastructure

| Dependency | Note |
|---|---|
| **CMake**, **Ninja**, **CTest** | Build system. |
| **GoogleTest**, **Catch2** | Unit test frameworks. |
| **pybind11**, **nanobind** | Binding glue for #51. |
| **fmt**, **spdlog**, **CLI11** | Formatting, logging, CLI parsing (#50). |
| **OpenMP** | Host threading substrate. The work-stealing scheduler of #46 is ours; this only provides threads. |
| **oneTBB** | Host threading primitives. **Caveat:** use it for threads, not for its task scheduler — the Chase-Lev deques and deterministic barriers of #46 must be ours, or the determinism guarantee becomes untestable. |
| **MPI** (OpenMPI, MPICH) | Distributed messaging for the eventual multi-GPU work (Bible §XI.C.5). |
| **zlib** | Decompression for `.gz`-packed benchmark instances (#2). |
| **emps** | The ~250-line standalone C program from `netlib.org/lp/data/emps.c` that expands Netlib's packed column format to MPS text. A file-format decompressor with zero optimization logic — it never sees an objective, a bound, or a basis. Built into `benchmarks/.tools/` by `fetch_corpus.py`, never into the solver. Remove it and we fetch MPS from another mirror. |

### 5.5 The benchmark oracle (ticket #2)

The external solver used as a correctness oracle is **HiGHS**, which is
**forbidden as a library** (§3.1). That is not a contradiction: the ban is on
*linking* it. Running its command-line executable as a subprocess over MPS/LP
files on disk is the sanctioned use under the Oracle rule (§2, Bible Part VII).

The boundary is enforced physically, not by intent:

- `tools/oracle/install_highs.sh` builds the HiGHS CLI into `build-oracle/` — a
  path in `.gitignore` and in `sovereignty.toml`'s `exclude_globs`, invisible to
  both the sovereignty scan and the solver's CMake project.
- `tools/oracle/run_oracle.py` only ever `subprocess.run`s the binary and parses
  its output files. It never imports a solver; if it cannot spawn one it reports
  status `error` rather than falling back to an in-process library. Its JSON
  output carries `backend.linked = false` as an asserted invariant.
- If `build-oracle/` were deleted, the sovereignty check and every solver test
  still pass. That is the proof it is scaffolding, not a dependency.

Any solver reachable as a file-in/answer-out CLI can be substituted via
`SOLVER_ORACLE=/path/to/solver`; `run_oracle.py` also has adapters for `cbc` and
(stub) `glpsol`/`scip`.

---

## 6. How the check works

`tools/sovereignty_check.py` reads `sovereignty.toml` and scans:

1. **Build manifests** — `CMakeLists.txt`, `*.cmake`, `pyproject.toml`,
   `requirements*.txt`, `vcpkg.json`, `conanfile.*`, `meson.build`, `Makefile`.
2. **Source** — C/C++/CUDA/HIP and Python, catching `#include` and `import`.
3. **Submodules** — `.gitmodules`.
4. **Vendored directories** — any subdirectory of `third_party/`, `extern/`,
   `vendor/` and friends whose *name* matches a forbidden entry. This catches a
   raw source drop that no manifest mentions.
5. **The resolved dependency graph** — with `--cmake-build-dir`, it reads
   `CMakeCache.txt`, `build.ninja` and `link.txt`. This matters because a
   manifest can be perfectly clean while a transitively-pulled target puts a
   forbidden library on the link line. Declared intent and resolved reality are
   different things, and the ticket's pass condition names the dependency
   *graph*.

### Why not just grep

The ticket says "grep the build manifest against the FORBIDDEN list". A literal
substring grep is both too loud and too quiet:

| Text | Substring grep | This checker |
|---|---|---|
| `import scipy` | ✗ false positive on `scip` | correctly ignored |
| `-lscip` | matches | matches |
| `libscip.so.8` | matches | matches |
| `coinor-libcbc-dev` | matches | matches |
| `SCIP_DIR` | matches | matches |
| `AES_CBC_MODE` | ✗ false positive | flagged, until a justified exception is added |

So the checker tokenizes instead: it splits on path and word separators, strips
`lib` prefixes and file extensions, and matches whole names. `scipy` never
resolves to `scip`, while `libscip.so.8` does.

The `scipy` case is not academic. A check that cries wolf on a legitimate
dependency gets disabled within a fortnight, and a disabled check protects
nothing.

### Exceptions

Confirmed false positives are declared in `[[exceptions]]` with a mandatory
`reason` and an optional path scope. The checker **refuses to run** on an
exception with no reason — silent suppression is not available, because an
undocumented exception is indistinguishable from a breach.

### Running it

```sh
python3 tools/sovereignty_check.py                          # scan the tree
python3 tools/sovereignty_check.py --check-ledger -v        # + ledger drift
python3 tools/sovereignty_check.py --cmake-build-dir build  # + dependency graph
python3 tools/sovereignty_check.py --explain nauty          # why is X classified?
python3 tools/test_sovereignty_check.py                     # test the checker
```

Exit codes: `0` clean · `1` violation · `2` policy error. A malformed policy
exits `2` rather than `0`, so a broken policy can never read as a pass.

It is also wired into the build itself: `cmake --build` runs the
`sovereignty_check` target as part of `ALL`, so a forbidden dependency breaks a
plain local build, not only CI.

---

## 7. Maintenance

This ledger is **living**. The rules:

1. **Classify before adding.** Unclassified is not the same as permitted. A new
   dependency gets an entry in `sovereignty.toml` — with its reason, in writing —
   in the same commit that introduces it.
2. **Write the reason for a reader who wasn't there.** "It's fine" is not a
   classification. Apply the vanishing test and record the answer.
3. **Never suppress without justification.** The checker enforces this.
4. **Re-examine the restricted tier when scopes change.** If L0 stops needing a
   fill-reducing ordering, METIS should leave the ledger, not linger.
5. **The check is not advisory.** A red sovereignty check is a broken build.

### Open items

- **Transitive native dependencies** are covered today by scanning the resolved
  CMake graph, which requires a configured build. Once real third-party targets
  exist (Phase 2 onward), extend CI to configure the build before checking, so
  the graph scan runs on every commit rather than only when a build tree happens
  to be present.
- **Python transitive dependencies** are not yet lock-file-verified. When a
  `requirements.lock` or equivalent exists, scan it — the PuLP-vendors-CBC case
  is exactly the kind of thing a lock file makes visible.
- **Short-name noise.** A few forbidden entries are short, collision-prone tokens
  (`dip`, `scs`, `dsp`, `clp`). They are deliberately matched loosely: a false
  positive costs one justified exception, while a false negative costs the
  project. If noise becomes a real burden, tighten matching per-entry rather than
  dropping the entry.

---

*Companion documents: `Sovereign_Solver_Bible.md` (§1.2 for the sovereignty
constraint) and `Sovereign_Solver_Build_Map.md` (ticket #1).*
