# Sovereign Solver — Detailed Build Map & Step-by-Step Implementation Guide (v1)

**For anyone building the Sovereign Solver who has NOT read the full Bible.** This file is self-contained and ordered so each ticket can be built and tested on its own before moving to the next. The companion `Sovereign_Solver_Bible.md` has the deep "why" — read it if a ticket's rationale is unclear. Where a ticket cites a Bible section (e.g. §4.2A), that's where the reasoning lives.

**Build model:** long-horizon research build, not a fixed sprint. Progress is **milestone-gated**, not clock-gated: each milestone (M0–M9) has a single **benchmark pass condition** that must go green before the next milestone's dependent tickets are trusted. The gates map 1:1 onto the Bible's Part IX build sequence. Unlike a hackathon, there is no "demo-ready by hour 8" — the equivalent discipline here is **"never build on an unvalidated numerical layer,"** because a silent error in a lower ticket doesn't just look wrong on stage, it invalidates every benchmark number above it.

This map is granular on purpose: bigger pieces are split so each ticket is buildable and independently testable in a single sitting, and every ticket has an explicit **Test** step, not just a "done when" description. For a numerical solver, the Test step is load-bearing — an optimization routine that returns a plausible-looking wrong answer is worse than one that crashes.

---

## How to Read This Map

Tickets are numbered in strict dependency order — `Blocked by` only ever points to a lower ticket number, so the logical build sequence never leaves you stuck on something undone.

### Priority tag

| Tag | Meaning |
|---|---|
| 🟥 **CORE** | The spine, or a headline differentiator. If you build nothing else, build these. Without them there is no solver. |
| 🟨 **SUPPORTING** | Real capability that makes the solver competitive, but the spine solves problems without it. Build after the spine works end-to-end on its target class. |
| ⬜ **SEED / FRONTIER** | A research bet where the literature is thin or silent. Build a *token / prototype* version to prove the idea, or carry it as a stated roadmap claim — the production version is future work, not a milestone target. |

### Track tag (which compute substrate a ticket primarily lives on)

| Tag | Meaning |
|---|---|
| 🟩 **GPU** | Device-side kernels, VRAM residency, CUDA/ROCm primitives. The performance spine. |
| 🟦 **CPU** | Host-side irregular logic — tree search, symmetry algebra, conflict derivation, decomposition masters, parsers. |
| 🟪 **HOST↔DEVICE** | The ticket's whole job is the boundary: batching, memory orchestration, warm-start overwrites, crossover hand-off. These are where speedups are won or lost. |

### Type tag

| Tag | Meaning |
|---|---|
| **Research** | Outcome is uncertain; the ticket may return "this doesn't work, here's the fallback." Budget for that. |
| **Prototype** | Known-achievable; the work is in doing it correctly and testing it. |
| **Infra** | Plumbing, harnesses, abstractions — no math, but everything above depends on it. |

### Per-ticket fields

Each ticket has: **Build** (what to actually make), **Watch out** (the specific mistake the Bible already flagged for this piece), **Test** (how to verify it works, standalone), **Done when** (the pass condition). Plus a **Gate** line: the milestone this ticket must be green for.

**Golden rule:** don't start a ticket until its blockers show green on their own Test step. A ticket that "mostly works" is not done — in a numerical stack the next ticket silently inherits its rounding errors, its degeneracy stalls, and its wrong duals, and you will spend three tickets' worth of time discovering the bug was two layers down.

**Oracle rule (solver-specific):** every ticket that computes an optimization result must be tested against an **independent answer** — either our own from-scratch simplex (#4), or an external permissively-licensed solver run strictly as a black-box CLI oracle (never linked; see Bible §1.2 and Part VII). "It returned a number" is never a passing test. "It returned the *same* number as the oracle, to tolerance" is.

---

## Milestone Gates (the spine of the schedule)

| Gate | Name | Pass condition (benchmark, not vibes) |
|---|---|---|
| **M0** | Substrate & correct LP | From-scratch CPU simplex solves Netlib to tolerance vs. oracle |
| **M1** | GPU continuous core | PDHG + pivoting-free IPM solve large LP on GPU; match oracle to tolerance, beat CPU wall-clock on a large-network instance |
| **M2** | Crossover | Exact vertex emitted from a GPU interior solution without erasing the GPU speedup |
| **M3** | Presolve & structure detection | ~90% of a commercial presolve's reductions captured; block/scenario structure classifier working |
| **M4** | Decomposition (Benders first) | Decomposition beats the monolithic solve on a stochastic unit-commitment / capacity-expansion set |
| **M5** | MILP engine core | Solves MIPLIB 2017 subset; beats one open-source MILP solver on that subset |
| **M6** | Symmetry + conflict + determinism | Measurable node-count reduction on high-symmetry / degenerate MIPLIB; bit-identical results across core counts |
| **M7** | RL/GNN augmentation | Learned policy beats default heuristic on held-out (out-of-distribution) instances |
| **M8** | Interfaces & domain hardening | Pyomo/PuLP/CVXPY drop-in works; robustness proven on real refinery/power/VRP models |
| **M9** | Full benchmark & validation | CI runs the whole corpus every commit; results reproduce and are logged held-out vs. development-time |

**Scheduling rule (solver-specific):** the GPU and CPU tracks can run in parallel across people once M0 is green — one person can push the GPU continuous core (Phase 2) while another builds the CPU simplex oracle and parsers (Phase 1), because they only meet at the oracle test. But **nothing in Phases 4–8 is trustworthy until M2 (crossover) is green**, because MILP branching, decomposition masters, and every "solve many small LPs" pattern all depend on getting an exact vertex out of the GPU. Get the continuous core and crossover *validated* before spreading out.

---

## Phase 0 — Foundations & Decisions (before any math)

### #1: Sovereignty Dependency Ledger
Priority: 🟥 CORE
Type: Infra
Track: 🟦 CPU
Gate: M0
Blocked by: —

**Build:** A written, living ledger classifying every third-party dependency as either PERMITTED (numerical primitive — BLAS/LAPACK, cuSPARSE/cuBLAS/cuSOLVER/cuDSS/Thrust, rocSPARSE/rocBLAS) or FORBIDDEN (a solver library — CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, OR-Tools, cuOpt, Ipopt; plus nauty/bliss/saucy and GCG/DIP/mpi-sppy/PIPS-IPM++). Apply the Bible's test to each: *"if this vanished tomorrow, is what remains still our optimization engine?"*
**Watch out:** the whole project is disqualified if a forbidden solver library leaks into the build tree. The gray-zone traps are the graph-automorphism libs (must reimplement — see #40) and the sparse-direct solvers like MUMPS/SuiteSparse (permissible as linear algebra, but note them explicitly so nobody assumes they carry any optimization logic). See Bible §1.2.
**Test:** grep the build manifest against the FORBIDDEN list; confirm zero matches. Repeat this as a CI check, not a one-time review.
**Done when:** every dependency is classified in writing, and a CI job fails the build if a forbidden name appears in the dependency graph.

### #2: Benchmark Corpus & Oracle Harness Sourced
Priority: 🟥 CORE
Type: Infra
Track: 🟦 CPU
Gate: M0
Blocked by: —

**Build:** Pull the standard corpora locally — Netlib (LP), MIPLIB 2017 (MILP, *including* its tagged high-symmetry and high-degeneracy instances), Mittelmann instance sets, and MILPBench. Set up at least one external permissively-licensed solver as a black-box CLI oracle (invoked via MPS/LP files on disk — never linked). See Bible Part VII.
**Watch out:** the oracle is a *testing* tool, not a dependency — it must be reachable only through file-in / answer-out CLI calls, or it violates #1. Also: keep the MIPLIB symmetry/degeneracy subset tagged separately from day one — it's the specific evidence for M6, and you'll want it isolated.
**Test:** run the oracle on one Netlib instance from the CLI; confirm you get an objective value and a solution vector back on disk.
**Done when:** all corpora are local, and the oracle answers a sample instance through a clean file-based interface.

### #3: Numerical Substrate Skeleton (L0)
Priority: 🟥 CORE
Type: Infra
Track: 🟪 HOST↔DEVICE
Gate: M0
Blocked by: —

**Build:** The L0 foundation: sparse matrix containers (CSR + CSC, with VBCSR and BSR stubs for later), a mixed-precision numeric type policy (fp64 baseline, fp32/fp16 paths flagged but unused yet), and — critically — the **backend abstraction interface** that routes every primitive call (SpMV, factorization, reduction, sort-by-key) through a thin layer with a CUDA implementation and a ROCm implementation stub. See Bible §4.1.
**Watch out:** if you don't put the CUDA↔ROCm abstraction in *now*, you will hardcode CUDA calls everywhere and lose hardware sovereignty (Bible Bet, Part V) — retrofitting it later is a rewrite. VRAM residency is the other non-negotiable: matrices load once and stay on device.
**Test:** allocate a CSR matrix on device, run one SpMV through the abstraction layer against a hand-checked dense result; confirm the abstraction dispatches correctly and the numbers match.
**Done when:** a matrix lives in VRAM, one primitive runs through the backend abstraction, and the ROCm path compiles (even if stubbed).

---

## Phase 1 — Correct LP on the CPU (the oracle spine — M0)

> This phase exists so that everything above it has something honest to be tested against. The CPU simplex here is not a performance component — it is the *correctness oracle we own*, and the MPS/LP parser it needs is shared by the entire stack.

### #4: From-Scratch CPU Revised Simplex
Priority: 🟥 CORE
Type: Prototype
Track: 🟦 CPU
Gate: M0
Blocked by: #3

**Build:** A revised dual simplex, written from mathematical foundations (Bible §4.2 Engine C). Include anti-cycling (Bland / lexicographic tie-breaking) from the start. This is Engine C in the concurrent pool later, but its first job is to be the internal oracle.
**Watch out:** this is deliberately *not* where performance lives (Bible Part II — don't chase the saturated CPU curve). Build it correct and clear, not fast. Skipping anti-cycling means it stalls on exactly the degenerate instances the PS grades you on.
**Test:** solve a handful of Netlib instances; confirm each objective matches the external oracle (#2) to tolerance, and confirm it terminates (doesn't cycle) on a known-degenerate instance.
**Done when:** the CPU simplex agrees with the external oracle on Netlib, and survives a degenerate instance without cycling.

### #5: MPS / LP Parser
Priority: 🟥 CORE
Type: Prototype
Track: 🟦 CPU
Gate: M0
Blocked by: #3

**Build:** A parser for MPS and LP file formats into the L0 sparse containers. This is the front door for every benchmark instance and later for the Python bindings (Bible §4.7).
**Watch out:** MPS has real dialect quirks (RANGES, BOUNDS types, free rows) — a parser that silently mishandles one produces a *different problem* than the benchmark intends, and your "wrong answer vs. oracle" will actually be a parsing bug. Test against the oracle's own parse.
**Test:** parse 10 varied MIPLIB/Netlib instances; confirm the dimensions (rows, cols, nonzeros) and objective sense match what the oracle reports for the same files.
**Done when:** the parser reproduces instance dimensions and structure identically to the oracle across a varied sample.

### #6: Preconditioning Module (Ruiz + Pock–Chambolle)
Priority: 🟥 CORE
Type: Prototype
Track: 🟩 GPU
Gate: M0
Blocked by: #3

**Build:** Ruiz rescaling (iteratively equilibrate row/column norms to unity) and Pock–Chambolle diagonal preconditioning, as a mandatory front-end that the GPU continuous engines (#8, #9) will require. See Bible §4.1, §6.2.
**Watch out:** this is not optional polish — without it PDHG stalls for thousands of iterations on industrial matrices (Bible §6.2). Build and validate it *before* PDHG so PDHG is never debugged on an un-preconditioned matrix.
**Test:** take an ill-scaled instance, apply the preconditioner, confirm row/column norms are driven toward unity and the condition-number estimate drops measurably.
**Done when:** preconditioning measurably improves conditioning on a deliberately ill-scaled instance, running on-device.

---

## Phase 2 — GPU Continuous Core (the performance spine — M1)

> This is where the strategic bet lives (Bible Part II): non-simplex, GPU-native, concurrent. Build each engine standalone against the oracle first, then race them at #10.

### #7: Farkas / Dual-Ray & Duals Plumbing
Priority: 🟥 CORE
Type: Infra
Track: 🟪 HOST↔DEVICE
Gate: M1
Blocked by: #4

**Build:** A clean internal contract for extracting dual values and, on infeasibility, a Farkas dual ray, from *any* continuous solve — simplex now, PDHG/IPM later. Downstream, Benders (#26) and conflict learning (#43) both consume this exact interface.
**Watch out:** if each engine exposes duals differently, decomposition and conflict learning become per-engine special cases. Define the contract once, here, against the simplex you already trust.
**Test:** on an infeasible LP, confirm the simplex returns a Farkas ray that certifies infeasibility (the ray, applied to the constraints, yields a contradiction); on a feasible LP, confirm duals satisfy complementary slackness to tolerance.
**Done when:** a single duals/Farkas interface returns validated dual info from the CPU simplex, ready for other engines to implement.

### #8: PDHG / PDLP Engine
Priority: 🟥 CORE
Type: Research
Track: 🟩 GPU
Gate: M1
Blocked by: #6, #7

**Build:** The first-order primal-dual hybrid gradient engine — SpMV + elementwise proximal projection per iteration, fully on-device, with adaptive restarts (Halpern / Peaceman–Rachford) to handle the two-stage convergence behavior. Support separable quadratic objectives via proximal maps (this is what serves QP and Progressive Hedging later). See Bible §4.2 Engine A.
**Watch out:** PDHG's native accuracy is ~1e-4 — do NOT try to squeeze exact precision out of it here; that's crossover's job (#11). Its documented weak point is slow tail convergence; the restarts are what rescue it, so build them in from the start, not as a patch.
**Test:** solve a large-network LP on-device; confirm the objective matches the oracle to ~1e-4 and the iterate is primal-dual feasible to tolerance. Confirm restarts fire and measurably shorten the tail on a case that otherwise plateaus.
**Done when:** PDHG solves a large LP on-device to first-order tolerance, matching the oracle, with restarts working.

### #9: Regularized Pivoting-Free IPM Engine
Priority: 🟥 CORE
Type: Research
Track: 🟩 GPU
Gate: M1
Blocked by: #6, #7

**Build:** The second-order barrier engine. Regularize the KKT system with primal/dual proximal terms into Symmetric Quasi-Definite (SQD) form so it admits an `LDLᵀ` factorization for any ordering; run symbolic factorization once on the CPU, numerical factorization + back-substitution on the GPU via cuDSS. See Bible §4.2 Engine B.
**Watch out:** the regularization schedule is the single most likely place to compute a silently-slightly-wrong answer (Bible §6.1, §10.5) — too much regularization biases the solution, too little reintroduces the instability you regularized to avoid. This engine is your *high-precision* path, so its precision claims must be validated hard against the oracle, not eyeballed.
**Test:** solve the same large LP as #8; confirm the objective matches the oracle to tight tolerance (much tighter than PDHG's 1e-4) and the KKT residual is driven down cleanly as the barrier parameter shrinks.
**Done when:** the IPM solves a large LP on-device to tight tolerance via GPU factorization, matching the oracle, with no pivoting on the critical path.

### #10: Concurrent Engine Race
Priority: 🟥 CORE
Type: Prototype
Track: 🟪 HOST↔DEVICE
Gate: M1
Blocked by: #8, #9, #4

**Build:** The concurrency harness that launches PDHG (GPU), pivoting-free IPM (GPU), and dual simplex (CPU) on the *same* model simultaneously, returns the answer from whichever converges first to a valid solution, and cancels the others. See Bible §4.2, Principle "concurrency by default."
**Watch out:** "first to a *valid* answer" — a fast wrong answer must lose. The cancellation must be clean (no leaked GPU memory, no half-freed factorization). This harness is reused verbatim by the decomposition race (#25), so build it general.
**Test:** run the race on a spread of instances (one huge-and-sparse, one small-and-degenerate); confirm the huge one is won by a GPU engine and the degenerate small one may be won by simplex — and every returned answer matches the oracle.
**Done when:** the race returns the correct answer from the fastest valid engine across problem classes, with clean cancellation.

---

## Phase 3 — Crossover (the MILP gateway — M2)

> Everything integer-related downstream depends on getting an exact basic vertex out of a GPU interior solution *without* a sequential CPU pass that erases the speedup (Amdahl). This is a headline differentiator and a genuine research bet (Bible §4.3).

### #11: Concurrent Checkpoint Crossover
Priority: 🟥 CORE
Type: Research
Track: 🟪 HOST↔DEVICE
Gate: M2
Blocked by: #8, #10

**Build:** Launch crossover threads from *intermediate* PDHG iterates at several tolerance checkpoints, overlapping the sequential CPU crossover (basis construction → dual push → primal push) with the still-running GPU iterations. The known-good path. See Bible §4.3.
**Watch out:** classical crossover is essentially simplex again — if you run it once, serially, after PDHG fully converges, you delete the GPU win (Amdahl). The overlap is the whole point. This is also your fallback if the spiral-axis bet (#12) doesn't pan out, so it must be solid on its own.
**Test:** take a PDHG interior solution on a large LP; confirm crossover produces a basic vertex whose objective matches the oracle exactly (not just to 1e-4), and confirm the wall-clock including crossover still beats the CPU simplex baseline.
**Done when:** an exact vertex is produced from a GPU interior solution, and total time still beats CPU — M2 gate.

### #12: Spiral-Axis Vertex Jump
Priority: ⬜ SEED / FRONTIER
Type: Research
Track: 🟩 GPU
Gate: M2 (stretch)
Blocked by: #11

**Build:** Use the documented spiral geometry of PDHG iterates (rotation improving feasibility + forward motion closing the gap) to project analytically toward the vertex the iterates spiral into, bypassing sequential pivots entirely. A publishable result if it works. See Bible §4.3, §5.2.
**Watch out:** this may not generalize beyond well-behaved LPs (Bible §10.1). Do NOT let it block M2 — #11 is the committed path; this is the moonshot layered on top. If it works even partially, it's a moat; if it doesn't, it's a clean roadmap line.
**Test:** on instances where it applies, confirm the jump lands on (or adjacent to, then cheaply corrects to) the same vertex #11 produces, faster.
**Done when:** either the spiral jump reaches the correct vertex faster than checkpoint crossover on a real class of instances, or it's documented as attempted-and-deferred with the reason.

---

## Phase 4 — Presolve & Structure Detection (M3)

### #13: Lightweight Dual-Preserving Presolve
Priority: 🟥 CORE
Type: Prototype
Track: 🟩 GPU
Gate: M3
Blocked by: #3, #7

**Build:** A GPU-native / tight-parallel presolve capturing the high-value reductions — singleton rows/columns, forcing and dominated constraints, basic bound tightening, coefficient/scaling normalization — targeting ~90% of a full commercial presolve's reductions. Dual-preserving, because L4 cut generation and pseudo-costs need valid duals. See Bible §4.4.
**Watch out:** presolve is sequential and logic-heavy by nature; a naïve CPU port becomes the Amdahl bottleneck that negates the fast GPU core (Bible §4.4). Dual-preservation is not optional — break it and you break Benders (#26) and conflict learning (#43) downstream.
**Test:** run presolve on MIPLIB instances; measure reduction in rows/cols/nonzeros vs. the oracle's presolve on the same instances; confirm duals recovered after postsolve still satisfy complementary slackness.
**Done when:** presolve captures ~90% of the oracle's reductions and postsolve reconstructs valid duals — M3 partial gate.

### #14: Hypergraph Structure Detection
Priority: 🟥 CORE
Type: Research
Track: 🟩 GPU
Gate: M3
Blocked by: #13

**Build:** Multi-level hypergraph partitioning (columns as nodes, rows as hyperedges, minimize the cut-size corresponding to coupling constraints) to detect block-angular / doubly-bordered (arrowhead) / scenario structure and permute the matrix accordingly. This feeds the L1.5 dispatch decision (#25). See Bible §4.4, §4.2A.
**Watch out:** hypergraph partitioning is NP-hard and historically CPU-heuristic-solved (like GCG's, which you may study but not link — #1). It must be *parallelized* here, or it becomes the very presolve bottleneck it's meant to enable decomposition around (Bible §10.11). This is flagged Research because a fast parallel partitioner is a genuine open question.
**Test:** on a known block-angular instance (e.g. a stochastic program with obvious scenario blocks), confirm the partitioner recovers the true block structure and emits a clean routing decision (DW vs. Benders vs. PH).
**Done when:** the classifier correctly identifies structure on instances with known decomposable form and routes them — M3 gate.

---

## Phase 5 — Decomposition Orchestrator L1.5 (M4)

> Build in the Bible's risk-ranked order: **Benders first** (lowest risk, highest ROI, hits the PS-named power/stochastic domains), then Dantzig-Wolfe LP mode, with Branch-and-Price as an explicit frontier stretch and Progressive Hedging as a matheuristic complement. See Bible §4.2A, §5.6. Everything here reuses the "solve many small related LPs" machinery — build that reuse deliberately (#24).

### #24: Batched Related-LP Warm-Start Engine
Priority: 🟥 CORE
Type: Infra
Track: 🟪 HOST↔DEVICE
Gate: M4
Blocked by: #10, #11

**Build:** The shared machinery for solving many small related LPs that differ only in objective (reduced costs) or RHS (fixings): keep CSR/VBCSR layouts resident, overwrite only the changed vectors in place via `cudaMemcpy`, and warm-start each solve from the previous primal/dual iterate. This one engine serves decomposition subproblems (#26, #29, #32) AND the L4 GPU-batched node bounding (#39). See Bible §4.2A.4.
**Watch out:** the Bible is explicit that this must be implemented *once* — if decomposition and L4 batching each grow their own version, you've doubled the hardest host↔device code and they'll drift. Build it general here, before either caller exists.
**Test:** solve a batch of LPs that differ only in RHS; confirm warm-starting from the neighbor's solution cuts iteration count substantially vs. cold starts, and that no layout rebuild happens between solves (profile the memcpy pattern).
**Done when:** batched related-LPs solve with in-place vector overwrite and warm-start, no layout rebuilds, one reusable engine.

### #25: L1.5 Dispatch & Decomposition Race Harness
Priority: 🟥 CORE
Type: Prototype
Track: 🟦 CPU
Gate: M4
Blocked by: #14, #10

**Build:** The orchestrator that takes the structure classification (#14) and either routes to a decomposition mode or, under the concurrency principle, *races* the chosen decomposition against the monolithic solve (#10), first-to-optimality wins. Reuses the #10 race harness. See Bible §4.2A, §5.6.
**Watch out:** decomposition doesn't *replace* the monolith — it races it (Bible §4.2A). A structured instance where decomposition happens to be slow must still be solved by the monolithic engine, not hang. Route by detected structure, but never bet everything on the route being optimal.
**Test:** feed a block-structured instance; confirm the orchestrator spins up the right mode on some streams and the monolith on others, and returns the first correct answer.
**Done when:** the orchestrator dispatches by structure and races decomposition against the monolith, returning the correct answer either way.

### #26: Benders / Integer L-Shaped — Master + GPU Scenario Subproblems
Priority: 🟥 CORE
Type: Prototype
Track: 🟪 HOST↔DEVICE
Gate: M4
Blocked by: #24, #25, #7

**Build:** The Benders spine: a CPU master problem (lives inside the L4 tree later; standalone LP master for now), broadcasting trial first-stage variables to the GPU, which evaluates recourse scenarios as independent L1 LPs massively in parallel (#24), returning feasibility cuts (infeasible scenarios) and optimality cuts (suboptimal-feasible) via the Farkas/duals interface (#7). See Bible §4.2A.1.
**Watch out:** this is the highest-ROI decomposition mode and the one that hits the PS's named power-dispatch/stochastic domains — protect its build time. The master must NOT be re-solved to exact optimality every iteration in the integer case (that discards the non-resumable B&B tree — use lazy constraint callbacks once #38 exists).
**Test:** on a two-stage stochastic LP with known optimum, confirm Benders converges to that optimum, and that infeasible scenarios generate valid feasibility cuts (verified via the Farkas ray).
**Done when:** Benders converges to the known optimum on a stochastic test instance with valid feasibility and optimality cuts.

### #27: Benders — Magnanti-Wong-Papadakos Deepest Cuts
Priority: 🟨 SUPPORTING
Type: Prototype
Track: 🟩 GPU
Gate: M4
Blocked by: #26

**Build:** The deepest-cut refinement: for each scenario, after the standard subproblem, solve a secondary GPU cut-generating LP against an interior core point x⁰ to select Pareto-optimal (deep) cuts instead of the first weak dual vertex. See Bible §4.2A.1.
**Watch out:** on degenerate network-flow / unit-commitment systems, the naïve first-dual-vertex cut is weak and blows up iteration count. This doubles subproblem work but is easily absorbed by GPU parallelism — the trade is worth it. Needs a valid interior core point; a bad x⁰ gives useless cuts.
**Test:** compare Benders iteration count with vs. without MWP cuts on a degenerate instance; confirm MWP materially reduces total iterations to the same optimum.
**Done when:** deepest cuts measurably cut Benders iteration count on a degenerate instance vs. plain cuts.

### #28: Benders — GPU k-Medoids Cut Filtering + Level Stabilization
Priority: 🟨 SUPPORTING
Type: Prototype
Track: 🟩 GPU
Gate: M4
Blocked by: #26

**Build:** Two anti-bloat/anti-oscillation mechanisms: (a) GPU k-medoids clustering on cosine similarity of cut coefficient vectors, injecting only one max-violation representative per cluster (up to ~95% fewer constraints crossing to the CPU master); (b) level-method stabilization projecting the incumbent onto a level set, formulated as a small dense QP solved on-device. See Bible §4.2A.1.
**Watch out:** without cut filtering, thousands of scenarios returning multi-cuts explode the master matrix and kill re-optimization speed. The level-method QP is small but *dense* — solve it on GPU (cuDSS / dense IPM), don't ship it to the CPU. Level-set parameter tuning is fiddly (Bible §4.2A.1 residual risk).
**Test:** with many scenarios, confirm k-medoids reduces injected constraints by a large factor without materially weakening the bound; confirm level stabilization damps the early "bang-bang" oscillation in the master's first-stage solution.
**Done when:** cut filtering shrinks master growth dramatically and level stabilization visibly reduces oscillation.

### #29: Dantzig-Wolfe — cuDSS RMP + Async Column Generation (LP mode)
Priority: 🟨 SUPPORTING
Type: Research
Track: 🟪 HOST↔DEVICE
Gate: M4
Blocked by: #24, #25

**Build:** DW in LP mode: the Restricted Master Problem (small, dense, degenerate) solved on-device via cuDSS pivot-free factorization (NOT PDHG — wrong tool for small-dense-degenerate); pricing subproblems batched across streams (#24), with VBCSR for heterogeneous block sizes; asynchronous column generation (append columns and re-solve the RMP as soon as any stream returns negative reduced costs, no global barrier). See Bible §4.2A.2.
**Watch out:** PDHG's ~1e-4 precision may not give accurate-enough reduced costs to keep the RMP from stalling — this is an explicit open question (Bible §6.5, §10.9). If the RMP stalls, that's likely why; validate reduced-cost accuracy before blaming the RMP solver.
**Test:** on a block-angular LP with known optimum, confirm column generation converges to that optimum, and that async column appends don't corrupt the RMP (compare against a synchronous run's optimum).
**Done when:** DW LP mode converges to the known optimum on a block-angular instance with asynchronous column generation working.

### #30: Dantzig-Wolfe — Wentges Dual Smoothing
Priority: 🟨 SUPPORTING
Type: Prototype
Track: 🟩 GPU
Gate: M4
Blocked by: #29

**Build:** Dual price smoothing against the tailing-off effect: Wentges smoothing (`π̃ = α·π̂ + (1-α)·π`) with in-out separation, α auto-adapted by a trust-region scheme computed entirely in cuBLAS — no CPU intervention. See Bible §4.2A.2.
**Watch out:** without smoothing, late-iteration duals oscillate and generate irrelevant columns (tailing-off). Keep the α-schedule on-device (cuBLAS) — routing it through the CPU per iteration reintroduces the latency the async design removed.
**Test:** compare column-generation iteration count and column relevance with vs. without smoothing on a degenerate RMP; confirm smoothing reduces wasted columns and iterations.
**Done when:** dual smoothing measurably reduces tailing-off on a degenerate instance, computed on-device.

### #31: Dantzig-Wolfe → Branch-and-Price (MILP)
Priority: ⬜ SEED / FRONTIER
Type: Research
Track: 🟪 HOST↔DEVICE
Gate: M5 (stretch)
Blocked by: #29, #38

**Build:** Embed column generation inside the L4 branch-and-cut tree, branching on the *original* x variables via Ryan-Foster branching (never on the fractional master λ, which destroys pricing-problem structure), adding localized bound constraints to subproblems without altering VBCSR layout. See Bible §4.2A.2, §5.6.
**Watch out:** the literature on full GPU Branch-and-Price is described as "virtually nonexistent" (Bible §4.2A "state of literature", §10.7) — this is genuine frontier, highest-risk item in the decomposition layer. Do NOT let it block M4 or M5; it's a stretch/roadmap claim, not a committed deliverable. Branching on λ instead of x is the classic fatal mistake — it blocks regeneration of previously selected columns.
**Test:** on a small structured MILP, confirm Ryan-Foster branching produces the correct integer optimum and that pricing subproblems still generate valid columns after branching bounds are added.
**Done when:** either B&P reaches the correct integer optimum on a small structured MILP, or it's carried as a documented frontier roadmap item with the warp-divergence/dynamic-memory risks stated.

### #32: Progressive Hedging (matheuristic complement)
Priority: 🟨 SUPPORTING
Type: Research
Track: 🟩 GPU
Gate: M4
Blocked by: #24, #8

**Build:** PH for large-scenario stochastic problems: instantiate |Ω| independent scenario solves directly in VRAM (each via #8's PDHG, which handles the augmented-Lagrangian quadratic penalty natively), compute consensus by pure GPU reduction (Thrust/cuBLAS — zero CPU, zero PCIe round-trip), update multipliers. For MILP, add adaptive element-wise ρⱼ and scenario bundling (exact non-anticipativity within BSR-formatted bundles, PH only between them). See Bible §4.2A.3.
**Watch out:** for MILP, PH is explicitly a *matheuristic* — it cannot guarantee optimality or a valid lower bound (Bible §4.2A.3, §10, high residual risk). Present it and use it as such; never claim exactness from a PH-MILP result. Aggressive ρ updates cause discrete "slamming" — bundling is the stabilizer.
**Test:** on a continuous stochastic LP, confirm PH converges to the known optimum. On a stochastic MILP, confirm it reaches a good feasible solution and that bundling prevents the cyclic slamming a naïve run exhibits — while labeling the result heuristic.
**Done when:** PH solves the continuous case to optimum and produces stable good-feasible MILP solutions with bundling, clearly flagged as heuristic — M4 gate.

---

## Phase 6 — MILP Engine Core (M5)

> Numbering note: Phase 5's DW/B&P tickets (#29–#31) are dependency-ordered against this phase — #31 (B&P) blocks on #38 below. Build the L4 tree core first; it's what makes the integer world real.

### #38: Branch-and-Cut Tree Core
Priority: 🟥 CORE
Type: Prototype
Track: 🟦 CPU
Gate: M5
Blocked by: #11, #7

**Build:** The CPU-side branch-and-cut spine: node pool, branching (pseudo-cost / reliability to start), incumbent management, and a cut pool with Gomory / MIR / cover cut generators. LP relaxations solved via the continuous core + crossover. See Bible §4.5.
**Watch out:** the CPU owns the irregular tree logic — resist the temptation to force tree control onto the GPU (branching divergence is why full GPU B&B is unsolved, Bible §4.5, §10.2). The GPU's job is bounding, not tree navigation. Depends on crossover (#11) because branching needs exact vertices.
**Test:** solve a set of small-to-medium MIPLIB instances; confirm each integer optimum matches the oracle, and that cuts actually tighten the relaxation (root gap shrinks when cuts are enabled).
**Done when:** the tree solves MIPLIB instances to the oracle's integer optimum with working cut generation — M5 partial gate.

### #39: GPU-Batched Node Bounding
Priority: 🟥 CORE
Type: Research
Track: 🟪 HOST↔DEVICE
Gate: M5
Blocked by: #38, #24

**Build:** Solve many sibling/cousin nodes' LP relaxations simultaneously as a batched GPU operation via the shared batched-LP engine (#24) — every node runs the *same* PDHG kernel, differing only in bound vectors, sidestepping warp divergence at the node level. Warm-start each from the parent. See Bible §4.5 "aggressive bet."
**Watch out:** this is a partial answer to the GPU-B&B open problem, not a full one — it parallelizes *bounding*, not the whole tree (Bible §10.2). Don't overclaim it as "B&B on GPU." The win comes only if node batches are large enough to amortize dispatch latency (ties into the batch-threshold tuning in #46).
**Test:** batch-bound a set of sibling nodes; confirm the batched objective bounds match what per-node sequential solving produces, and that throughput (nodes bounded/sec) beats sequential bounding at batch sizes near the target threshold.
**Done when:** batched node bounding matches sequential bounds and beats it on throughput at realistic batch sizes.

### #40: Shared Solution Pool + Fix-and-Propagate Heuristics
Priority: 🟥 CORE
Type: Prototype
Track: 🟪 HOST↔DEVICE
Gate: M5
Blocked by: #39

**Build:** The CHAP-style shared pool: the GPU streams low-precision LP relaxation solutions into a pool; CPU-side fix-and-propagate and rounding/feasibility-pump heuristics (and later neural diving) consume it to surface incumbents early and crush the primal integral. See Bible §4.5.
**Watch out:** this is the pragmatic answer to "we can't run exact B&B on GPU" — use the GPU as a bounding/heuristic firehose, keep discrete logic on CPU (Bible §5.5). The pool is shared state under concurrency — mind race conditions on incumbent updates (foreshadows the determinism work in #45).
**Test:** confirm heuristics pull relaxation solutions from the pool and produce feasible incumbents earlier than pure branch-and-bound would (measure primal integral with vs. without).
**Done when:** the pool feeds heuristics that measurably improve the primal integral vs. tree search alone — M5 gate (beat one open-source MILP solver on the subset).

---

## Phase 7 — Symmetry, Conflict Learning & Determinism (M6)

> These three CPU-side subsystems form a filtration membrane upstream of the GPU batch: every LP admitted to the batch must carry genuinely novel bounding information. Documented to cut B&B node count 40–50% on degenerate/symmetric MIPLIB — the exact class the PS grades on (Bible §4.5.1–4.5.4, §5.7).

### #41: Bipartite Graph + From-Scratch Individualization-Refinement
Priority: 🟨 SUPPORTING
Type: Research
Track: 🟦 CPU
Gate: M6
Blocked by: #38, #1

**Build:** Construct the colored bipartite graph (variable nodes colored by objective coeff + bounds + type; constraint nodes by sense + RHS; numeric coefficients as auxiliary colored chain-nodes), then a from-scratch Individualization-Refinement engine: equitable-partition color refinement, individualization branching in an IR search tree, generating the automorphism group. Allocation-free structures (doubly-linked lists + contiguous cell-boundary arrays). See Bible §4.5.1.
**Watch out:** nauty / bliss / saucy are FORBIDDEN (#1) — this must be reimplemented from the mathematical foundation. On asymmetric instances, blind IR wastes CPU cycles — gate it behind cheap degree/hash fingerprinting that aborts when no symmetry is plausible (Bible §4.5.1 residual risk).
**Test:** on an instance with known symmetry (e.g. a set-partitioning problem), confirm IR recovers the expected automorphism generators; on an asymmetric instance, confirm the fingerprint check aborts IR fast without running the full search.
**Done when:** IR recovers known symmetry generators and cheaply bails on asymmetric instances.

### #42: Schreier-Sims BSGS + Orbital Branching + Orbitopal Fixing
Priority: 🟨 SUPPORTING
Type: Research
Track: 🟦 CPU
Gate: M6
Blocked by: #41

**Build:** Represent the (possibly astronomically large) symmetry group compactly via Schreier-Sims — base + strong generating set, Schreier vectors, Sims filter for polynomial memory bound (randomized variant for scale). Then orbital branching (branch on an orbit representative; left branch fixes the whole orbit to 0) and orbitopal fixing (linear-time DP against the orbitope boundary for partitioning constraints). Maintain the group incrementally down the tree via stabilizer descent (array lookups, no IR re-run). See Bible §4.5.1.
**Watch out:** explicit permutation enumeration is impossible for large groups — BSGS is mandatory. Randomized Schreier-Sims can miss obscure generators if tuned too aggressively (Bible §4.5.1 residual risk). Incremental maintenance is what keeps per-node symmetry cost cheap — don't recompute the full automorphism at each node.
**Test:** on a high-symmetry MIPLIB instance, confirm orbital branching explores measurably fewer nodes than plain branching to the same optimum, and that incremental stabilizer descent produces the correct child-node group.
**Done when:** orbital branching + orbitopal fixing cut node count on a symmetric instance, with cheap incremental group maintenance.

### #43: Isomorphism Pre-Filter into the GPU Batch
Priority: 🟨 SUPPORTING
Type: Prototype
Track: 🟪 HOST↔DEVICE
Gate: M6
Blocked by: #42, #39

**Build:** Before a sibling/cousin node is admitted to the GPU batch (#39), query the BSGS: if the node is symmetric to an already-bounded or already-queued node, prune it on the CPU for free — the GPU never spends a slot on a provably redundant LP. See Bible §4.5.1 "GPU interaction."
**Watch out:** this is the payoff that connects symmetry to the batching architecture — symmetric-node pruning is what compounds effective GPU throughput (Bible §4.5.4). The orbit query must be cheap (BSGS transversal lookup), or the filter costs more than the GPU slot it saves.
**Test:** on a symmetric instance, confirm nodes symmetric to already-processed ones are filtered out before batch dispatch, and that GPU batch occupancy now holds strictly non-redundant LPs (measure the redundant-LP rate before/after).
**Done when:** symmetric redundant nodes are pruned pre-batch and measured GPU redundant-work drops.

### #44: Cut-Based Conflict-Driven Learning
Priority: 🟨 SUPPORTING
Type: Research
Track: 🟦 CPU
Gate: M6
Blocked by: #38, #7

**Build:** When a GPU-bounded LP is infeasible or exceeds the cutoff, extract the Farkas ray (#7), form the proof-constraint, and reduce it by backward linear combination with the reason constraint for the most recent bound propagation, applying MIR / fractional saturation at each step, terminating at a 1UIP-equivalent condition — yielding a globally valid no-good cut. Manage the pool by Literal Block Distance (retain low-LBD, decay high-LBD); classify dense cuts "propagation-only" (CPU sparse-dot-product checks, excluded from the GPU LP matrix). See Bible §4.5.2.
**Watch out:** deep recursive MIR rounding amplifies floating-point drift — this is the single highest-flagged numerical risk in L4 (Bible §4.5.2, §6.6, §10.12). Run the backward derivation in rational/exact arithmetic or with strict epsilon safeguards, and re-validate every cut against the original system before injecting it — an invalid cut can prune the optimum and you'll never know. Cut-based CDCL over general-integer/continuous variables is itself experimental (Bible §10.12).
**Test:** on an infeasible subtree, confirm the derived no-good cut is globally valid (re-check it prunes only genuinely infeasible regions against the oracle) and that it prunes additional unrelated nodes downstream. Confirm exact-arithmetic derivation produces no invalid cuts across a stress set.
**Done when:** conflict cuts are validated globally valid, prune real subtrees, and derivation is numerically safe.

### #45: Symmetry-Aware Conflict Generalization
Priority: ⬜ SEED / FRONTIER
Type: Research
Track: 🟦 CPU
Gate: M6 (stretch)
Blocked by: #42, #44

**Build:** When a conflict cut is derived at a node with an active BSGS, permute its variables across the known orbits to instantly generate a whole family of symmetric no-good cuts — without re-running Farkas extraction or MIR reduction per cut. See Bible §4.5.4.
**Watch out:** this synergy is sparse in classical MIP literature (standard in SAT/CP) — a genuine differentiation point but a research bet (Bible §4.5.4). The permuted cuts must be re-validated too (same numerical caution as #44). Don't let it block M6 — it's a compounding bonus on top of #43+#44, not a gate.
**Test:** confirm a single derived cut yields multiple valid symmetric cuts via orbit permutation, each independently validated, and that injecting the family prunes disjoint subtrees a single cut wouldn't reach.
**Done when:** one conflict proof generates a validated family of symmetric cuts pruning disjoint subtrees, or it's carried as a roadmap claim.

### #46: Deterministic Work-Stealing Scheduler
Priority: 🟨 SUPPORTING
Type: Prototype
Track: 🟦 CPU
Gate: M6
Blocked by: #40, #39

**Build:** Chase-Lev lock-free work-stealing deques (one per worker; pop/push bottom locally, steal from the top of a random victim), plus structural determinism: node-count stamps as the internal clock, trailing read barriers (a discovered incumbent/cut is timestamped and not applied by other workers until their local counter passes the stamp), and deterministic tie-breaking (seeds hashed from node lineage, never thread ID). GPU batch dispatch fires on a deterministic queue-size threshold (e.g. 4096), never a wall-clock timeout. See Bible §4.5.3.
**Watch out:** determinism has a real, *accepted* cost — trailing barriers make the solver explore somewhat more nodes than an instant-sharing solver would (Bible §4.5.3, §6.7, §10.14). Mitigate the *latency* via task-queue oversubscription; never mitigate it by dropping the determinism guarantee. GPU batch starvation vs. over-aggressive CPU filtering is the single most sensitive tuning parameter in L4.
**Test:** run the same instance on 1, 4, and 16 workers; confirm bit-identical node count, search path, and final solution across all three. Confirm the GPU batch queue stays adequately fed (occupancy near threshold) under the deterministic dispatch.
**Done when:** results are bit-identical across core counts and the GPU stays fed — M6 gate.

---

## Phase 8 — ML / RL Augmentation (M7)

### #47: GNN State Encoder
Priority: 🟨 SUPPORTING
Type: Research
Track: 🟩 GPU
Gate: M7
Blocked by: #38

**Build:** A graph neural network encoding the variable-constraint bipartite graph (permutation-invariant by construction) as the state representation for the learned policies. Trained/evaluated on MILPBench (100k instances). See Bible §4.6, §5.4.
**Watch out:** this is a hot-swappable *plugin* — the solver must degrade gracefully to the #38 default heuristics if the model is absent or underperforms (Bible §4.6). Don't wire it in as a hard dependency of the tree.
**Test:** confirm the encoder produces stable embeddings invariant to variable/constraint reordering (permute an instance; confirm the embedding is unchanged).
**Done when:** the GNN encodes instances permutation-invariantly and plugs into the tree without becoming a hard dependency.

### #48: End-to-End RL Branching/Node Policy
Priority: ⬜ SEED / FRONTIER
Type: Research
Track: 🟩 GPU
Gate: M7
Blocked by: #47, #46

**Build:** A reinforcement-learning agent whose reward is the terminal objective itself — minimize global B&B tree size (equivalently, the primal-dual integral) — NOT imitation of strong branching. This deliberately avoids imitation learning's proven perturbation instability and covariate shift. See Bible §4.6, §5.4.
**Watch out:** the entire reason for choosing RL over imitation is that imitating strong branching is *mathematically proven* to fail (local score-matching → exponentially larger trees; Bible §5.4). The gate is NOT "does it train" — it's "does it beat the default heuristic on instances it never saw" (Bible §10.4). Out-of-distribution generalization is the whole test; in-distribution wins prove nothing.
**Test:** train on a MILPBench subset, evaluate on a *held-out* class the agent never trained on; confirm it reduces tree size vs. the #38 default heuristic on that held-out class, not just on training-like instances.
**Done when:** the RL policy beats the default heuristic on out-of-distribution instances — M7 gate — or is documented as a frontier attempt with results.

### #49: Learned Cut Selection
Priority: ⬜ SEED / FRONTIER
Type: Research
Track: 🟩 GPU
Gate: M7 (stretch)
Blocked by: #47, #38

**Build:** Replace the default linear cut-scoring function with a learned selector that ranks generated cuts (Gomory/MIR/cover) by expected usefulness. See Bible §4.6.
**Watch out:** cut selection is a plugin like #48 — degrade gracefully to the linear scorer. Same held-out discipline: a learned selector that only helps on training-like instances isn't a real win.
**Test:** on held-out instances, confirm the learned selector reduces solve time or node count vs. the default linear scorer.
**Done when:** learned cut selection beats the default scorer on held-out instances, or is carried as roadmap.

---

## Phase 9 — Interfaces & Domain Hardening (M8)

### #50: C-API + CLI
Priority: 🟥 CORE
Type: Infra
Track: 🟦 CPU
Gate: M8
Blocked by: #10, #38

**Build:** A stable C-API (in the shape of a thin-client / `libsolver`) and a CLI so existing pipelines drop the solver in with a header change, and so the benchmark harness (#2) can drive it. See Bible §4.7.
**Watch out:** the API is the drop-in path for industrial models — keep it close to the shapes existing solvers expose so integration is a header change, not a rewrite of the caller.
**Test:** solve a benchmark instance end-to-end through the C-API and again through the CLI; confirm both match the oracle.
**Done when:** the solver is callable via C-API and CLI, both producing oracle-correct results.

### #51: Python Bindings — Pyomo / PuLP / CVXPY Compatibility
Priority: 🟥 CORE
Type: Infra
Track: 🟦 CPU
Gate: M8
Blocked by: #50

**Build:** Python bindings API-compatible with Pyomo, PuLP, and CVXPY (Pyomo especially — it dominates power-dispatch/energy modeling, a named target domain). Study how PuLP serializes to MPS/LP; don't invent a modeling language. See Bible §4.7.
**Watch out:** the goal is drop-in for millions of lines of existing industrial modeling code — speak the existing frontends' protocol rather than adding a new one. Pyomo compatibility is disproportionately valuable given the power-systems use cases.
**Test:** take an existing Pyomo (or PuLP) model, swap the solver backend to ours, and confirm it solves to the same optimum the model's original solver produced.
**Done when:** an unmodified Pyomo/PuLP/CVXPY model solves correctly on our backend as a drop-in.

### #52: Domain Hardening Suite (Refinery / Power / VRP)
Priority: 🟥 CORE
Type: Prototype
Track: 🟦 CPU
Gate: M8
Blocked by: #51, #26, #40

**Build:** A curated suite of real-shape industrial models from the PS's named domains — refinery scheduling / crude blending, stochastic unit commitment (exercises Benders, #26), and vehicle routing — run through the full stack including decomposition, proving robustness on degenerate, ill-conditioned, real-world formulations. See Bible §5.6, Part VII.
**Watch out:** these are the instances the PS actually cares about — degeneracy and ill-conditioning here are the grading criteria (Bible Part VI), not edge cases. A solver that aces MIPLIB but fails a real refinery blend hasn't met the brief.
**Test:** solve each domain model to the oracle's optimum (or, where the oracle times out, to a validated feasible-with-bound); confirm reliable convergence with no divergence on the ill-conditioned cases.
**Done when:** every domain model solves reliably and matches the oracle where the oracle can solve it — M8 gate.

---

## Phase 10 — Full Benchmark & Validation Discipline (M9, do not skip)

### #53: CI/CD Regression Corpus
Priority: 🟥 CORE
Type: Infra
Track: 🟦 CPU
Gate: M9
Blocked by: #2, #10, #38

**Build:** Wire Netlib, MIPLIB 2017, Mittelmann, MILPBench, and the stochastic/block-angular decomposition sets into CI so every commit to the core triggers regression against the whole corpus, checked against the oracle. See Bible Part VII.
**Watch out:** a numerical solver silently regresses — a refactor that shifts a rounding boundary passes unit tests and fails three benchmark classes. The corpus in CI is the only thing that catches this. Any commit that changes a result must justify it.
**Test:** push a deliberately wrong change; confirm CI catches the regression against the oracle and fails the build.
**Done when:** every commit auto-runs the corpus and fails on any oracle disagreement.

### #54: Metrics Beyond Wall-Clock + Held-Out Discipline
Priority: 🟥 CORE
Type: Prototype
Track: 🟦 CPU
Gate: M9
Blocked by: #53, #42, #44

**Build:** Track solver-internal metrics — root-node gap, nodes explored, primal/dual integral — and, specifically, **node-count reduction attributable to symmetry (#42/#43) and conflict learning (#44)** as a separately-logged figure (that reduction is the direct evidence for Bet 7). Report every headline number as either **held-out** (first run on an instance class never tuned against during development) or **development-time**, and never blend them. See Bible Part VII; this mirrors the non-circular validation discipline used on prior team projects.
**Watch out:** a number tuned against during development and then reported as a result is circular — the held-out/development-time split must be honest, not decorative. The symmetry/conflict node-count reduction is your strongest concrete claim against the exact MIPLIB class the PS names; measure it cleanly or you can't defend it.
**Test:** re-run two metrics and confirm they reproduce within a sane margin; confirm at least the core node-count-reduction claim has one genuine held-out run logged separately from development-time numbers.
**Done when:** every headline metric is real, reproducible, and explicitly labeled held-out vs. development-time.

### #55: Reproducible Benchmark Report vs. Established Solver
Priority: 🟥 CORE
Type: Prototype
Track: 🟦 CPU
Gate: M9
Blocked by: #54

**Build:** The deliverable the PS asks for: a report comparing solution quality and performance against at least one established commercial or open-source solver across the corpora, plus a demonstration of numerical robustness on a curated degenerate / weak-LP-relaxation / ill-conditioned suite where weaker implementations struggle. See PS "Expected Solution", Bible Part VII.
**Watch out:** the PS pass condition is match-or-beat at least one established solver AND demonstrated robustness on hard instances — both halves are required. The robustness demonstration is graded as heavily as the speed comparison (Bible Part VI); don't let a strong speed table hide a thin robustness story.
**Test:** confirm every number in the report traces to a reproducible CI run (#53) and a real held-out/development-time label (#54); dry-run the comparison against the chosen established solver.
**Done when:** the report shows match-or-beat on at least one established solver plus reliable convergence on the hard-instance suite, every number reproducible — M9 gate, PS satisfied.

---

## Future Work (out of scope for this build)

These aren't rejected ideas — they're real extensions carried in Bible Part XI as an open research queue, not yet given a deep-research pass. Worth naming if asked; worth returning to once the M0–M9 spine is proven.

1. **Anderson / Nesterov acceleration for PDHG** — extrapolation to cut iteration counts further, shrinking how often crossover is needed (Bible §XI.A.1). Roadmap until the continuous core is solid.
2. **Randomized / sketching-based preconditioners** — cheaper conditioning estimates for very large systems, improving both the mixed-precision gate and PDHG speed (Bible §XI.A.2).
3. **GPU-parallel automatic differentiation for MINLP / spatial B&B** — the eventual path to the NLP/MINLP extension the PS names as modular growth; almost nobody has combined GPU-native AD with spatial B&B (Bible §XI.B.3).
4. **McCormick envelope tightening on GPU** — domain-specific strength on the crude-blending pooling problem the PS names, ahead of full MINLP (Bible §XI.B.4).
5. **Multi-GPU model-parallel solving** — partition a single massive matrix across GPUs past single-GPU VRAM limits; complements (doesn't replace) the decomposition layer, since it applies even to non-decomposable monoliths (Bible §XI.C.5).
6. **Portfolio / predict-then-solve algorithm selection** — a classifier that predicts the winning engine/mode and skips racing the rest, recovering wasted GPU-hours while keeping the robustness story (Bible §XI.C.6).
7. **Full GPU Branch-and-Price (#31) at production scale** — the literature is silent here; the hackathon/first-cut build carries it as a frontier prototype at most (Bible §4.2A, §10.7).

---

*This build map is a companion to `Sovereign_Solver_Bible.md`. The Bible explains WHY; this map tells you WHAT to build, in WHAT order, how to test each piece, which track (GPU/CPU/host↔device) and milestone gate each belongs to, and what's genuinely future work. When a step and the Bible disagree, the Bible is the source of truth on intent — but follow this map's ticket order, track tags, gate assignments, and Test steps for execution. The Oracle rule and the Golden rule are the two that keep a numerical solver honest: build nothing on an unvalidated layer, and never accept "it returned a number" as a passing test.*
