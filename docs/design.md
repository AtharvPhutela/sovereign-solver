# The Sovereign GPU-Accelerated Optimization Solver — Technical Bible

*A from-scratch LP / MILP / QP solver core, engineered for the GPU era, targeting the unsolved frontiers of optimization science rather than replicating a saturated CPU legacy.*

**Revision note:** this edition integrates two deep-research passes — GPU-parallel decomposition methods (Dantzig-Wolfe/column generation, Benders, Progressive Hedging) and tree-search craft beyond plain branch-and-cut (symmetry/orbital branching, conflict-driven learning, deterministic parallel scheduling). Both are now first-class architecture, not aspirations. A new Part XI carries forward the research directions that haven't been deep-dived yet.

---

## 0. How to read this document

This is the master reference for the project — the "bible." It is opinionated on purpose. Every architectural choice here is anchored to a specific unsolved problem or contested question in the current research landscape, so that we spend our engineering effort where the field is *soft*, not where it is already cemented by forty years of proprietary heuristics.

The structure:

- **Part I** decodes the problem statement and settles the one constraint that shapes everything: *"built from scratch."*
- **Part II** states the strategic bet — why we do **not** try to out-Gurobi Gurobi on a CPU.
- **Part III** is the architecture: a layered, GPU-native-first system with the CPU demoted to an orchestration and irregular-logic role.
- **Part IV** goes layer by layer with the algorithm choices and the specific differentiator each layer carries — including the L1.5 decomposition orchestrator and the fully fleshed-out L4 mixed-integer engine.
- **Part V** is the frontier bets — the places where we are deliberately trying to advance the state of the art.
- **Part VI** covers numerical robustness, the thing the PS actually grades us on.
- **Part VII** is benchmarking and validation.
- **Part VIII** is the differentiation table — the crisp "how we differ from every open-source solver" answer.
- **Part IX** is a logical build sequence.
- **Part X** is an honest open-questions register.
- **Part XI** is a running list of research directions flagged as high-value but not yet deep-dived — candidates for the next research pass, explicitly not committed architecture.

---

# PART I — What the problem statement actually demands

## 1.1 The literal ask

Build a **solver core**, not a modeling environment. The deliverable is an engine plus a thin API/CLI. No GUI required. It must:

- Support **LP, MILP, QP** now; be **modular** enough to grow into MIQP / NLP / MINLP later.
- Exploit **sparse linear algebra**, **numerical linear algebra**, **multi-core parallelism**, and **GPU acceleration where it measurably helps**.
- Be graded on **numerical stability, scalability, and reliable convergence** — explicitly on degenerate, ill-conditioned, and hard mixed-integer instances.
- Solve standard **MIPLIB / Netlib / Mittelmann** instances and beat or match **at least one** established commercial or open-source solver.
- Serve **refinery scheduling, crude blending, process optimization, production planning, logistics, power dispatch, transportation, supply chain**.

## 1.2 The one constraint that governs the whole design

> *"It shall not be built upon any existing open source solver library but shall be built from scratch from mathematical foundation."*

This single sentence is the design's center of gravity.

**What is forbidden (a "solver library"):** CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, lp_solve, OR-Tools/GLOP, cuOpt, Ipopt, Bonmin, Couenne — and, per the new material, also **nauty, bliss, saucy** (graph automorphism libraries) and any existing decomposition framework (GCG, COIN-OR DIP, mpi-sppy, PIPS-IPM++). We may not link these, fork them, or wrap them. We may study their published algorithms and use them as **benchmarking oracles** only.

**What is permitted (a "numerical primitive"):** BLAS/LAPACK (OpenBLAS, BLIS), sparse-matrix and factorization primitives (cuSPARSE, cuBLAS, cuSOLVER, **cuDSS**, Thrust; AMD rocSPARSE/rocBLAS), and general array/parallel libraries. These are *linear algebra*, not *optimization*. Building a solver "from the mathematical foundation" does not mean re-deriving the FFT; it means **we own the optimization algorithms** — including, now, the graph automorphism engine (Individualization-Refinement, Schreier-Sims) and the decomposition orchestration (Dantzig-Wolfe, Benders, Progressive Hedging) — and assemble them on top of standard numerical bricks.

**The sovereignty test we apply to every dependency:** *"If this library vanished tomorrow, is what remains still our optimization engine?"* This test now explicitly extends to the symmetry engine and the decomposition layer, both of which must be reimplemented from their mathematical foundations rather than linked from existing tools.

---

# PART II — The strategic bet

## 2.1 The trap: rebuilding the CPU legacy

The obvious path is to reimplement a revised dual simplex + branch-and-cut stack on the CPU. **This is a trap.** The CPU paradigm is a saturated technological curve defined by decades of proprietary, un-published heuristic tuning inside Gurobi and CPLEX. A from-scratch CPU solver lands where HiGHS already is: 10–20× slower than commercial.

## 2.2 The bet: be GPU-native from the first line

Linear programming has been mapped onto the GPU via first-order methods (PDHG/PDLP), whose inner loop is nothing but sparse matrix-vector multiply plus elementwise projection — no sequential dependency, saturates thousands of GPU cores. SpMV is memory-bandwidth bound; modern GPUs offer roughly two orders of magnitude more bandwidth than CPUs. The o9 Solutions precedent (30M-variable supply-chain model, 11 minutes on 8 CPU cores → 57 seconds on one B200, within 0.008% of commercial optimum) is the proof of concept.

So we invert the relationship: **the GPU is the primary compute substrate; the CPU exists to do the irregular, logic-heavy work the GPU is bad at** — including, as of this revision, the branch-and-cut tree's discrete logic, the symmetry group algebra, the conflict-cut derivation, and the master problems of our decomposition methods.

## 2.3 Why this is a winnable frontier

The GPU-native world is genuinely unfinished. Beyond the original five frontier problems (crossover, pivoting-free factorization, ML generalization crisis, GPU-native presolve, GPU B&B divergence), the new research passes surface two more concrete, currently-unclaimed frontiers:

6. **GPU-native structure exploitation** — no open-source solver races structure-adaptive decomposition (Benders/Dantzig-Wolfe/Progressive Hedging) against a monolithic GPU solve as standard behavior. Full GPU Branch-and-Price is described in the literature as "virtually nonexistent."
7. **A sovereign, from-scratch symmetry + conflict-learning engine acting as a compounding CPU-side filter** for the GPU batch — documented to cut B&B node counts by 40–50% on the exact class of degenerate/symmetric MIPLIB instances the PS names as the grading bar.

These two join the original five as our differentiation surface.

---

# PART III — Architecture overview

The system is a **heterogeneous, asynchronous, GPU-first stack**. Data lives in GPU VRAM by default; it crosses the PCIe boundary as rarely as possible.

```
┌─────────────────────────────────────────────────────────────────────┐
│  L6  INTERFACE LAYER                                                  │
│      MPS/LP parser · C-API · Python bindings (Pyomo/PuLP/CVXPY) · CLI │
├─────────────────────────────────────────────────────────────────────┤
│  L5  ML / RL AUGMENTATION (optional, hot-swappable plugins)           │
│      GNN state encoder · end-to-end RL branching/node policy ·        │
│      learned cut selection                                            │
├─────────────────────────────────────────────────────────────────────┤
│  L4  MIXED-INTEGER ENGINE  (CPU logic + GPU bounding)                 │
│      branch-and-cut tree · SYMMETRY: bipartite graph + from-scratch   │
│      Individualization-Refinement + Schreier-Sims BSGS + orbital      │
│      branching + orbitopal fixing · CONFLICT LEARNING: cut-based      │
│      Farkas/MIR no-good derivation, LBD-managed pool · DETERMINISTIC  │
│      SCHEDULING: Chase-Lev work-stealing deques + trailing read       │
│      barriers · cut pool · shared solution pool · fix-and-propagate   │
│         ▲ bounds / incumbents / cuts    │ nodes + Benders trial pts   │
├──────────────────────────────────────── │ ────────────────────────────┤
│  L1.5 DECOMPOSITION ORCHESTRATOR  (cross-cutting: spans L1 / L3 / L4)  │
│      structure-adaptive dispatch: Benders/integer L-shaped (priority  │
│      1) · Dantzig-Wolfe / column generation, LP mode (priority 2,     │
│      Branch-and-Price is a stretch goal) · Progressive Hedging        │
│      (matheuristic complement) · races concurrently against the       │
│      monolithic L1 solve; first-to-converge wins                      │
├─────────────────────────────────────────────────────────────────────┤
│  L3  PRESOLVE  (GPU-native, dual-preserving, "lightweight")           │
│      singletons · forcing constraints · bound tightening · scaling ·  │
│      + multi-level hypergraph structure detection (feeds L1.5 routing)│
├─────────────────────────────────────────────────────────────────────┤
│  L2  CROSSOVER / VERTEX IDENTIFICATION                                │
│      concurrent checkpoint crossover · spiral-axis vertex jump        │
├─────────────────────────────────────────────────────────────────────┤
│  L1  CONTINUOUS CORE  (LP / QP)                                       │
│      ┌── PDHG/PDLP engine (GPU, first-order, fast low-precision) ──┐  │
│      │            run CONCURRENTLY, first-to-converge wins         │  │
│      └── regularized pivoting-free IPM (GPU, second-order, exact) ─┘  │
│           + CPU dual-simplex fallback for degenerate/small cases      │
├─────────────────────────────────────────────────────────────────────┤
│  L0  NUMERICAL SUBSTRATE                                              │
│      GPU-resident CSR/CSC/VBCSR/BSR · mixed precision (fp64/fp32/fp16 │
│      + iterative refinement) · preconditioning (Ruiz, Pock–Chambolle) │
│      · backend abstraction over cuSPARSE/cuBLAS/cuSOLVER/cuDSS ↔ ROCm │
└─────────────────────────────────────────────────────────────────────┘
```

Four principles hold across every layer (the fourth is new this revision):

- **VRAM residency.** The constraint matrix — and now the RMP, subproblems, and scenario replicas — stays on the device. PCIe transfers are the enemy.
- **Concurrency by default.** Multiple algorithms attack the same model simultaneously; the fastest to a valid answer wins and cancels the rest. This now explicitly includes decomposition modes racing the monolithic solve, not just L1's internal engine race.
- **The GPU proposes, the CPU disposes.** The GPU is a bound/heuristic/subproblem firehose; the CPU makes the discrete, irregular decisions — tree logic, symmetry algebra, conflict derivation, decomposition master problems.
- **The CPU-side filter must earn every GPU cycle it spends.** Symmetry pruning and conflict-cut checking exist specifically so that every LP admitted to the GPU batch carries genuinely novel bounding information — cheap CPU logic protects expensive GPU throughput.

---

# PART IV — Layer-by-layer

## 4.1 L0 — Numerical substrate

**Sparse formats.** Constraint matrices are stored CSR and CSC. The decomposition layer additionally requires **VBCSR** (Variable Block CSR, for heterogeneous pricing-problem block sizes in Dantzig-Wolfe) and **BSR** (Block Sparse Row, for Progressive Hedging's scenario bundles) — both device-resident, built once at load or at decomposition-mode entry.

**Backend abstraction.** Every primitive call — SpMV, factorization, sort-by-key, reduction — goes through a thin internal interface with CUDA and ROCm implementations. Costs a little indirection, buys hardware sovereignty.

**Mixed precision.** Baseline correctness is fp64. Mixed-precision solves with iterative refinement trade fp32/fp16 tensor-core speed for a corrected fp64 answer, falling back to full fp64 when conditioning demands it (§6.3).

**Preconditioning.** Ruiz rescaling and Pock–Chambolle diagonal preconditioning as a mandatory front-end to L1. Without this, PDHG stalls for thousands of iterations on industrial matrices.

## 4.2 L1 — Continuous core (LP / QP)

Two engines, run concurrently, first-to-converge wins:

**Engine A — PDHG / PDLP (first-order, GPU-native).** Dualize coupling constraints into the Lagrangian, alternate primal/dual proximal projections. Each iteration = SpMV + elementwise clamp. Handles PDHG's known pathologies via **adaptive restarts** (Halpern, Peaceman–Rachford) for the two-stage convergence behavior, and accepts its modest native accuracy (~1e-4), handing off to Engine B or crossover (L2) for the last digits. PDHG also natively handles **separable quadratic objectives** via proximal mapping with near-zero overhead — this is what makes it the right engine for Progressive Hedging's augmented-Lagrangian penalty term (§4.2A.3) as well as plain QP.

**Engine B — regularized pivoting-free IPM (second-order, GPU-native).** Regularizes the KKT matrix with primal/dual proximal terms into **Symmetric Quasi-Definite (SQD)** form, which admits an `LDLᵀ` factorization for any fill-reducing ordering. Symbolic factorization runs once on the CPU; numerical factorization and back-substitution run massively parallel on the GPU via **cuDSS**. No sequential pivoting on the critical path.

**Engine C — CPU dual simplex (fallback).** For small, highly degenerate, tightly constrained problems, a classical revised dual simplex on the CPU, written from scratch, running in the concurrent pool and anchoring correctness testing.

**QP.** Convex QP extends the IPM (quadratic objective folds into the KKT) and, for separable/box-constrained structure, an ADMM/OSQP-style operator-splitting variant.

## 4.2A L1.5 — Decomposition Orchestrator *(new this revision)*

### Positioning and dispatch logic

L1.5 is **cross-cutting**, not a strict sequential layer: it is triggered by structure detection during L3 presolve, its subproblems run on L1's engines, and — for Benders — its master problem lives inside L4's branch-and-cut tree. Structure detection uses **multi-level hypergraph partitioning** (columns as nodes, rows as hyperedges, minimizing the cut-size corresponding to coupling constraints) run natively during L3, permuting the matrix into block-angular or doubly-bordered (arrowhead) form. Once detected, the instance is routed:

- Block-angular structure with few coupling constraints → **Dantzig-Wolfe mode**.
- Scenario-structured model with integer first-stage variables → **Benders mode**.
- Purely stochastic, exactness not required → **Progressive Hedging mode**.

Under the concurrency-first philosophy, **decomposition does not replace the monolithic solve — it races it.** The CPU assigns the monolithic matrix to a subset of GPU streams (standard L1 IPM/PDHG) while L1.5 spins up the appropriate decomposition method on the remaining streams. Whichever converges to optimality first wins and terminates the other.

**Priority order, set by risk and ROI (per the source research's own explicit assessment):**

| Mode | Architectural risk | Algorithmic risk | Build priority |
|---|---|---|---|
| Benders / integer L-shaped | Low–moderate | Low–moderate | **1st — highest immediate ROI** |
| Dantzig-Wolfe, LP mode | Moderate | Low | 2nd |
| Dantzig-Wolfe → Branch-and-Price (MILP) | **High** — literature "virtually nonexistent" | High | Stretch goal, not a committed deliverable |
| Progressive Hedging | Low (architecture) | **High** (no MILP convergence guarantee) | Complement/fallback, matheuristic only |

### 4.2A.1 Benders / integer L-shaped decomposition

The Master Problem (MP) resides on the CPU inside L4 as a branch-and-cut tree. The **integer L-shaped method** is implemented via lazy constraint callbacks: whenever L4 finds an integer-feasible incumbent (or a fractional solution during node bounding), it suspends the node and broadcasts the trial first-stage variables $x^*$ to the GPU. Second-stage recourse scenarios are evaluated as independent L1 LPs, massively parallel across CUDA streams. Infeasible scenarios return **feasibility cuts**; suboptimal-but-feasible scenarios return **optimality cuts**. The CPU injects these directly into the active L4 tree, tightening the relaxation globally.

$$\min c^\top x + \theta \quad \text{s.t. } Ax \leq b,\ x \in X,\quad \theta \geq \sum_\omega p_\omega (h_\omega - T_\omega x)^\top \pi_\omega^k \ \ \forall k \in \mathcal{K}_{opt},\quad 0 \geq (h_\omega - T_\omega x)^\top \mu_\omega^k \ \ \forall k \in \mathcal{K}_{feas}$$

**Deepest cuts (Magnanti-Wong-Papadakos):** rather than accept the first dual vertex encountered (which can be weak/degenerate on highly degenerate network-flow and unit-commitment systems), the GPU solves a secondary cut-generating LP against an interior core point $x^0$ to find Pareto-optimal cuts:

$$\max \pi_\omega^\top(h_\omega - T_\omega x^0) \quad \text{s.t. } W_\omega^\top \pi_\omega \leq q_\omega,\quad \pi_\omega^\top(h_\omega - T_\omega x^*) = v_\omega^*$$

This doubles subproblem work but drastically cuts the number of Benders iterations needed — the doubled work is absorbed easily by GPU parallelism.

**GPU-specific mitigations:**
- **Master-problem bloat** (thousands of scenarios returning multi-cuts per iteration) is controlled by **GPU $k$-medoids cut filtering** on cosine similarity of cut coefficient vectors, selecting one representative (max-violation) cut per cluster — up to a 95% reduction in constraints injected into L4, entirely in VRAM.
- **Oscillation / slow tail convergence** is controlled by **level-method stabilization**: project the incumbent onto a level set defined by a convex combination of the current bounds, formulated as a small but dense QP, solved natively via cuDSS or a custom dense primal-dual IPM.

| Failure mode | Mitigation | Residual risk |
|---|---|---|
| Weak/degenerate cuts | Magnanti-Wong-Papadakos deepest cuts | Low |
| Master problem bloat | GPU $k$-medoids cut filtering | Low |
| Oscillation / slow convergence | Level method stabilization (dense QP) | Moderate — parameter tuning |
| Scenario load imbalance | Asynchronous/partial-update Benders | Moderate — race-condition complexity in L4 callbacks |

### 4.2A.2 Dantzig-Wolfe decomposition / column generation

The RMP (dimensionally small but dense and degenerate) is solved on the GPU via **cuDSS pivot-free factorization**, not PDHG — PDHG's slow tail convergence makes it a poor fit for small, dense, degenerate matrices. Pricing subproblems are batched across GPU streams: homogeneous blocks use standard CSR; heterogeneous blocks are padded into **VBCSR** to keep warp execution uniform.

**Asynchronous column generation** eliminates the synchronous master/subproblem round-trip: as soon as any subset of streams returns negative-reduced-cost columns, they're appended to the RMP, the RMP is re-solved via cuDSS, and updated duals are broadcast to the next available streams — no global barrier.

**Dual stabilization** against the tailing-off effect uses **Wentges smoothing + in-out separation**:

$$\tilde\pi^t = \alpha\hat\pi + (1-\alpha)\pi^t$$

with $\alpha$ auto-adapted via a trust-region scheme computed entirely in cuBLAS — no CPU intervention.

**Throughput trick:** the L1 PDHG solver, applied to pricing subproblems, streams *intermediate* improving iterates as candidate columns rather than waiting for exact convergence — trading subproblem exactness for parallel column throughput.

**MILP extension (Branch-and-Price):** branching must occur on the original $x$ variables via **Ryan-Foster branching**, never on the fractional master $\lambda$ variables (which would destroy pricing-problem structure and block regeneration of previously selected columns). Localized bound constraints are added to subproblems without altering VBCSR layout, keeping the GPU-batched node-bounding machinery intact. This is the single highest-risk item in the whole decomposition layer — full GPU Branch-and-Price has no meaningful literature precedent.

| Failure mode | Mitigation | Residual risk |
|---|---|---|
| Tailing-off effect | Wentges dual smoothing + in-out separation | Low |
| Warp divergence in pricing problems | VBCSR padding / stream grouping by block size | Moderate — VRAM cost of padding |
| Synchronization overhead | Asynchronous column generation | Low |
| RMP memory bloat | Column aging / deletion of persistently non-improving columns | Moderate — risk of cycling (regenerating purged columns) |

### 4.2A.3 Progressive Hedging

The most naturally GPU-parallel of the three: it bypasses the L1.5 master/subproblem coordination entirely. The L0 substrate instantiates $|\Omega|$ independent solver instances directly in VRAM; each scenario is solved by the standard L1 engine (PDHG/IPM for continuous, batched B&B for MILP). Consensus is computed by pure GPU reduction (Thrust / cuBLAS) — zero CPU intervention, zero PCIe round-trip.

$$x_s^{(k+1)} = \arg\min_{x_s\in X_s}\Big(f_s(x_s) + w_s^{(k)\top}x_s + \tfrac{\rho}{2}\|x_s-\bar x^{(k)}\|^2\Big),\quad \bar x^{(k+1)}=\sum_s p_s x_s^{(k+1)},\quad w_s^{(k+1)}=w_s^{(k)}+\rho(x_s^{(k+1)}-\bar x^{(k+1)})$$

The quadratic penalty is handled natively by PDHG's proximal-mapping machinery at near-zero extra cost.

**For MILP, PH is explicitly a matheuristic, not an exact method** — integrality destroys the theoretical convergence guarantee. Mitigations: **adaptive element-wise $\rho_j$** (Watson & Woodruff — amplify the penalty on specific discrete variables that refuse to converge to consensus), and **scenario bundling** (enforce exact non-anticipativity within local bundles of, e.g., 100 scenarios, use PH only to reconcile between bundles) — this maps naturally to BSR format and balances load across SMs.

| Failure mode | Mitigation | Residual risk |
|---|---|---|
| No MILP convergence guarantee | Adaptive $\rho$ + variable fixing when >95% scenario agreement | **High — fundamentally heuristic; cannot guarantee optimality or a valid lower bound** |
| Quadratic objective overhead | PDHG's native proximal-mapping QP support | Low |
| Oscillation / slamming | Scenario bundling as stabilization anchors | Moderate — VRAM cost of larger bundle subproblems |

### 4.2A.4 Shared infrastructure and warm-starting

All three modes reuse the same batched-relaxation machinery already planned for L4's GPU node-bounding: **"solving many small related LPs" is implemented exactly once in the codebase.** Between iterations, only objectives (reduced costs) or right-hand sides (fixings) change — the L1.5 layer overwrites these in place via `cudaMemcpy` without rebuilding CSR/VBCSR layouts, letting L1 warm-start from the previous primal/dual iterate.

**Studied-not-linked references** (sovereignty-compliant — read for architecture, never forked): **GCG**'s hypergraph-based automatic block detection (must be reimplemented as a GPU-parallel operation, since GCG's is sequential CPU); **mpi-sppy**'s adaptive $\rho$ initialization via a scenario-diversity pre-computation step; **PIPS-IPM++**'s Schur-complement decomposition, which validates using cuDSS to factor the dense linking-variable system efficiently.

## 4.3 L2 — Crossover / vertex identification *(unchanged from prior revision)*

Concurrent checkpoint crossover (launch crossover threads from intermediate PDHG iterates at multiple tolerance checkpoints, overlapping CPU finishing work with ongoing GPU iteration) plus the spiral-axis vertex-jump research bet (using PDHG's documented rotation/forward-motion spiral geometry to project analytically toward the vertex it's spiraling into, bypassing sequential pivots).

## 4.4 L3 — Presolve (GPU-native, lightweight, dual-preserving)

Singleton rows/columns, forcing/dominated constraints, basic bound tightening, coefficient/scaling normalization — engineered to run on GPU or tight parallel C, capturing ~90% of a full commercial presolve's reductions. **Newly folded in this revision:** L3 also performs the **multi-level hypergraph structure detection** that drives L1.5's dispatch decision (§4.2A) — this must itself be a GPU-parallelized operation (unlike GCG's sequential CPU version) to avoid becoming the very Amdahl's-Law bottleneck presolve exists to prevent.

## 4.5 L4 — Mixed-integer engine (heterogeneous branch-and-cut) *(substantially expanded this revision)*

The CPU owns the tree — node selection, branching, cut pool, conflict analysis. The GPU is a bounding-and-heuristic firehose, streaming LP relaxation bounds and candidate solutions into a shared solution pool (CHAP-style), against which fix-and-propagate heuristics and neural diving run. Cutting planes (Gomory, MIR, cover) are generated and scored via a pluggable selector. **GPU-batched node bounding** solves many sibling/cousin nodes' LP relaxations simultaneously as a batched operation — same PDHG kernel, differing only in bound vectors — sidestepping warp divergence at the node level.

This revision adds three CPU-side subsystems that sit **upstream of the GPU batch as a filtration membrane**, ensuring every LP admitted to the batch carries genuinely novel bounding information.

### 4.5.1 Symmetry detection and orbital branching

**Bipartite graph construction.** The MIP is transformed into a colored bipartite graph: variable nodes colored by objective coefficient + bounds + type; constraint nodes colored by sense + RHS; edges for nonzero coefficients. Since standard automorphism algorithms don't support edge colors, numeric coefficients are encoded as auxiliary colored chain-nodes inserted along the edge. This construction is a pure CPU-side, allocation-conscious preprocessing step, run once per instance (or once per node when maintaining symmetry incrementally — see below).

**Individualization-Refinement (IR), reimplemented from scratch** (no nauty/bliss/saucy, per the sovereignty constraint): an **equitable-partition color-refinement loop** — a partition is equitable if every pair of same-cell vertices has identical connection counts to every other cell; cells that fail this are fractured by degree count. When the partition isn't discrete, the algorithm **individualizes**: isolates one vertex from a multi-vertex cell into its own singleton color, branches in a depth-first IR search tree, and re-refines. The permutations discovered this way generate the graph's automorphism group. Requires allocation-free structures (doubly-linked lists + contiguous cell-boundary arrays) to hit production performance.

**Schreier-Sims / Base-and-Strong-Generating-Set (BSGS).** Highly symmetric MIPs (set-partitioning, unit-commitment) can have astronomically large symmetry groups, impossible to enumerate. Schreier-Sims represents the group compactly via a **stabilizer-subgroup chain** plus **Schreier vectors** (orbit trees enabling fast coset-representative lookup). A **Sims filter** discards redundant generators to guarantee a polynomial memory bound; **randomized Schreier-Sims** (Monte Carlo variant) scales further at a small risk of missing obscure generators if tuned too aggressively.

**Orbital branching:** instead of a single-variable disjunction, branch on a representative variable from a fractional **orbit** under the node's stabilizer subgroup — the right branch fixes the representative to 1, the left branch fixes the **entire orbit** to 0 at once. **Orbitopal fixing:** for partitioning-structured constraints, a linear-time DP compares current fixings against the orbitope's boundary (the convex hull of binary matrices with lexicographically non-increasing columns) and propagates forced fixings.

**GPU interaction — isomorphism pruning as a pre-filter.** Before a sibling/cousin node is admitted to the GPU batch, an orbit query against the BSGS checks whether it's symmetric to an already-bounded or already-queued node. If so, it's pruned on the CPU for free — the GPU never wastes a slot on a provably redundant LP. **Dynamic maintenance is cheap:** a child node's symmetry group is the stabilizer of the branched variable within the parent's group, computed incrementally from existing Schreier vectors (basic array lookups) without re-invoking IR.

| Failure mode | Mitigation | Residual risk |
|---|---|---|
| Detection overhead on asymmetric instances | Cheap degree/hash fingerprinting aborts full IR when no symmetry is plausible | Low |
| Schreier-Sims memory explosion | Sims filter + randomized Schreier-Sims | Moderate — aggressive tuning can miss generators |
| Dynamic symmetry maintenance cost | Stabilizer subgroup descent via existing Schreier vectors, no IR re-invocation | Low |
| Presolve symmetry destruction (L3 reductions break structural symmetry) | Delay aggressive symmetry-breaking presolve until after initial IR detection; restrict variable aggregation to within-orbit only | Moderate — requires tight L3/L4 coupling |

### 4.5.2 Cut-based conflict-driven learning (CDCL for MIP)

Not SAT's classical implication-graph clause learning (too weak once continuous relaxations are involved) — **cut-based conflict analysis**, reasoning directly over the LP's linear inequalities via Chvátal-Gomory/MIR rounding, following the modern shift in pseudo-Boolean and cutting-edge MIP literature.

**Trigger:** fires when a GPU-batched LP relaxation bounds a node — infeasible, or objective exceeds the incumbent cutoff.

**Derivation:** on infeasibility, extract the **Farkas dual ray** $\lambda$ from the L1 solve, forming the proof-constraint $\sum_i \lambda_i(A_ix) \geq \sum_i \lambda_i b_i$. This raw constraint is dense and locally specific; the algorithm reduces it by iteratively combining it with the "reason constraint" for the most recent bound propagation, eliminating that variable, and applying **MIR/fractional saturation** at each step. The backward derivation terminates at a **1UIP-equivalent condition** (a single variable at the current decision level remains), yielding a globally valid no-good cut for the L4 cut pool.

**Pool management:** **Literal Block Distance (LBD)** scoring — cuts spanning fewer distinct decision levels are more generalized and retained; high-LBD cuts decay and are purged. **Propagation-only classification:** dense learned cuts are excluded from the GPU's LP matrix entirely and evaluated as CPU-side sparse dot products for bound tightening only — protecting simplex/barrier throughput from cut bloat.

**Pre-filter role:** before a node is queued for the GPU batch, a trivial sparse-dot-product check against the active conflict database can discard it outright — another free CPU-side save of GPU throughput.

| Failure mode | Mitigation | Residual risk |
|---|---|---|
| LP slowdown from dense cuts | "Propagation-only" classification — evaluated on CPU, excluded from the GPU LP matrix | Moderate — balancing cut strength vs. LP sparsity needs adaptive tuning |
| Numerical instability in cut derivation | Exact/rational arithmetic for the backward derivation, or strict epsilon safeguards before MIR rounding | **High — one of the hardest failure modes to diagnose in continuous-discrete solvers** |
| Conflict-database explosion | Aggressive LBD-weighted aging, periodic purge at deterministic restarts | Low |
| Weak Farkas rays (overly dense, poorly generalizing) | Ray sparsification — minimize the $L_1$ norm of the ray's multipliers via a secondary lightweight LP | Moderate — adds latency to the derivation pipeline |

### 4.5.3 Deterministic parallel / work-stealing scheduling

**Chase-Lev lock-free work-stealing deques**, one per CPU worker. Workers pop/push locally from the **bottom** (cache-local, natural depth-first exploration); idle workers steal from the **top** of a random victim's deque, which captures older, higher-level nodes governing larger unexplored subtrees — maximizing work transferred per steal and minimizing steal frequency.

**Structural determinism** (a strict requirement: identical search path, node count, and solution regardless of core count or hardware) is achieved via three mechanisms:
- **Node-count stamps** replace wall-clock time as the solver's internal clock.
- **Trailing read barriers:** when a worker discovers a new incumbent or conflict cut, it timestamps the discovery with the current global node-count; other workers are barred from applying it until their own local counter passes that stamp — guaranteeing reproducible information-propagation timing regardless of scheduling jitter.
- **Deterministic tie-breaking:** all tie-breaks use pseudo-random seeds hashed from node lineage (path from root), never from thread ID or memory address.

**GPU batching under determinism:** dispatch fires on a **deterministic node-count/queue-size threshold** (e.g. $N_{batch}=4096$), never a wall-clock timeout; bounded results scatter back to workers by deterministic node ID.

| Failure mode | Mitigation | Residual risk |
|---|---|---|
| Deterministic overhead & latency (barriers delay bound propagation, inflating node count vs. a non-deterministic solver) | Task-queue oversubscription — far more logical tasks than physical threads, so a blocked worker switches to independent work | **High — a structural, accepted cost of determinism, not a bug** |
| GPU batch starvation (heavy CPU-side filtering slows queue fill rate) | Deterministic, depth-based adaptive batch triggers rather than static volume thresholds | Moderate — the most sensitive tuning parameter in L4 |
| Memory contention on shared conflict pool / incumbent | Thread-local read-mostly caching, deferred batched updates at deterministic sync boundaries | Low |

### 4.5.4 Cross-cutting synthesis within L4

**Symmetry-aware conflict generalization:** when a conflict cut is derived at a node with an active BSGS, permuting its variables across the known orbits instantly generates a whole family of symmetric no-good cuts — without re-running Farkas extraction or MIR reduction per cut. This specific synergy is sparse in classical MIP literature but standard in SAT/CP, and is a genuine differentiation point here.

**Quantified benefit:** aggressive conflict analysis plus symmetry pruning together are documented to cut total B&B node count by **up to 40–50% on degenerate MIPLIB-2017-class benchmarks** — directly targeting the exact hard-instance class (extreme degeneracy, massive symmetry) the PS itself names as the grading bar.

**Recommended L4 pipeline:** L3 presolve handoff → symmetry init (bipartite graph + IR + Schreier-Sims, once, CPU) → parallel dispatch (Chase-Lev deques per worker) → worker loop (pop/steal → incremental stabilizer descent → conflict-pool + orbitopal-fixing pre-filter → push survivors to the global GPU batch queue) → GPU sync (deterministic threshold dispatch) → conflict derivation (Farkas extraction, MIR backward derivation, symmetry-permutation generalization, trailing-barrier-respecting insertion into the global pool).

**Architectural principle:** symmetry handling and conflict learning together form a CPU-side filtration membrane that guarantees every LP reaching the GPU batch carries genuinely new bounding information — this is the real lever that compounds the batched-bounding architecture's throughput beyond raw parallelism alone.

## 4.6 L5 — ML / RL augmentation (hot-swappable) *(unchanged)*

Everything above works without ML. ML is a set of plugins that replace default heuristics, so the solver degrades gracefully and stays inspectable.

## 4.7 L6 — Interface layer *(unchanged)*

MPS/LP parsers, a C-API, Python bindings API-compatible with Pyomo/PuLP/CVXPY, a CLI, and optionally a REST/JSON microservice endpoint.

---

# PART V — The frontier bets

## 5.1 Bet 1 — Dual-engine continuous core (PDHG + pivoting-free IPM), not simplex

Non-simplex methods as the core, concurrent racing across the size/conditioning spectrum.

## 5.2 Bet 2 — Solve the crossover bottleneck (concurrent + spiral-axis)

The make-or-break for using GPU LP inside MILP.

## 5.3 Bet 3 — GPU-native lightweight presolve

Presolve that runs where the solve runs, so it never becomes the Amdahl bottleneck.

## 5.4 Bet 4 — End-to-end RL for the tree, not imitation learning

Reward the actual objective (global tree size), sidestepping imitation learning's proven perturbation instability.

## 5.5 Bet 5 — Heterogeneous pool-based primal heuristics

GPU streams relaxations into a shared pool; CPU runs fix-and-propagate and local search against it.

## 5.6 Bet 6 — Structure-adaptive decomposition, racing the monolith *(new)*

L1.5 detects block-angular/scenario structure during presolve and races Benders, Dantzig-Wolfe, and Progressive Hedging against the monolithic L1 solve — prioritized by the research's own risk assessment: **Benders first** (lowest risk, highest ROI, and it directly targets the PS-named power-dispatch/stochastic domains via GPU-scenario-parallel subproblem evaluation with Magnanti-Wong-Papadakos deepest cuts and $k$-medoids cut filtering); **Dantzig-Wolfe LP mode second**; full **GPU Branch-and-Price** as an explicit stretch goal (the literature is silent on it — first-mover territory if it works); **Progressive Hedging** kept as a matheuristic complement for very-large-scenario-count stochastic MILP where exactness isn't required. No open-source solver races structure-adaptive GPU decomposition against a monolithic GPU solve as standard behavior.

## 5.7 Bet 7 — Sovereign symmetry + conflict-learning engine as a compounding CPU-side filter *(new)*

A from-scratch Individualization-Refinement + Schreier-Sims symmetry engine and cut-based conflict-driven learning (Farkas-ray/MIR no-goods, LBD-managed) sit upstream of the GPU batch as a filtration membrane. Symmetry-aware conflict generalization compounds the two together. This directly targets the PS's own named hard-instance class (MIPLIB "hard/open": extreme degeneracy, massive symmetry) with a documented 40–50% node-count reduction on degenerate benchmark sets — a concrete, gradable win. Paired with deterministic Chase-Lev scheduling, it also gives bit-for-bit reproducible search across hardware, matching the structural-determinism guarantee of commercial solvers like FICO Xpress — something no open-source solver currently offers under full parallelism.

---

# PART VI — Numerical robustness (what the PS actually grades)

## 6.1 Degeneracy and cycling *(unchanged)*

Regularization in the IPM, anti-cycling tie-breaking in the simplex fallback, adaptive restarts in PDHG.

## 6.2 Conditioning and preconditioning *(unchanged)*

Ruiz + Pock–Chambolle mandatory; condition estimates route worst-conditioned blocks to fp64-only paths.

## 6.3 Mixed-precision stability *(unchanged)*

Iterative refinement plus conditioning-gated fp32/fp64 fallback.

## 6.4 Presolve as robustness *(unchanged)*

Dual-preserving presolve reduces conditioning burden before the first iteration.

## 6.5 Decomposition-specific numerical risks *(new)*

- **Dual oscillation / tailing-off** in column generation — mitigated by Wentges smoothing + in-out separation (GPU-resident, cuBLAS-driven).
- **Weak/degenerate Benders cuts** — mitigated by Magnanti-Wong-Papadakos deepest-cut generation.
- **cuDSS pivot-free stability on decomposition master/RMP matrices is explicitly uncertain.** cuDSS is new and trades away standard pivoting for speed; its behavior on the notoriously ill-conditioned master matrices these decomposition schemes generate is unproven. Treated as a Tier-1 risk: we need either an internal dense-factorization fallback or the same SQD-style regularization used in §4.2 Engine B before trusting cuDSS on decomposition masters. **The source research is explicit that if cuDSS fails here, the solver currently lacks a dense factorization fallback — this is an open build item, not a solved problem.**
- **PDHG's ~1e-4 native precision is a real risk for column-generation reduced costs** — imprecise duals can stall the master. Whether moderate-precision PDHG subproblem solves give accurate-enough dual information is explicitly unproven; tracked in Part X rather than asserted as solved.

## 6.6 Conflict-derivation numerical stability *(new)*

Deep, recursive MIR/CG rounding sequences amplify floating-point drift — the single highest-flagged residual risk anywhere in the L4 conflict-learning subsystem. Policy: run the backward Farkas/MIR derivation on the CPU with either rational/exact arithmetic or strict epsilon-relaxation safeguards, and never inject a conflict cut into the pool without a validity re-check against the original constraint system.

## 6.7 Determinism vs. speed tradeoff *(new)*

Trailing read barriers are a structural cost, not a bug: a deterministic solver will always explore somewhat more nodes on average than one that applies new bounds instantly. We accept this deliberately — it buys reproducibility, debuggability, and parity with commercial-grade determinism guarantees — and mitigate the *latency*, never the guarantee, via task-queue oversubscription.

---

# PART VII — Benchmarking and validation

**Correctness oracles.** Every solved instance's answer is checked against an independent, permissively-licensed solver run strictly as a black-box CLI (a testing oracle, not a dependency) and against our own from-scratch simplex.

**Standard corpora, wired into CI/CD:** Netlib LP, MIPLIB 2017 (now specifically including its high-symmetry, high-degeneracy instances as the direct test of Bet 7), Mittelmann tracks, and MILPBench (100k instances / 60 classes, for RL training and asymptotic scaling studies). **Decomposition-specific additions:** stochastic-programming test libraries (scenario-structured instances for Benders/PH) and block-angular MIP sets for Dantzig-Wolfe LP mode.

**Metrics beyond wall-clock:** root-node gap, nodes explored, primal/dual integral — and now also **node-count reduction attributable to symmetry/conflict pruning** as an explicit, separately-tracked metric, since that reduction is the direct evidence for Bet 7.

**The PS bar.** Match or beat at least one established solver on these sets, and demonstrate reliable convergence on a curated degenerate/ill-conditioned suite. The concurrent multi-engine, multi-decomposition-mode design is aimed squarely at clearing this bar across problem *classes*, not on a lucky subset.

---

# PART VIII — How this differs from the current open-source market

| Dimension | HiGHS | SCIP / SoPlex | COIN-OR (CBC/Clp/Ipopt) | OR-Tools | NVIDIA cuOpt | **This project** |
|---|---|---|---|---|---|---|
| Primary substrate | CPU | CPU | CPU | CPU (PDLP proto GPU) | **GPU** | **GPU-native, CPU for logic** |
| LP core | dual simplex / IPM | SoPlex simplex | Clp simplex | GLOP simplex / PDLP | PDLP | **dual engine: PDHG + pivoting-free IPM, concurrent** |
| Exact MILP tree | CPU B&C | CPU B&C (strong) | CPU B&C | CP-SAT | beta, heuristic | **CPU tree + GPU-batched node bounding** |
| Crossover to vertex | classical | classical | classical | classical | classical/CPU | **concurrent checkpoint + spiral-axis jump** |
| Presolve | CPU | CPU (PaPILO) | CPU | CPU | PSLP (GPU) | **GPU-native, dual-preserving, + hypergraph structure detection** |
| Structure exploitation (decomposition) | none native | plugin-based (GCG, separate tool) | Cbc/DIP exists, CPU-only | none native | none native | **GPU-native, structure-adaptive (Benders/DW/PH), races the monolithic solve** |
| Symmetry handling | none native | symmetry breaking via linked external tools (e.g. bliss) | none native | none native | none | **from-scratch IR + Schreier-Sims, GPU-batch isomorphism pre-filter** |
| Conflict learning | none | classical conflict analysis (CPU) | none | none | none | **cut-based Farkas/MIR CDCL, LBD-managed, symmetry-generalized** |
| Deterministic parallelism | n/a (mostly serial) | partial | n/a | n/a | n/a | **Chase-Lev deques + trailing barriers, bit-reproducible across hardware** |
| Learning for search | none | none | none | none | GPU heuristics | **end-to-end RL on global tree size (not imitation)** |
| Hardware portability | CPU | CPU | CPU | CPU | **CUDA only** | **CUDA ↔ ROCm abstraction** |
| Inspectable / sovereign | MIT, yes | Apache, yes | EPL | Apache | Apache, NVIDIA-controlled roadmap | **fully owned, from mathematical foundation** |
| SIMD/vectorization | **absent (documented)** | partial | partial | partial | GPU-parallel | **hardware-aware by design** |

**Seven things no shipping open-source solver combines**, each mapped to a named open problem:

1. A **dual continuous engine** (PDHG + pivoting-free IPM), both GPU-native, racing concurrently.
2. A **crossover that doesn't kill the speedup** — concurrent checkpointing plus the spiral-axis vertex jump.
3. **Presolve that runs where the solve runs**, with GPU-native hypergraph structure detection built in.
4. **RL that optimizes the actual objective** (global tree size), not imitation.
5. **Heterogeneous pool-based primal heuristics.**
6. **Structure-adaptive decomposition** (Benders/Dantzig-Wolfe/Progressive Hedging) racing the monolithic GPU solve as standard behavior.
7. **A sovereign symmetry + conflict-learning engine** as a compounding CPU-side filter, with deterministic parallel scheduling for bit-reproducible search.

---

# PART IX — A logical build sequence

**Stage 1 — Substrate + a correct LP.** L0 + MPS/LP parser + from-scratch CPU revised simplex. Goal: solve Netlib correctly; this is the correctness oracle.

**Stage 2 — GPU PDHG.** PDLP-style PDHG with Ruiz + Pock–Chambolle preconditioning and adaptive restarts. Goal: match cuOpt's feasible-point behavior on Large-Network-LP.

**Stage 3 — Pivoting-free IPM.** Regularize KKT → SQD, factor with cuDSS, add mixed-precision iterative refinement. Goal: high-precision LP/QP on GPU, racing Stage 1 and Stage 2.

**Stage 4 — Crossover.** Concurrent checkpoint crossover first (known-good), then the spiral-axis experiment. Goal: exact vertices without erasing the GPU speedup.

**Stage 5 — Presolve + structure detection.** Lightweight dual-preserving presolve, plus GPU-parallelized hypergraph partitioning for block-angular/scenario detection. Goal: ~90% of commercial presolve reductions, and a working structure classifier feeding L1.5's dispatch logic.

**Stage 6 — Decomposition layer (L1.5).** Build in priority order: **Benders/integer L-shaped first** (MWP deepest cuts, $k$-medoids cut filtering, level-method stabilization) — goal: beat the monolithic solve on a stochastic unit-commitment or capacity-expansion test set. **Dantzig-Wolfe LP mode second** (cuDSS RMP, async column generation, Wentges smoothing) — goal: beat the monolithic solve on block-angular MIPLIB-adjacent instances. **Progressive Hedging** as a lower-priority complement. **Branch-and-Price (MILP DW)** deferred as an explicit stretch goal.

**Stage 7 — MILP engine core.** CPU branch-and-cut tree, cut pool, shared solution pool, fix-and-propagate heuristics, GPU-batched node bounding. Goal: solve MIPLIB 2017 instances; beat one open-source MILP solver on a subset.

**Stage 7b — Symmetry engine.** Bipartite graph construction, from-scratch IR, Schreier-Sims/BSGS, orbital branching, orbitopal fixing, isomorphism pre-filtering into the GPU batch. Goal: measurable node-count reduction on MIPLIB's high-symmetry instances.

**Stage 7c — Conflict-driven learning.** Farkas-ray extraction, backward MIR derivation, LBD pool management, propagation-only cut classification. Goal: measurable node-count reduction on degenerate MIPLIB instances; validated numerical stability under deep derivation chains.

**Stage 7d — Deterministic scheduling.** Chase-Lev work-stealing deques, trailing read barriers, deterministic GPU batch dispatch. Goal: bit-identical results across different core counts/hardware on the full Stage 7–7c test suite.

**Stage 8 — RL/GNN plugins.** GNN encoder + end-to-end RL branching/node policy + learned cut selection, trained on MILPBench. Goal: measurable tree-size reduction vs. the Stage 7 default heuristics, tested out-of-distribution.

**Stage 9 — Interfaces + domain hardening.** C-API, Pyomo/PuLP/CVXPY compatibility, REST endpoint, curated industrial suite (refinery blending, unit commitment, VRP) proving robustness on degenerate, ill-conditioned, real-world models — now including stochastic/decomposable formulations as first-class test cases given Stage 6.

---

# PART X — Open-questions register

**Core continuous/crossover/presolve (original):**
1. Spiral-axis crossover may not generalize beyond well-behaved LPs. Fallback: concurrent checkpoint crossover.
2. GPU-batched node bounding helps the bounding step; full tree-on-GPU remains unsolved.
3. Mixed-precision divergence on the nastiest industrial matrices is real; the conditioning-gated fallback trades speed for correctness deliberately.
4. RL generalization risk — the gate is out-of-distribution performance vs. the default heuristic, tested explicitly in Stage 8.
5. Pivoting-free factorization relies on regularization keeping the KKT well-behaved; pathological systems may still force pivoting on specific blocks.
6. Sovereignty vs. cuDSS — true hardware sovereignty depends on ROCm reaching parity; until then, the fast path is NVIDIA-first with a CPU/ROCm correctness fallback.

**Decomposition layer (new):**
7. **GPU-native Branch-and-Price feasibility is genuinely unknown.** Batching identical scenario subproblems is trivial; batching *heterogeneous* block-angular pricing problems dynamically across an active B&B tree is an unsolved warp-divergence and dynamic-memory-allocation problem. Whether VBCSR formatting offers enough throughput to offset this complexity is unproven.
8. **Adaptive memory budgets for cut/column pools.** Precise thresholds for triggering $k$-medoids cut filtering vs. blind column purging are empirical and unvalidated at industrial scale.
9. **PDHG's moderate precision may not support accurate-enough reduced costs for column generation** — unproven whether this stalls the RMP in practice.
10. **cuDSS pivot-free stability on decomposition master matrices is uncertain**, and the solver currently has no dense-factorization fallback if it fails — a named, explicit gap.
11. **GPU-parallel hypergraph partitioning overhead.** Automatic block-structure detection is NP-hard and historically CPU-heuristic-solved; whether a parallelized version inside L3 presolve avoids wiping out the decomposition speedup is a major algorithmic unknown.

**Symmetry / conflict / determinism (new):**
12. **Cut-based CDCL over general continuous variables is experimental.** The theory is well-established for pure 0-1 pseudo-Boolean programs; extension to general-integer and unbounded continuous variables — specifically preventing infinite fractional-rounding loops — lacks deep empirical coverage.
13. **The GPU-aware conflict-checking crossover point is unquantified.** At what conflict-pool size does CPU-side checking start starving the GPU batch queue? No literature precedent exists for batch-checking a large dynamic conflict pool on GPU itself.
14. **Balancing variable pruning aggressiveness against GPU utilization** under highly volatile per-node CPU processing times, while respecting deterministic node-stamp barriers, is a scheduling challenge absent from current academic literature.
15. **The boundary between L3 continuous presolve and L4 symmetry preservation is vaguely defined** — how deep coefficient tightening can safely go before it destroys the algebraic permutations orbital branching depends on is not established in the literature; presolve and symmetry detection are generally treated as isolated phases rather than a continuous feedback loop.

None of these sink the project. They are the frontier — which is exactly where the differentiation lives.

---

# PART XI — Potential research areas (not yet integrated)

The two research passes folded into Parts IV–X above (decomposition methods; symmetry/conflict/determinism) came from dedicated deep-research efforts. The items below were flagged earlier as high-value directions but have **not** yet received the same depth of treatment — they are carried forward here as an open research queue, not committed architecture. Each note states why the direction could plausibly move the bible's results further.

## XI.A — Acceleration for the continuous core

**1. Anderson acceleration / Nesterov-style momentum for PDHG.**
Beyond adaptive restarts, extrapolation techniques from fixed-point iteration theory (Anderson mixing) may cut PDHG iteration counts substantially further. *Potential benefit:* directly attacks PDHG's documented weak point — slow tail convergence — independent of and complementary to crossover; if it works, it shrinks how often crossover is even needed, because usable precision arrives faster.

**2. Randomized / sketching-based preconditioner construction.**
Randomized numerical linear algebra (sketching) can estimate conditioning and build preconditioners cheaply for very large sparse systems where exact conditioning estimates are too costly. *Potential benefit:* a two-for-one improvement — better mixed-precision fallback decisions (§6.3) and faster PDHG convergence, from the same technique.

## XI.B — Extending reach toward MINLP/QP domain strength

**3. GPU-parallel automatic differentiation for the future MINLP/spatial-B&B layer.**
Reverse-mode AD on GPU (as used in ML frameworks) could compute gradients/Hessians for nonconvex terms (bilinear pooling-problem terms, refinery blending) fast enough to make spatial branch-and-bound GPU-viable later. *Potential benefit:* genuinely forward-looking — almost nobody has combined GPU-native AD with spatial B&B for MINLP, consistent with the bible's "unsolved frontier" positioning for the eventual MINLP extension named in the PS's modular-growth requirement.

**4. McCormick envelope tightening via GPU-parallel bound propagation.**
For the pooling-problem/blending nonconvexity the PS names explicitly, tighter relaxations come from iteratively tightening McCormick envelopes — parallelizable across many bilinear terms simultaneously on GPU. *Potential benefit:* a concrete, domain-specific technique demonstrating strength on crude blending specifically, one of the PS's named target applications, ahead of full MINLP support.

## XI.C — Systems-level scaling

**5. Multi-GPU model-parallel solving.**
Partitioning a single massive LP/MILP across multiple GPUs (not just multiple scenarios/decomposition subproblems) via graph/hypergraph partitioning of the constraint matrix, minimizing inter-GPU communication. *Potential benefit:* turns "fast on one GPU" into "scales past single-GPU VRAM limits entirely" — a natural extension of the o9 Solutions precedent already cited in Part II, and complementary to (not redundant with) the L1.5 decomposition layer, since it applies even to monolithic matrices that don't decompose structurally.

**6. Portfolio / algorithm-selection ML (predict-then-solve).**
Rather than always racing every engine (PDHG, IPM, simplex, and now every applicable decomposition mode) concurrently, a lightweight classifier on problem features could predict which engine/mode is likely to win and skip the rest. *Potential benefit:* pure efficiency gain — same concurrency-based robustness story, less wasted GPU-hours per solve, faster production wall-clock without sacrificing the "race everything" robustness guarantee that underlies Bets 1 and 6.

---

*End of bible. Parts VIII and XI are the two most likely sections to be read on their own — the differentiation table for the pitch, the research queue for what's next.*
