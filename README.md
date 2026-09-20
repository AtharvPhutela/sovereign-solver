# Sovereign Solver

A from-scratch, GPU-ready optimization solver core for Linear Programming (LP),
with a roadmap towards Mixed-Integer (MILP) and Quadratic (QP) programming.
Built for the Smart India Hackathon problem statement
*"Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to
Express / CPLEX)"*.

No existing solver library is used as a foundation. Every algorithm that
actually solves a problem is written here, from the mathematics up.

## Team

| Name | Role |
|------|------|
| _TODO_ | _TODO_ |
| _TODO_ | _TODO_ |
| _TODO_ | _TODO_ |

**Team name:** _TODO_  
**Institution:** _TODO_  
**Problem statement ID:** _TODO_

## Status

| Area | State |
|------|-------|
| MPS / LP file readers | Done |
| Revised simplex (sparse, from scratch) | Done |
| Scaling / preconditioning (Ruiz, Pock-Chambolle) | Done |
| Primal-dual hybrid gradient (PDHG / PDLP-style) | Done |
| Regularized interior-point method | Done |
| Farkas / dual-ray infeasibility and unboundedness certificates | Done |
| Concurrent engine race (simplex / PDHG / IPM) | Done |
| Checkpoint crossover (interior/first-order point to vertex) | Done |
| Spiral-axis vertex jump | Done (experimental) |
| Presolve with dual-preserving postsolve | Done (lightweight) |
| Hypergraph structure detection | Done |
| CUDA / ROCm backends | Written; not verified on real GPU hardware in CI |
| Branch-and-bound / cuts (MILP) | Not started |
| Quadratic programming (QP) | Not started |

_TODO: update this table before submission so it matches what you demo._

## Architecture

```
include/sovereign/   public headers
src/l0/              numerical substrate: sparse containers, numeric policy,
                     host / CUDA / HIP backend abstraction
src/l1/              algorithms: simplex, PDHG, IPM, crossover, presolve, race
src/io/              MPS and LP readers
apps/                sovereign-cli command-line front end
tests/               unit tests (C++) and oracle cross-checks (Python)
tools/               sovereignty checker, oracle comparison scripts
benchmarks/          corpus manifest, downloader, tiny smoke instances
```

Design documents live in [docs/](docs/): [design](docs/design.md) (architecture
and rationale), [design-simple](docs/design-simple.md) (plain-language overview),
[build-map](docs/build-map.md) (ticket-by-ticket plan) and
[problem-statement](docs/problem-statement.md).

## Building

Requirements: a C++20 compiler, CMake 3.24+, Python 3.8+ (tests and tools).
CUDA and ROCm are optional and auto-detected.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful options:

| Option | Default | Meaning |
|--------|---------|---------|
| `SOVEREIGN_CUDA` | `AUTO` | `AUTO`, `ON` or `OFF` for the CUDA backend |
| `SOVEREIGN_HIP` | `AUTO` | `AUTO`, `ON` or `OFF` for the ROCm/HIP backend |
| `SOVEREIGN_INDEX64` | `OFF` | 64-bit indices for instances beyond 2^31 nonzeros |

## Usage

```sh
build/apps/sovereign-cli info  model.mps                    # rows, columns, nonzeros
build/apps/sovereign-cli solve model.mps                    # revised simplex
build/apps/sovereign-cli solve --engine race model.mps      # simplex, PDHG and IPM concurrently
build/apps/sovereign-cli solve --engine crossover model.mps # first-order/IPM point, then vertex
build/apps/sovereign-cli presolve model.mps                 # presolve, solve, postsolve, verify
build/apps/sovereign-cli structure model.mps                # structure detection
```

Common flags: `--json`, `--time-limit S`, `--tolerance T`, `--verbose`.
Run `sovereign-cli` with no arguments for the full list.

Try it on the bundled instances: `benchmarks/smoke/*.mps`.

## Verification against a reference solver

Correctness is checked by comparing against HiGHS **used only as an external
black-box command-line program**. It is never linked, vendored or included in
the build; if it is absent, the solver and its tests still build and pass.

```sh
tools/oracle/install_highs.sh          # builds HiGHS into build-oracle/ (outside the build)
python3 benchmarks/fetch_corpus.py     # downloads Netlib etc. into benchmarks/data/
python3 tools/check_simplex_vs_oracle.py
```

See [benchmarks/README.md](benchmarks/README.md) and
[tools/oracle/README.md](tools/oracle/README.md).

## Sovereignty check

`tools/sovereignty_check.py` scans the source tree, build manifests and
dependencies for forbidden solver libraries (HiGHS, CBC, GLPK, SCIP, CPLEX,
Gurobi, and others) and fails the build if any appear. The policy is in
[sovereignty.toml](sovereignty.toml) and the reasoning in
[docs/dependency-ledger.md](docs/dependency-ledger.md).

## Benchmark results

_TODO: add a table of Netlib instances: rows/cols, objective, solve time,
and comparison against HiGHS._

| Instance | Rows x Cols | Our objective | HiGHS objective | Our time | HiGHS time |
|----------|-------------|---------------|-----------------|----------|------------|
| _TODO_ | | | | | |

## Limitations

- LP only for now; MILP and QP are on the roadmap, not implemented.
- GPU backends have not been validated on real hardware in CI.
- _TODO: add anything else you want to be upfront about._

## License

_TODO: not yet chosen._
