# Benchmark oracle

Build Map ticket #2. The **independent answer** every future ticket's result is
checked against (Oracle rule; Bible §1.2, Part VII).

## The sovereignty position

The default oracle is **HiGHS**, which is *forbidden as a library* in
`sovereignty.toml`. That is not a contradiction: the ban is on **linking** it.
Running its command-line executable as a subprocess over files on disk is the
sanctioned use. See `DEPENDENCY_LEDGER.md` §5.5.

The boundary is enforced physically:

- `install_highs.sh` builds the HiGHS CLI into `build-oracle/` — gitignored and
  in `sovereignty.toml`'s `exclude_globs`, invisible to the sovereignty scan and
  to the solver's CMake project.
- `run_oracle.py` only ever `subprocess.run`s the binary and parses its output
  files. It never imports a solver. If it cannot spawn one it returns status
  `error` — it does **not** fall back to an in-process library. Its JSON output
  carries `backend.linked = false` as an asserted invariant.
- Delete `build-oracle/` and the sovereignty check plus every solver test still
  pass. That is the proof it is scaffolding, not a dependency.

## Usage

```sh
# build the oracle (one time, ~2 min)
tools/oracle/install_highs.sh              # or: cmake --build build --target oracle

# solve one model, normalized JSON to stdout
tools/oracle/run_oracle.py benchmarks/data/netlib_lp/afiro.mps

# with primal/dual vectors, written to a file
tools/oracle/run_oracle.py model.mps.gz --with-solution --out result.json

# check the oracle against Netlib's published reference objectives
tools/oracle/check_reference.py            # a fixed sample
tools/oracle/check_reference.py --all      # every downloaded instance with a ref
```

Use a different solver: `SOLVER_ORACLE=/path/to/solver tools/oracle/run_oracle.py …`.
Adapters exist for `highs` and `cbc`; `glpsol`/`scip` are recognized as names
but need an adapter written.

## Normalized result schema (`oracle_schema = 1`)

```json
{
  "oracle_schema": 1,
  "instance": "afiro",
  "model_sha256": "…",
  "backend": { "name": "highs", "binary": "…", "version": "…",
               "linked": false, "argv": [...], "returncode": 0 },
  "invocation": { "threads": 1, "time_limit_s": null },
  "status": "optimal",              // optimal | infeasible | unbounded |
                                    // infeasible_or_unbounded | time_limit |
                                    // iteration_limit | error
  "objective": -464.7531428571,     // null unless status has a point in hand
  "solve_seconds": 0.01,
  "primal_present": true, "dual_present": true,
  "primal": { "X01": 80.0, ... },   // only with --with-solution
  "dual":   { "X01": 0.0,  ... }
}
```

`threads` defaults to **1** so reference answers are reproducible.

## HiGHS version notes

Pinned to `v1.11.0` in `install_highs.sh`. That CLI has **no** `--threads` or
`--write_solution_style` flags — both go through an options file, which
`run_oracle.py` writes. `--parallel` and `--time_limit` are real CLI flags.
Solution parsing targets `write_solution_style = 1` (the labelled columnar
table); the parser tolerates version drift by skipping rows it can't read.
