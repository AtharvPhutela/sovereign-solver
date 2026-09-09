# Benchmark corpora

Build Map ticket #2. What lives here, and how to get the data.

## Layout

| Path | In git? | What |
|---|---|---|
| `corpus.toml` | yes | Manifest: every test set, its source, licence, unpack method |
| `fetch_corpus.py` | yes | Downloader + verifier (stdlib only, no `requests`) |
| `smoke/` | **yes** | 4 hand-verified tiny instances + `expected.toml` reference answers |
| `data/` | no (gitignored) | Downloaded corpora land here |
| `.tools/` | no (gitignored) | Scratch: the compiled `emps` decoder, cached index files |

Only `smoke/` is committed, so CI and the first solver (#4) have something to
check against with **no network and no solver**.

## Usage

```sh
python3 benchmarks/fetch_corpus.py            # default sets (Netlib LP, ~40 MB)
python3 benchmarks/fetch_corpus.py --list     # sets + local status
python3 benchmarks/fetch_corpus.py --verify   # re-hash local files vs the lock
python3 benchmarks/fetch_corpus.py --set miplib2017    # a deferred set (stub for now)
python3 benchmarks/fetch_corpus.py --clean netlib_lp
```

Or via CMake: `cmake --build build --target corpus` / `--target corpus-verify`.

## Netlib LP

Fetched by default. Netlib stores each problem in a packed column format;
`fetch_corpus.py` builds `emps` (a ~250-line standalone C decompressor from
netlib.org, **no optimization logic** — classified in `sovereignty.toml`) and
expands each to MPS.

- **Reference objective values** are parsed from the Netlib `readme` "PROBLEM
  SUMMARY TABLE" (MINOS 5.3 values, plus the CPLEX cross-check values the readme
  records where they differ) into `data/netlib_lp/_reference.toml`. This is an
  **oracle-rule reference**: an independent published answer, not a value this
  project computed.
- `data/netlib_lp/_lock.toml` records the sha256 of every expanded instance.
- **Known gap:** a few large problems (`stocfor3`, `truss`, …) ship as shell
  archives rather than a raw packed file and are skipped for M0. ~90 of ~92
  instances are fetched.

## Deferred sets

`miplib2017` (+ its isolated `miplib2017_hard` subset — the M6 symmetry/
degeneracy evidence, kept separate on purpose), `mittelmann`, `milpbench`,
`qplib`. Manifest entries exist with URLs and licence notes; the fetchers are
stubs that print provenance. Implement each when the milestone that needs it
starts (MIPLIB → M5, Mittelmann → M5/M9, MILPBench → M7).

## The smoke set

| Instance | Status | Objective | For |
|---|---|---|---|
| `tiny_lp` | optimal | −10/3 | basic feasible bounded LP, unique fractional optimum |
| `tiny_infeasible` | infeasible | — | status handling |
| `tiny_unbounded` | unbounded | — | status handling |
| `tiny_degenerate` | optimal | −2 | primal-degenerate optimum (anti-cycling target for #4) |

Every value in `smoke/expected.toml` is derived by hand; the derivation is in
each `.mps` file's comment header.
