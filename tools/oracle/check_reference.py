#!/usr/bin/env python3
"""Ticket #2 pass-condition probe: solve real instances through the CLI oracle
and check the objective against the independent published reference value.

Exit 0 if every checked instance matches (or if nothing is available to check --
a fresh checkout has neither the oracle nor the corpus, and that is not a
failure). Exit 1 only on a genuine objective mismatch: the oracle disagreeing
with the Netlib reference means the oracle, the parser, or the corpus is wrong,
and nothing downstream should be trusted until it is resolved.

    check_reference.py                 # a default handful of instances
    check_reference.py --all           # every downloaded instance with a ref
    check_reference.py afiro sc50a ...  # named instances
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "tools" / "oracle"))
import run_oracle as ro  # noqa: E402

NETLIB = REPO / "benchmarks" / "data" / "netlib_lp"
REFERENCE = NETLIB / "_reference.toml"
DEFAULT_SET = ["afiro", "sc50a", "sc50b", "adlittle", "blend", "stocfor1",
               "degen2", "bandm", "share2b", "beaconfd"]
REL_TOL = 1e-5
ABS_TOL = 1e-4


def rel_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="oracle vs. Netlib reference (ticket #2)")
    ap.add_argument("instances", nargs="*", help="instance names (default: a fixed sample)")
    ap.add_argument("--all", action="store_true", help="check every downloaded instance with a reference")
    ap.add_argument("--time-limit", type=float, default=120)
    args = ap.parse_args(argv)

    name, exe = ro.resolve_backend(None)
    if exe is None:
        print("SKIP: no oracle solver (run tools/oracle/install_highs.sh). Not a failure.")
        return 0
    if not REFERENCE.is_file():
        print("SKIP: Netlib corpus not downloaded (run benchmarks/fetch_corpus.py). Not a failure.")
        return 0

    reference = tomllib.loads(REFERENCE.read_text())
    if args.all:
        names = sorted(k for k, v in reference.items()
                       if isinstance(v, dict) and "optimal_value" in v
                       and (NETLIB / f"{k}.mps").is_file())
    elif args.instances:
        names = args.instances
    else:
        names = [n for n in DEFAULT_SET if (NETLIB / f"{n}.mps").is_file()]

    if not names:
        print("SKIP: no instances with reference values available locally.")
        return 0

    print(f"oracle: {name} ({exe})")
    print(f"checking {len(names)} instance(s) against Netlib reference objectives\n")

    failures, checked = [], 0
    for n in names:
        mps = NETLIB / f"{n}.mps"
        ref = reference.get(n, {})
        if not mps.is_file() or "optimal_value" not in ref:
            print(f"  {n:<12} skip (no file or no reference)")
            continue
        res = ro.solve(mps, backend=None, threads=1, time_limit=args.time_limit,
                       with_solution=False)
        checked += 1
        want = float(ref["optimal_value"])
        got = res["objective"]
        if res["status"] != "optimal" or got is None:
            failures.append(n)
            print(f"  {n:<12} FAIL  status={res['status']} objective={got}")
        elif rel_close(got, want):
            print(f"  {n:<12} ok    {got:.8g}  (ref {want:.8g})")
        else:
            failures.append(n)
            print(f"  {n:<12} FAIL  {got:.10g} != ref {want:.10g}  (dev {abs(got-want):.3g})")

    print()
    if failures:
        print(f"FAIL: {len(failures)}/{checked} disagree with the reference: {', '.join(failures)}")
        return 1
    print(f"PASS: {checked}/{checked} match the Netlib reference to tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
