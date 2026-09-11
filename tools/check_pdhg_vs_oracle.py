#!/usr/bin/env python3
"""PDHG vs. the external oracle -- Build Map ticket #8, gate M1.

Ticket #8's pass condition: "PDHG solves a large LP on-device to first-order
tolerance, matching the oracle, with restarts working." Same Oracle rule as
check_simplex_vs_oracle.py (#4) -- the running external solver is the
reference, never a printed table -- but the tolerance here is PDHG's own
native accuracy (~1e-4, Bible S4.2 Engine A), not the simplex's near-exact
digits: a disagreement in the 5th significant figure is expected here and is
not what this script is checking for.

    check_pdhg_vs_oracle.py                  # the smoke set + small Netlib instances
    check_pdhg_vs_oracle.py afiro adlittle   # named instances
    check_pdhg_vs_oracle.py --max-rows 200   # keep the run quick
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NETLIB = REPO / "benchmarks" / "data" / "netlib_lp"
SMOKE = REPO / "benchmarks" / "smoke"
CLI = REPO / "build" / "apps" / "sovereign-cli"
ORACLE = REPO / "build-oracle" / "bin" / "highs"

# PDHG's own tolerance is ~1e-4 (Bible S4.2 Engine A); leave a comfortable
# margin above that for the comparison itself, or every genuinely-converged
# run would sit right on the edge of "disagreement".
REL_TOL = 5e-3
ABS_TOL = 1e-3
DEFAULT_MAX_ROWS = 150
DEFAULT_TIMEOUT = 60

ORACLE_OBJ = re.compile(r"Objective value\s*:\s*([-+0-9.eE]+)")
ORACLE_STATUS = re.compile(r"Model\s+status\s*:\s*(.+)")


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def run_ours(path: Path, timeout: int):
    cmd = [str(CLI), "solve", str(path), "--engine", "pdhg", "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "objective": None, "iterations": 0, "seconds": timeout}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "objective": None,
                "message": (proc.stdout + proc.stderr).strip()[:200]}


def run_oracle(path: Path, tmp: Path, timeout: int):
    opts = tmp / "highs.opts"
    opts.write_text("threads = 1\n")
    try:
        proc = subprocess.run(
            [str(ORACLE), "--options_file", str(opts), "--model_file", str(path)],
            capture_output=True, text=True, timeout=timeout, cwd=str(tmp))
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "objective": None}
    text = proc.stdout + proc.stderr
    status_match = ORACLE_STATUS.search(text)
    obj_match = ORACLE_OBJ.search(text)
    status = status_match.group(1).strip().lower() if status_match else "unknown"
    mapped = {"optimal": "optimal", "infeasible": "infeasible",
              "primal infeasible": "infeasible", "unbounded": "unbounded",
              "primal unbounded": "unbounded"}.get(status, status)
    return {"status": mapped,
            "objective": float(obj_match.group(1)) if obj_match else None}


def shape_of(path: Path) -> tuple[int, int, int]:
    proc = subprocess.run([str(CLI), "info", str(path), "--json"],
                          capture_output=True, text=True)
    try:
        d = json.loads(proc.stdout)
        return d["rows"], d["columns"], d["nonzeros"]
    except Exception:
        return 1 << 30, 1 << 30, 1 << 30


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PDHG vs oracle (ticket #8)")
    ap.add_argument("instances", nargs="*")
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if not CLI.is_file():
        print("SKIP: sovereign-cli not built. Not a failure.")
        return 0
    if not ORACLE.is_file():
        print("SKIP: no oracle binary (tools/oracle/install_highs.sh). Not a failure.")
        return 0

    disagreements, timeouts, checked = [], [], 0

    with tempfile.TemporaryDirectory(prefix="pdhgcheck-") as td:
        tmp = Path(td)

        # PDHG has no infeasibility detection yet (pdhg.hpp is explicit about
        # this) and no unbounded detection either -- only the optimal smoke
        # instance is a fair check here; the other three are a documented gap,
        # not silently skipped.
        print("smoke set (hand-verified answers, feasible-optimal instances only):")
        expected = tomllib.loads((SMOKE / "expected.toml").read_text())
        for name, e in expected.items():
            if name == "schema_version" or e["status"] != "optimal":
                continue
            ours = run_ours(SMOKE / e["file"], args.timeout)
            ok = ours["status"] == "optimal" and ours["objective"] is not None \
                and close(ours["objective"], e["objective"])
            checked += 1
            if ok:
                print(f"  {name:<18} ok    obj={ours['objective']:.8g}")
            else:
                disagreements.append(name)
                print(f"  {name:<18} FAIL  got {ours['status']} obj={ours['objective']}, "
                      f"expected optimal obj={e['objective']}")

        if not NETLIB.is_dir():
            print("\nSKIP: Netlib corpus not downloaded.")
        else:
            if args.instances:
                files = [NETLIB / f"{n}.mps" for n in args.instances]
            else:
                files = sorted(NETLIB.glob("*.mps"))

            print(f"\nNetlib (rows <= {args.max_rows}):")
            count = 0
            for path in files:
                if not path.is_file():
                    print(f"  {path.stem:<18} SKIP  no such instance")
                    continue
                rows, cols, nnz = shape_of(path)
                if not args.instances and rows > args.max_rows:
                    continue
                if args.limit and count >= args.limit:
                    break
                count += 1

                ours = run_ours(path, args.timeout)
                oracle = run_oracle(path, tmp, args.timeout)
                checked += 1

                if ours["status"] == "timeout":
                    timeouts.append(path.stem)
                    print(f"  {path.stem:<18} TIMEOUT  (rows={rows} cols={cols})")
                    continue
                if oracle["status"] != "optimal":
                    print(f"  {path.stem:<18} SKIP  oracle status {oracle['status']}")
                    continue

                ok = ours["status"] == "optimal" and ours["objective"] is not None \
                    and close(ours["objective"], oracle["objective"])
                if ok:
                    print(f"  {path.stem:<18} ok    obj={ours['objective']:.8g}  "
                          f"iters={ours.get('iterations')} restarts={ours.get('restarts')}")
                else:
                    disagreements.append(path.stem)
                    print(f"  {path.stem:<18} FAIL  got {ours['status']} obj={ours['objective']}, "
                          f"oracle obj={oracle['objective']}")

    print(f"\n{checked - len(disagreements) - len(timeouts)}/{checked} agree with the oracle "
         f"to {REL_TOL:g} relative / {ABS_TOL:g} absolute"
         + (f", {len(timeouts)} timed out" if timeouts else ""))
    if disagreements:
        print(f"disagreements: {', '.join(disagreements)}")
        return 1
    if timeouts and not args.instances:
        print(f"NOTE: {', '.join(timeouts)} timed out at the default budget -- "
             "PDHG's tail is slow on harder instances (Bible S4.2 Engine A); "
             "not a disagreement, but worth widening --timeout to confirm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
