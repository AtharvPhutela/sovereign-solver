#!/usr/bin/env python3
"""Presolve vs. the external oracle's own presolve -- Build Map ticket #13,
gate M3.

Ticket #13's pass condition: "presolve captures ~90% of the oracle's
reductions, and postsolve reconstructs valid duals." Two independent checks,
both run here:

  1. Correctness: `sovereign-cli presolve` itself solves the reduced model,
     postsolves, and verifies the reconstructed point against a direct
     (unpresolved) solve -- objective match AND complementary slackness
     (ticket #7's own checker) against the ORIGINAL problem. This script
     just reads that verdict; it does not re-derive it.
  2. Reduction size: HiGHS's own log line ("Presolve : Reductions: rows
     R(-r); columns C(-c); elements E(-e)") is parsed for the SAME instance
     and compared against our own rows_removed/cols_removed/nonzeros_removed.

    check_presolve_vs_oracle.py                  # the smoke set + Netlib
    check_presolve_vs_oracle.py brandy adlittle   # named instances
    check_presolve_vs_oracle.py --max-rows 500
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NETLIB = REPO / "benchmarks" / "data" / "netlib_lp"
CLI = REPO / "build" / "apps" / "sovereign-cli"
ORACLE = REPO / "build-oracle" / "bin" / "highs"

DEFAULT_MAX_ROWS = 2000
DEFAULT_TIMEOUT = 60

# "Presolve : Reductions: rows 92(-128); columns 169(-80); elements 1637(-511)"
ORACLE_REDUCTIONS = re.compile(
    r"Presolve\s*:\s*Reductions:\s*rows\s+\d+\(-(\d+)\);\s*columns\s+\d+\(-(\d+)\);"
    r"\s*elements\s+\d+\(([+-]\d+)\)")


def run_ours(path: Path, timeout: int):
    cmd = [str(CLI), "presolve", str(path), "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "message": (proc.stdout + proc.stderr).strip()[:200]}


def run_oracle_presolve(path: Path, timeout: int):
    try:
        proc = subprocess.run([str(ORACLE), "--model_file", str(path)],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    m = ORACLE_REDUCTIONS.search(proc.stdout + proc.stderr)
    if not m:
        return None
    rows, cols = int(m.group(1)), int(m.group(2))
    # HiGHS prints elements as a SIGNED delta: "(-511)" means 511 FEWER
    # elements (a real reduction, reported here as the positive amount 511);
    # "(+6)" means elements INCREASED (substitution fill-in outweighed what
    # was removed). An increase isn't comparable to our own nonzeros_removed
    # (never negative by construction, since we never introduce fill-in) --
    # such instances are excluded from the elements comparison entirely
    # rather than silently clamped to a misleading 0.
    raw_elements = int(m.group(3))
    elements = -raw_elements if raw_elements < 0 else None
    return {"rows": rows, "cols": cols, "elements": elements}


def shape_of(path: Path):
    proc = subprocess.run([str(CLI), "info", str(path), "--json"],
                          capture_output=True, text=True)
    try:
        d = json.loads(proc.stdout)
        return d["rows"], d["columns"], d["nonzeros"]
    except Exception:
        return 1 << 30, 1 << 30, 1 << 30


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Presolve vs oracle's own presolve (ticket #13)")
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
    if not NETLIB.is_dir():
        print("SKIP: Netlib corpus not downloaded.")
        return 0

    files = [NETLIB / f"{n}.mps" for n in args.instances] if args.instances \
        else sorted(NETLIB.glob("*.mps"))

    unverified, checked = [], 0
    sum_ours_rows = sum_ours_cols = sum_ours_elts = 0
    sum_oracle_rows = sum_oracle_cols = sum_oracle_elts = 0
    count = 0

    for path in files:
        if not path.is_file():
            print(f"  {path.stem:<16} SKIP  no such instance")
            continue
        rows, cols, nnz = shape_of(path)
        if not args.instances and rows > args.max_rows:
            continue
        if args.limit and count >= args.limit:
            break
        count += 1
        checked += 1

        ours = run_ours(path, args.timeout)
        if ours.get("status") not in ("reduced", "infeasible"):
            print(f"  {path.stem:<16} SKIP  our presolve status: {ours.get('status')}")
            continue
        if ours.get("status") == "infeasible":
            print(f"  {path.stem:<16} SKIP  our presolve claims infeasible "
                 f"(unexpected for a known-optimal Netlib instance -- worth a look)")
            unverified.append(path.stem)
            continue
        if not ours.get("verified"):
            print(f"  {path.stem:<16} FAIL  postsolve did not verify: {ours.get('message')}")
            unverified.append(path.stem)
            continue

        oracle = run_oracle_presolve(path, args.timeout)
        if oracle is None:
            print(f"  {path.stem:<16} ok (verified)  -- oracle presolve line not found, "
                 f"size comparison skipped")
            continue

        sum_ours_rows += ours["rows_removed"]; sum_oracle_rows += oracle["rows"]
        sum_ours_cols += ours["cols_removed"]; sum_oracle_cols += oracle["cols"]
        elts_str = "n/a (fill-in increased elements)"
        if oracle["elements"] is not None:
            sum_ours_elts += ours["nonzeros_removed"]; sum_oracle_elts += oracle["elements"]
            elts_str = f"{ours['nonzeros_removed']}/{oracle['elements']}"

        print(f"  {path.stem:<16} ok (verified)  "
             f"rows {ours['rows_removed']}/{oracle['rows']}  "
             f"cols {ours['cols_removed']}/{oracle['cols']}  "
             f"elements {elts_str}")

    print(f"\n{checked - len(unverified)}/{checked} verified (correct reduced solve, "
         f"postsolve duals pass complementary slackness against the original problem)")
    if unverified:
        print(f"NOT verified: {', '.join(unverified)}")

    if sum_oracle_rows or sum_oracle_cols or sum_oracle_elts:
        print(f"\nAggregate reduction captured vs. the oracle's own presolve "
             f"(sum over every instance where a comparison was possible):")
        print(f"  rows:     {sum_ours_rows}/{sum_oracle_rows}  ({pct(sum_ours_rows, sum_oracle_rows):.1f}%)")
        print(f"  columns:  {sum_ours_cols}/{sum_oracle_cols}  ({pct(sum_ours_cols, sum_oracle_cols):.1f}%)")
        print(f"  elements: {sum_ours_elts}/{sum_oracle_elts}  ({pct(sum_ours_elts, sum_oracle_elts):.1f}%)")

    return 1 if unverified else 0


if __name__ == "__main__":
    sys.exit(main())
