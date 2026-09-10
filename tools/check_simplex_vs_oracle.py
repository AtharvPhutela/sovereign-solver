#!/usr/bin/env python3
"""From-scratch simplex vs. the external oracle -- Build Map ticket #4, gate M0.

Ticket #4's pass condition: "the CPU simplex agrees with the external oracle on
Netlib, and survives a degenerate instance without cycling."

WHY THE ORACLE AND NOT THE NETLIB README. The readme's objective values come
from MINOS 5.3 (September 1988), and the readme itself carries a table of
instances where a later solver disagrees. Two of them bite here: `e226` and
`scagr7`, where our answer and the oracle's agree with each other and differ
from the readme. The Oracle rule exists precisely for this -- the independent
*running solver* is the reference, and a 37-year-old printed table is not. Where
the two disagree this harness says so rather than picking a winner silently.

Exit 0 if every checked instance agrees (or nothing is available), 1 otherwise.

    check_simplex_vs_oracle.py                    # every instance we can handle
    check_simplex_vs_oracle.py afiro blend        # named instances
    check_simplex_vs_oracle.py --max-rows 500     # keep the run quick
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

# The dense-basis factorization limits how large an instance the oracle solver
# can take; past this it is honest to skip rather than grind. Kept well below
# the hard guard in simplex.cpp so the default run stays quick.
DEFAULT_MAX_ROWS = 1200
DEFAULT_TIMEOUT = 180

ORACLE_OBJ = re.compile(r"Objective value\s*:\s*([-+0-9.eE]+)")
ORACLE_STATUS = re.compile(r"Model\s+status\s*:\s*(.+)")
REL_TOL = 1e-6
ABS_TOL = 1e-6


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def run_ours(path: Path, timeout: int, extra: list[str] | None = None):
    cmd = [str(CLI), "solve", str(path), "--json"] + (extra or [])
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


def readme_reference(name: str):
    ref = NETLIB / "_reference.toml"
    if not ref.is_file():
        return None
    return tomllib.loads(ref.read_text()).get(name, {}).get("optimal_value")


def shape_of(path: Path) -> tuple[int, int, int]:
    proc = subprocess.run([str(CLI), "info", str(path), "--json"],
                          capture_output=True, text=True)
    try:
        d = json.loads(proc.stdout)
        return d["rows"], d["columns"], d["nonzeros"]
    except Exception:
        return 1 << 30, 1 << 30, 1 << 30


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="simplex vs oracle (ticket #4)")
    ap.add_argument("instances", nargs="*")
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    # Wide instances (few rows, thousands of columns: fit2d is 26 x 10500) are
    # cheap to factorize but slow to price, since pricing is O(columns) every
    # pivot. Cap columns and nonzeros too so the default run stays bounded --
    # these are a performance study for the dense oracle, not a correctness gap.
    ap.add_argument("--max-cols", type=int, default=5000)
    ap.add_argument("--max-nnz", type=int, default=120000)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if not CLI.is_file():
        print("SKIP: sovereign-cli not built. Not a failure.")
        return 0
    if not ORACLE.is_file():
        print("SKIP: no oracle binary (tools/oracle/install_highs.sh). Not a failure.")
        return 0

    disagreements, errors, skipped, checked = [], [], [], 0
    readme_conflicts = []

    with tempfile.TemporaryDirectory(prefix="simplexcheck-") as td:
        tmp = Path(td)

        # The committed smoke set first: it has hand-derived answers, so it
        # checks the solver against arithmetic rather than against another
        # program. If these are wrong, nothing below is worth reading.
        print("smoke set (hand-verified answers):")
        expected = tomllib.loads((SMOKE / "expected.toml").read_text())
        for name, e in expected.items():
            if name == "schema_version":
                continue
            ours = run_ours(SMOKE / e["file"], args.timeout)
            ok = ours["status"] == e["status"]
            if ok and e["status"] == "optimal":
                ok = ours["objective"] is not None and close(ours["objective"], e["objective"])
            checked += 1
            if ok:
                print(f"  {name:<18} ok    {ours['status']}"
                      + (f"  obj={ours['objective']:.10g}" if ours["objective"] is not None else ""))
            else:
                disagreements.append(name)
                print(f"  {name:<18} FAIL  got {ours['status']} obj={ours['objective']}, "
                      f"expected {e['status']} obj={e.get('objective')}")

        if not NETLIB.is_dir():
            print("\nSKIP: Netlib corpus not downloaded.")
        else:
            if args.instances:
                files = [NETLIB / f"{n}.mps" for n in args.instances]
            else:
                files = sorted(NETLIB.glob("*.mps"))

            print(f"\nNetlib LP (instances up to {args.max_rows} rows):")
            shown = 0
            for f in files:
                if not f.is_file():
                    errors.append((f.stem, "missing"))
                    continue
                if not args.instances:
                    r, c, nz = shape_of(f)
                    if r > args.max_rows or c > args.max_cols or nz > args.max_nnz:
                        skipped.append((f.stem, r))
                        continue
                if args.limit and shown >= args.limit:
                    break
                shown += 1

                ours = run_ours(f, args.timeout)
                theirs = run_oracle(f, tmp, args.timeout)

                if ours["status"] in ("timeout", "error", "numerical_failure"):
                    errors.append((f.stem, f"ours: {ours['status']} "
                                           f"{ours.get('message','')}".strip()))
                    print(f"  {f.stem:<12} ERROR ours={ours['status']} "
                          f"{ours.get('message','')}")
                    continue
                if theirs["status"] in ("timeout", "unknown"):
                    errors.append((f.stem, f"oracle: {theirs['status']}"))
                    continue

                checked += 1
                if ours["status"] != theirs["status"]:
                    disagreements.append(f.stem)
                    print(f"  {f.stem:<12} DIFFER status ours={ours['status']} "
                          f"oracle={theirs['status']}")
                    continue
                if ours["status"] != "optimal":
                    print(f"  {f.stem:<12} ok    {ours['status']} (both)")
                    continue

                if close(ours["objective"], theirs["objective"]):
                    note = ""
                    ref = readme_reference(f.stem)
                    if ref is not None and not close(theirs["objective"], float(ref)):
                        # We and the oracle agree; the 1988 table does not.
                        readme_conflicts.append(f.stem)
                        note = "  [readme value is stale]"
                    print(f"  {f.stem:<12} ok    {ours['objective']:.10g}"
                          f"  it={ours['iterations']:<6} {ours['seconds']:.2f}s{note}")
                else:
                    disagreements.append(f.stem)
                    print(f"  {f.stem:<12} DIFFER ours {ours['objective']:.12g}  "
                          f"oracle {theirs['objective']:.12g}")

    print()
    if skipped:
        print(f"{len(skipped)} instance(s) skipped as too large for the dense basis "
              f"(>{args.max_rows} rows): "
              + ", ".join(f"{n}({r})" for n, r in skipped[:8])
              + (" ..." if len(skipped) > 8 else ""))
    if readme_conflicts:
        print(f"{len(readme_conflicts)} instance(s) where we and the oracle agree but the "
              f"Netlib readme's 1988 value differs: {', '.join(readme_conflicts)}")
        print("  (the running oracle is the reference; the printed table is not)")
    if errors:
        print(f"\n{len(errors)} instance(s) could not be compared:")
        for name, why in errors[:12]:
            print(f"  {name}: {why}")
    if disagreements:
        print(f"\nFAIL: {len(disagreements)} disagreement(s): {', '.join(disagreements)}")
        return 1
    if errors:
        print("\nFAIL: errors above.")
        return 1
    print(f"PASS: {checked}/{checked} agree with the oracle (and with the hand-verified "
          f"smoke answers).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
