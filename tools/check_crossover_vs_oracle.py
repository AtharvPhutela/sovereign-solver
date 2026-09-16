#!/usr/bin/env python3
"""Crossover vs. the external oracle -- Build Map ticket #11, gate M2.

Ticket #11's pass condition: "an exact vertex is produced from a GPU interior
solution, and total time still beats CPU." The exactness half is checkable
here at simplex-grade tolerance (Bible: crossover's whole point is that it is
NOT PDHG's ~1e-4) -- same tight tolerance check_simplex_vs_oracle.py (#4)
uses, deliberately not PDHG's looser one. The wall-clock-beats-CPU half needs
a real GPU to mean anything (HANDOFF S9/S17.2/S20.3); this script also prints
the crossover-vs-plain-simplex wall time on this machine's Host backend as a
mechanical sanity check, not a demonstration of the GPU claim.

    check_crossover_vs_oracle.py                  # the smoke set + small Netlib instances
    check_crossover_vs_oracle.py afiro adlittle    # named instances
    check_crossover_vs_oracle.py --max-rows 200    # keep the run quick
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NETLIB = REPO / "benchmarks" / "data" / "netlib_lp"
SMOKE = REPO / "benchmarks" / "smoke"
CLI = REPO / "build" / "apps" / "sovereign-cli"
ORACLE = REPO / "build-oracle" / "bin" / "highs"

# Exact-tolerance, same as check_simplex_vs_oracle.py -- crossover's whole
# point is producing simplex-grade digits, not PDHG's ~1e-4.
REL_TOL = 1e-6
ABS_TOL = 1e-6
DEFAULT_MAX_ROWS = 250
DEFAULT_TIMEOUT = 60

ORACLE_OBJ = re.compile(r"Objective value\s*:\s*([-+0-9.eE]+)")
ORACLE_STATUS = re.compile(r"Model\s+status\s*:\s*(.+)")


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def run_ours(path: Path, engine: str, timeout: int):
    cmd = [str(CLI), "solve", str(path), "--engine", engine, "--json"]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "objective": None, "seconds": timeout}, timeout
    wall = time.monotonic() - t0
    try:
        return json.loads(proc.stdout), wall
    except json.JSONDecodeError:
        return {"status": "error", "objective": None,
                "message": (proc.stdout + proc.stderr).strip()[:200]}, wall


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
    ap = argparse.ArgumentParser(description="Crossover vs oracle (ticket #11)")
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

    with tempfile.TemporaryDirectory(prefix="crossovercheck-") as td:
        tmp = Path(td)

        print("smoke set (hand-verified answers, feasible-optimal instances only):")
        expected = tomllib.loads((SMOKE / "expected.toml").read_text())
        for name, e in expected.items():
            if name == "schema_version" or e["status"] != "optimal":
                continue
            ours, _ = run_ours(SMOKE / e["file"], "crossover", args.timeout)
            ok = ours["status"] == "optimal" and ours["objective"] is not None \
                and close(ours["objective"], e["objective"])
            checked += 1
            if ok:
                print(f"  {name:<18} ok    obj={ours['objective']:.10g}")
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

            print(f"\nNetlib (rows <= {args.max_rows}), crossover vs. oracle "
                 f"(+ plain-simplex wall time on this machine's Host backend, "
                 f"NOT the GPU claim -- see HANDOFF S17.2/S20.3):")
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

                ours, xover_wall = run_ours(path, "crossover", args.timeout)
                _, simplex_wall = run_ours(path, "simplex", args.timeout)
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
                    faster = "faster" if xover_wall < simplex_wall else "slower"
                    print(f"  {path.stem:<18} ok    obj={ours['objective']:.10g}  "
                          f"checkpoints={ours.get('checkpoint_attempts')}  "
                          f"xover={xover_wall*1000:.1f}ms simplex={simplex_wall*1000:.1f}ms ({faster})")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
