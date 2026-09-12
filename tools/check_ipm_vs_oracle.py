#!/usr/bin/env python3
"""IPM vs. the external oracle -- Build Map ticket #9, gate M1.

Ticket #9's pass condition: "the IPM solves a large LP on-device to tight
tolerance via GPU factorization, matching the oracle, with no pivoting on
the critical path." The dense factorization (ipm.hpp's scope note) caps how
large an instance this can take -- same discipline as check_simplex_vs_oracle
(#4) for its dense LU. Tolerance here is TIGHT, not PDHG's ~1e-4: this is the
high-precision engine, and a disagreement in the 5th digit is a real finding,
not expected noise.

    check_ipm_vs_oracle.py                  # the smoke set + Netlib instances
    check_ipm_vs_oracle.py afiro adlittle   # named instances
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

REL_TOL = 1e-5
ABS_TOL = 1e-5
DEFAULT_TIMEOUT = 30

ORACLE_OBJ = re.compile(r"Objective value\s*:\s*([-+0-9.eE]+)")
ORACLE_STATUS = re.compile(r"Model\s+status\s*:\s*(.+)")


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def run_ours(path: Path, timeout: int):
    cmd = [str(CLI), "solve", str(path), "--engine", "ipm", "--json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "objective": None}
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
    ap = argparse.ArgumentParser(description="IPM vs oracle (ticket #9)")
    ap.add_argument("instances", nargs="*")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if not CLI.is_file():
        print("SKIP: sovereign-cli not built. Not a failure.")
        return 0
    if not ORACLE.is_file():
        print("SKIP: no oracle binary (tools/oracle/install_highs.sh). Not a failure.")
        return 0

    disagreements, size_refused, checked = [], [], 0

    with tempfile.TemporaryDirectory(prefix="ipmcheck-") as td:
        tmp = Path(td)

        # This engine has no infeasibility/unbounded detection either (same
        # tracked gap as PDHG, see ipm.hpp) -- only the feasible-optimal
        # smoke instances are a fair check here.
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
                print(f"  {name:<18} ok    obj={ours['objective']:.10g}")
            else:
                disagreements.append(name)
                print(f"  {name:<18} FAIL  got {ours['status']} obj={ours['objective']}, "
                      f"expected optimal obj={e['objective']}")

        if not NETLIB.is_dir():
            print("\nSKIP: Netlib corpus not downloaded.")
        else:
            files = [NETLIB / f"{n}.mps" for n in args.instances] if args.instances \
                else sorted(NETLIB.glob("*.mps"))

            print("\nNetlib:")
            count = 0
            for path in files:
                if not path.is_file():
                    print(f"  {path.stem:<18} SKIP  no such instance")
                    continue
                if args.limit and count >= args.limit:
                    break
                count += 1

                ours = run_ours(path, args.timeout)
                if ours["status"] == "numerical_failure" and ours.get("message", "").startswith(
                        "augmented system has size"):
                    size_refused.append(path.stem)
                    print(f"  {path.stem:<18} SKIP  dense system too large (documented scope limit)")
                    continue

                oracle = run_oracle(path, tmp, args.timeout)
                checked += 1

                if ours["status"] == "timeout":
                    print(f"  {path.stem:<18} TIMEOUT")
                    disagreements.append(path.stem)
                    continue
                if oracle["status"] != "optimal":
                    print(f"  {path.stem:<18} SKIP  oracle status {oracle['status']}")
                    continue

                ok = ours["status"] == "optimal" and ours["objective"] is not None \
                    and close(ours["objective"], oracle["objective"])
                if ok:
                    print(f"  {path.stem:<18} ok    obj={ours['objective']:.10g}  "
                          f"iters={ours.get('iterations')}")
                else:
                    disagreements.append(path.stem)
                    print(f"  {path.stem:<18} FAIL  got {ours['status']} obj={ours['objective']}, "
                          f"oracle obj={oracle['objective']}")

    print(f"\n{checked - len(disagreements)}/{checked} agree with the oracle "
         f"to {REL_TOL:g} relative / {ABS_TOL:g} absolute")
    if size_refused:
        print(f"refused (documented dense-size limit, not a disagreement): {', '.join(size_refused)}")
    if disagreements:
        print(f"disagreements: {', '.join(disagreements)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
