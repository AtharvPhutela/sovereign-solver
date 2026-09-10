#!/usr/bin/env python3
"""Parser shape check against the oracle -- Build Map ticket #5, gate M0.

Ticket #5's pass condition: "the parser reproduces instance dimensions and
structure identically to the oracle across a varied sample."

So this diffs our reader's (rows, columns, nonzeros) against what the external
oracle reports for the same file. That comparison is the point -- a parser
bug is invisible in our own output and only shows up as disagreement with an
independent reader of the same bytes.

Exit 0 if every checked instance matches (or if nothing is available to check),
1 on any disagreement.

    check_parser_vs_oracle.py                 # every downloaded Netlib instance
    check_parser_vs_oracle.py afiro forplan   # named instances
    check_parser_vs_oracle.py --limit 20
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
SMOKE = REPO / "benchmarks" / "smoke"
CLI_CANDIDATES = [REPO / "build" / "apps" / "sovereign-cli",
                  REPO / "build" / "apps" / "sovereign_cli"]
ORACLE_CANDIDATES = [REPO / "build-oracle" / "bin" / "highs"]

# "LP   afiro has 27 rows; 32 cols; 83 nonzeros"
ORACLE_SHAPE = re.compile(
    r"has\s+(\d+)\s+rows;\s*(\d+)\s+cols;\s*(\d+)\s+nonzeros", re.IGNORECASE)


def find(candidates):
    for c in candidates:
        if c.is_file():
            return c
    return None


def our_shape(cli: Path, path: Path):
    proc = subprocess.run([str(cli), "info", str(path), "--json"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None, (proc.stdout + proc.stderr).strip()[:200]
    try:
        d = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return None, f"unparseable output: {exc}"
    if "error" in d:
        return None, d["error"]
    return d, None


def oracle_shape(oracle: Path, path: Path, tmp: Path):
    # Parse-only would be ideal; HiGHS prints the shape before solving, so a
    # zero iteration limit gets it cheaply without solving anything.
    opts = tmp / "highs.opts"
    opts.write_text("simplex_iteration_limit = 0\nthreads = 1\n")
    proc = subprocess.run(
        [str(oracle), "--options_file", str(opts), "--model_file", str(path)],
        capture_output=True, text=True, cwd=str(tmp))
    text = proc.stdout + proc.stderr
    m = ORACLE_SHAPE.search(text)
    if not m:
        return None, text.strip().splitlines()[-1][:200] if text.strip() else "no output"
    return {"rows": int(m.group(1)), "columns": int(m.group(2)),
            "nonzeros": int(m.group(3))}, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="parser vs oracle shape diff (ticket #5)")
    ap.add_argument("instances", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    cli = find(CLI_CANDIDATES)
    if cli is None:
        print("SKIP: sovereign-cli not built (cmake --build build). Not a failure.")
        return 0
    oracle = find(ORACLE_CANDIDATES)
    if oracle is None:
        print("SKIP: no oracle binary (tools/oracle/install_highs.sh). Not a failure.")
        return 0
    if not NETLIB.is_dir():
        print("SKIP: Netlib corpus not downloaded (benchmarks/fetch_corpus.py). Not a failure.")
        return 0

    if args.instances:
        files = [NETLIB / f"{n}.mps" for n in args.instances]
    else:
        files = sorted(NETLIB.glob("*.mps"))
        if args.limit:
            files = files[: args.limit]
    if not files:
        print("SKIP: no instances to check.")
        return 0

    import tempfile
    mismatched, errors, checked = [], [], 0

    with tempfile.TemporaryDirectory(prefix="parsecheck-") as td:
        tmp = Path(td)
        print(f"comparing {len(files)} instance(s) against the oracle's own parse\n")
        for f in files:
            if not f.is_file():
                errors.append((f.stem, "file not found"))
                continue
            ours, err = our_shape(cli, f)
            if ours is None:
                errors.append((f.stem, f"our reader: {err}"))
                print(f"  {f.stem:<12} ERROR  {err}")
                continue
            theirs, err = oracle_shape(oracle, f, tmp)
            if theirs is None:
                errors.append((f.stem, f"oracle: {err}"))
                print(f"  {f.stem:<12} ERROR  oracle: {err}")
                continue

            checked += 1
            same = all(ours[k] == theirs[k] for k in ("rows", "columns", "nonzeros"))
            if same:
                extra = ""
                if args.verbose and ours.get("warnings"):
                    extra = f"  ({len(ours['warnings'])} reader warning(s))"
                print(f"  {f.stem:<12} ok     {ours['rows']}x{ours['columns']}, "
                      f"{ours['nonzeros']} nnz{extra}")
            else:
                mismatched.append(f.stem)
                print(f"  {f.stem:<12} DIFFER ours {ours['rows']}x{ours['columns']}/"
                      f"{ours['nonzeros']}nnz  vs  oracle {theirs['rows']}x"
                      f"{theirs['columns']}/{theirs['nonzeros']}nnz")

    print()
    if errors:
        print(f"{len(errors)} instance(s) could not be compared:")
        for name, why in errors[:10]:
            print(f"  {name}: {why}")
    if mismatched:
        print(f"FAIL: {len(mismatched)}/{checked} disagree with the oracle: "
              f"{', '.join(mismatched)}")
        return 1
    if errors:
        print(f"FAIL: {len(errors)} instance(s) errored.")
        return 1
    print(f"PASS: {checked}/{checked} match the oracle's rows, columns and nonzeros.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
