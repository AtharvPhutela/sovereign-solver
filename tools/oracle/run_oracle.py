#!/usr/bin/env python3
"""Black-box optimization oracle -- Build Map ticket #2, gate M0.

Runs an established solver as a SUBPROCESS over an MPS/LP file on disk and
normalizes its answer to a single JSON schema. This is the "independent answer"
side of the Oracle rule: every ticket that computes an optimization result is
checked against what this produces (Bible S1.2, Part VII).

    run_oracle.py model.mps
    run_oracle.py model.mps.gz --out result.json --with-solution
    run_oracle.py model.lp --time-limit 60 --backend highs

SOVEREIGNTY INVARIANT (asserted in the output as backend.linked = false):
this module never imports a solver, never links one, and never puts one on the
solver's link line. It only spawns a process and reads files. If it cannot do
that, it reports status "error" -- it does not fall back to an in-process
library.

Backend resolution order (first hit wins):
  1. --backend PATH  or  $SOLVER_ORACLE       explicit executable
  2. build-oracle/bin/highs                   built by tools/oracle/install_highs.sh
  3. highs / cbc / glpsol / scip on $PATH

Normalized status vocabulary:
  optimal · infeasible · unbounded · infeasible_or_unbounded ·
  time_limit · iteration_limit · error
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ORACLE_SCHEMA = 1
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILT_HIGHS = REPO_ROOT / "build-oracle" / "bin" / "highs"

STATUS = {
    "optimal", "infeasible", "unbounded", "infeasible_or_unbounded",
    "time_limit", "iteration_limit", "error",
}


# --------------------------------------------------------------------------
# backend discovery
# --------------------------------------------------------------------------

def resolve_backend(explicit: str | None) -> tuple[str, Path] | tuple[str, None]:
    """Return (backend_name, executable_path) or (name, None) if nothing found."""
    candidates: list[tuple[str, Path]] = []

    chosen = explicit or os.environ.get("SOLVER_ORACLE")
    if chosen:
        p = Path(chosen)
        if p.is_file() and os.access(p, os.X_OK):
            return _name_from_path(p), p
        found = shutil.which(chosen)
        if found:
            return _name_from_path(Path(found)), Path(found)
        return _name_from_path(p), None

    if BUILT_HIGHS.is_file() and os.access(BUILT_HIGHS, os.X_OK):
        return "highs", BUILT_HIGHS

    for name in ("highs", "cbc", "glpsol", "scip"):
        found = shutil.which(name)
        if found:
            candidates.append((name, Path(found)))
    if candidates:
        return candidates[0]
    return "none", None


def _name_from_path(p: Path) -> str:
    stem = p.name.lower()
    for known in ("highs", "cbc", "glpsol", "glpk", "scip"):
        if known in stem:
            return "glpsol" if known == "glpk" else known
    return stem


# --------------------------------------------------------------------------
# model handling
# --------------------------------------------------------------------------

def prepare_model(path: Path, workdir: Path) -> tuple[Path, str]:
    """Decompress .gz if needed. Return (usable_path, sha256_of_original_bytes)."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if path.suffix == ".gz":
        inner = workdir / path.stem            # foo.mps.gz -> foo.mps
        inner.write_bytes(gzip.decompress(raw))
        return inner, digest
    return path, digest


# --------------------------------------------------------------------------
# HiGHS adapter
# --------------------------------------------------------------------------

_HIGHS_STATUS_MAP = {
    "optimal": "optimal",
    "infeasible": "infeasible",
    "primal infeasible": "infeasible",
    "unbounded": "unbounded",
    "primal unbounded": "unbounded",
    "unbounded or infeasible": "infeasible_or_unbounded",
    "primal infeasible or unbounded": "infeasible_or_unbounded",
    "time limit reached": "time_limit",
    "reached time limit": "time_limit",
    "iteration limit reached": "iteration_limit",
    "reached iteration limit": "iteration_limit",
}


def run_highs(exe: Path, model: Path, sol_path: Path, *, threads: int,
              time_limit: float | None) -> tuple[subprocess.CompletedProcess, list[str]]:
    # HiGHS 1.11 CLI has no --threads / --write_solution_style flags: those go
    # through an options file. --parallel and --time_limit are real CLI flags.
    opts = sol_path.parent / "highs.opts"
    opt_lines = [
        f"threads = {threads}",
        "write_solution_style = 1",     # labelled columnar table (see parser)
        "write_solution_to_file = true",
    ]
    if time_limit is not None:
        opt_lines.append(f"time_limit = {time_limit}")
    opts.write_text("\n".join(opt_lines) + "\n")

    argv = [
        str(exe),
        "--options_file", str(opts),
        "--solution_file", str(sol_path),
        "--parallel", "off",
        "--model_file", str(model),
    ]
    # HiGHS drops a HiGHS.log in the working directory; keep it in the scratch
    # dir, not wherever the caller happened to be.
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=sol_path.parent)
    return proc, argv


def parse_highs(proc: subprocess.CompletedProcess, sol_path: Path) -> dict:
    text = proc.stdout + "\n" + proc.stderr
    status = None
    objective = None

    m = re.search(r"Model\s+status\s*:?\s*(.+)", text)
    if m:
        status = _HIGHS_STATUS_MAP.get(m.group(1).strip().lower())
    m = re.search(r"Objective\s+value\s*:?\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)", text)
    if m:
        objective = float(m.group(1))

    primal: dict[str, float] = {}
    dual: dict[str, float] = {}
    if sol_path.is_file():
        s = sol_path.read_text(errors="replace")
        sm = re.search(r"Model status\s*:?\s*(.+)", s)
        if sm and status is None:
            status = _HIGHS_STATUS_MAP.get(sm.group(1).strip().lower())
        om = re.search(r"Objective value\s*:?\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)", s)
        if om and objective is None:
            objective = float(om.group(1))
        primal, dual = _parse_highs_style1_table(s)

    if status is None:
        status = "error"
    return {"status": status, "objective": objective, "primal": primal, "dual": dual}


def _parse_highs_style1_table(s: str) -> tuple[dict[str, float], dict[str, float]]:
    """Parse HiGHS write_solution_style = 1 (labelled columnar table).

        Columns
            Index Status  Lower  Upper  Primal  Dual  Name
                0     BS      0    inf      80     0   X01
                ...
        Rows
            Index Status  Lower  Upper  Primal  Dual  Name
                ...

    Returns (primal, dual) each mapping variable/constraint name -> value, with
    columns and rows merged into one namespace (HiGHS names don't collide).
    Robust to version drift: rows that don't start with an integer index or
    don't have the expected column count are skipped.
    """
    primal: dict[str, float] = {}
    dual: dict[str, float] = {}
    section: str | None = None
    for line in s.splitlines():
        t = line.strip()
        low = t.lower()
        if low == "columns":
            section = "col"; continue
        if low == "rows":
            section = "row"; continue
        if low.startswith(("model status", "objective value", "basis")):
            section = None; continue
        if section is None or not t:
            continue
        parts = t.split()
        if len(parts) < 7 or not parts[0].lstrip("-").isdigit():
            continue                                     # header row or noise
        # Index Status Lower Upper Primal Dual Name...  (Name may contain spaces)
        try:
            pval = float(parts[4])
            dval = float(parts[5])
        except ValueError:
            continue
        name = " ".join(parts[6:])
        primal[name] = pval
        dual[name] = dval
    return primal, dual


# --------------------------------------------------------------------------
# CBC adapter (bonus -- used if highs absent and cbc present)
# --------------------------------------------------------------------------

def run_cbc(exe: Path, model: Path, sol_path: Path, *, threads: int,
           time_limit: float | None) -> tuple[subprocess.CompletedProcess, list[str]]:
    argv = [str(exe), str(model)]
    if time_limit is not None:
        argv += ["-seconds", str(time_limit)]
    argv += ["-threads", str(max(1, threads)), "-solve", "-solution", str(sol_path)]
    return subprocess.run(argv, capture_output=True, text=True), argv


def parse_cbc(proc: subprocess.CompletedProcess, sol_path: Path) -> dict:
    text = proc.stdout + "\n" + proc.stderr
    status = "error"
    objective = None
    if re.search(r"Optimal solution found", text) or re.search(r"Optimal - objective", text):
        status = "optimal"
    elif "infeasible" in text.lower():
        status = "infeasible"
    elif "unbounded" in text.lower():
        status = "unbounded"
    elif "stopped on time" in text.lower():
        status = "time_limit"
    m = re.search(r"[Oo]bjective(?: value)?\s*(?:=|:)?\s*([+-]?[\d.]+(?:[eE][+-]?\d+)?)", text)
    if m:
        objective = float(m.group(1))
    primal: dict[str, float] = {}
    if sol_path.is_file():
        for line in sol_path.read_text(errors="replace").splitlines():
            parts = line.split()
            # CBC solution rows: <index> <name> <value> <reduced-cost>
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    primal[parts[1]] = float(parts[2])
                except ValueError:
                    pass
            m2 = re.match(r"Optimal - objective value\s+([+-]?[\d.eE]+)", line)
            if m2 and objective is None:
                objective = float(m2.group(1))
    return {"status": status, "objective": objective, "primal": primal, "dual": {}}


ADAPTERS = {
    "highs": (run_highs, parse_highs),
    "cbc": (run_cbc, parse_cbc),
}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def backend_version(name: str, exe: Path) -> str:
    for flag in ("--version", "-v", "version"):
        try:
            p = subprocess.run([str(exe), flag], capture_output=True, text=True, timeout=10)
            out = (p.stdout + p.stderr).strip().splitlines()
            if out:
                return out[0][:120]
        except (subprocess.SubprocessError, OSError):
            continue
    return "unknown"


def solve(model_path: Path, *, backend: str | None, threads: int,
          time_limit: float | None, with_solution: bool) -> dict:
    name, exe = resolve_backend(backend)
    result: dict = {
        "oracle_schema": ORACLE_SCHEMA,
        "instance": model_path.name.split(".")[0],
        "model_path": str(model_path),
        "backend": {"name": name, "binary": str(exe) if exe else None, "linked": False},
        "invocation": {"threads": threads, "time_limit_s": time_limit},
        "status": "error",
        "objective": None,
        "solve_seconds": None,
    }

    if exe is None:
        result["error"] = (
            f"no oracle solver found (looked for $SOLVER_ORACLE, {BUILT_HIGHS}, "
            f"and highs/cbc/glpsol/scip on PATH). "
            f"Run tools/oracle/install_highs.sh."
        )
        return result

    if name not in ADAPTERS:
        result["error"] = f"backend '{name}' has no adapter (have: {', '.join(ADAPTERS)})"
        return result

    result["backend"]["version"] = backend_version(name, exe)
    run_fn, parse_fn = ADAPTERS[name]

    with tempfile.TemporaryDirectory(prefix="oracle-") as td:
        work = Path(td)
        model, digest = prepare_model(model_path, work)
        result["model_sha256"] = digest
        sol = work / "solution.txt"

        t0 = time.perf_counter()
        try:
            proc, argv = run_fn(exe, model, sol, threads=threads, time_limit=time_limit)
        except (subprocess.SubprocessError, OSError) as exc:
            result["error"] = f"subprocess failed: {exc}"
            return result
        result["solve_seconds"] = round(time.perf_counter() - t0, 4)
        result["backend"]["argv"] = argv
        result["backend"]["returncode"] = proc.returncode
        result["stdout_tail"] = proc.stdout.strip().splitlines()[-8:]

        parsed = parse_fn(proc, sol)

    result["status"] = parsed["status"] if parsed["status"] in STATUS else "error"
    # An objective value is only meaningful when the solver actually has a point
    # in hand. Solvers still print "Objective value: 0" for an infeasible model;
    # don't let that leak into the reference record.
    if result["status"] in ("infeasible", "unbounded", "infeasible_or_unbounded", "error"):
        result["objective"] = None
    else:
        result["objective"] = parsed["objective"]
    result["primal_present"] = bool(parsed["primal"])
    result["dual_present"] = bool(parsed["dual"])
    if with_solution:
        result["primal"] = parsed["primal"]
        result["dual"] = parsed["dual"]
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Black-box optimization oracle (ticket #2)")
    ap.add_argument("model", type=Path, help="MPS / MPS.gz / LP file")
    ap.add_argument("--backend", help="explicit solver executable or name (else auto)")
    ap.add_argument("--threads", type=int, default=1,
                    help="solver threads (default 1: reproducible reference answers)")
    ap.add_argument("--time-limit", type=float, default=None, metavar="SECONDS")
    ap.add_argument("--with-solution", action="store_true",
                    help="include full primal/dual vectors in the JSON")
    ap.add_argument("--out", type=Path, help="write JSON here instead of stdout")
    args = ap.parse_args(argv)

    if not args.model.is_file():
        print(f"no such file: {args.model}", file=sys.stderr)
        return 2

    result = solve(args.model, backend=args.backend, threads=args.threads,
                   time_limit=args.time_limit, with_solution=args.with_solution)

    blob = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(blob + "\n")
        print(f"{result['status']:<12} obj={result['objective']}  -> {args.out}")
    else:
        print(blob)

    return 0 if result["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
