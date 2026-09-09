#!/usr/bin/env python3
"""Tests for the black-box oracle wrapper -- Build Map ticket #2.

Two layers:

* Parsing/normalization is tested against a FAKE solver -- a shell script that
  emits canned output -- so the wrapper's contract is pinned without needing any
  real solver present.
* If a real oracle is available (build-oracle/bin/highs or one on PATH), the
  smoke instances are solved for real and checked against benchmarks/smoke/
  expected.toml. Skipped, not failed, when absent.

Run:  python3 tests/test_oracle.py
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools" / "oracle"))
import run_oracle as ro  # noqa: E402

SMOKE = REPO / "benchmarks" / "smoke"
BUILT_HIGHS = REPO / "build-oracle" / "bin" / "highs"


@contextlib.contextmanager
def tempdir():
    d = Path(tempfile.mkdtemp(prefix="oracle-test-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _which_any(names):
    return any(shutil.which(n) for n in names)


def _write_exec(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


class BackendResolution(unittest.TestCase):
    def test_explicit_missing_path_is_not_a_crash(self):
        name, exe = ro.resolve_backend("/no/such/solver")
        self.assertIsNone(exe)

    def test_env_var_is_honoured(self):
        with tempdir() as td:
            fake = _write_exec(td / "highs", "#!/bin/sh\necho hi\n")
            old = os.environ.get("SOLVER_ORACLE")
            os.environ["SOLVER_ORACLE"] = str(fake)
            try:
                name, exe = ro.resolve_backend(None)
            finally:
                os.environ.pop("SOLVER_ORACLE", None)
                if old is not None:
                    os.environ["SOLVER_ORACLE"] = old
            self.assertEqual(name, "highs")
            self.assertEqual(exe, fake)

    def test_name_inference_from_path(self):
        self.assertEqual(ro._name_from_path(Path("/opt/x/glpsol")), "glpsol")
        self.assertEqual(ro._name_from_path(Path("/a/CBC")), "cbc")


class NoBackend(unittest.TestCase):
    def test_reports_error_status_not_exception(self):
        with tempdir() as td:
            model = td / "m.mps"
            model.write_text("NAME T\nROWS\n N C\nCOLUMNS\nRHS\nENDATA\n")
            os.environ["SOLVER_ORACLE"] = str(td / "definitely-absent")
            try:
                res = ro.solve(model, backend=None, threads=1,
                               time_limit=None, with_solution=False)
            finally:
                os.environ.pop("SOLVER_ORACLE", None)
        self.assertEqual(res["status"], "error")
        self.assertIn("no oracle solver found", res["error"])
        self.assertFalse(res["backend"]["linked"])   # invariant holds even on error


class FakeHighsParsing(unittest.TestCase):
    """Pin the HiGHS adapter's parsing without a real HiGHS."""

    CANNED_STDOUT = (
        "Model name          : t\n"
        "Model status        : Optimal\n"
        "Objective value     : -4.6475314286e+02\n"
        "HiGHS run time      :          0.00\n"
    )
    CANNED_SOL = (
        "Columns\n"
        "    Index Status        Lower        Upper       Primal         Dual  Name\n"
        "        0     BS            0          inf           80            0  X01\n"
        "        1     LB            0          inf            0      2.24966  X02\n"
        "Rows\n"
        "    Index Status        Lower        Upper       Primal         Dual  Name\n"
        "        0     FX            0            0            0    -0.628571  R09\n"
        "\n"
        "Model status: Optimal\n"
        "\n"
        "Objective value: -464.7531428571428\n"
    )

    def _fake(self, td: Path, *, status="Optimal", obj="-4.6475314286e+02", rc=0) -> Path:
        # Echoes canned stdout and writes the canned solution file HiGHS would.
        script = (
            "#!/bin/sh\n"
            "sol=''\n"
            'while [ $# -gt 0 ]; do\n'
            '  case "$1" in\n'
            "    --solution_file) sol=\"$2\"; shift 2;;\n"
            "    *) shift;;\n"
            "  esac\n"
            "done\n"
            f"printf '%s' '{self.CANNED_STDOUT.replace(chr(39), '')}'\n"
            f"printf '{status}\\n' >/dev/null\n"
            '[ -n "$sol" ] && cat > "$sol" <<\'SOLEOF\'\n'
            f"{self.CANNED_SOL}"
            "SOLEOF\n"
            f"exit {rc}\n"
        )
        return _write_exec(td / "highs", script)

    def test_optimal_parsed(self):
        with tempdir() as td:
            model = td / "t.mps"
            model.write_text("NAME t\nROWS\n N COST\nCOLUMNS\nRHS\nENDATA\n")
            res = ro.solve(model, backend=str(self._fake(td)), threads=1,
                           time_limit=None, with_solution=True)
        self.assertEqual(res["status"], "optimal")
        self.assertAlmostEqual(res["objective"], -464.7531428571, places=6)
        self.assertFalse(res["backend"]["linked"])
        self.assertEqual(res["primal"]["X01"], 80.0)
        self.assertEqual(res["dual"]["X02"], 2.24966)
        self.assertEqual(res["dual"]["R09"], -0.628571)
        self.assertTrue(res["model_sha256"])

    def test_gz_model_is_decompressed(self):
        import gzip
        with tempdir() as td:
            model = td / "t.mps.gz"
            model.write_bytes(gzip.compress(b"NAME t\nROWS\n N COST\nCOLUMNS\nRHS\nENDATA\n"))
            res = ro.solve(model, backend=str(self._fake(td)), threads=1,
                           time_limit=None, with_solution=False)
        self.assertEqual(res["status"], "optimal")

    def test_infeasible_nulls_the_objective(self):
        with tempdir() as td:
            model = td / "t.mps"
            model.write_text("NAME t\nROWS\n N C\nCOLUMNS\nRHS\nENDATA\n")
            fake = self._fake(td, status="Infeasible")
            # override: make stdout say Infeasible
            _write_exec(fake, fake.read_text().replace(
                "Model status        : Optimal", "Model status        : Infeasible").replace(
                "Model status: Optimal", "Model status: Infeasible"))
            res = ro.solve(model, backend=str(fake), threads=1,
                           time_limit=None, with_solution=False)
        self.assertEqual(res["status"], "infeasible")
        self.assertIsNone(res["objective"])


class StyleTableParser(unittest.TestCase):
    def test_handles_missing_columns_gracefully(self):
        primal, dual = ro._parse_highs_style1_table(
            "Columns\n   header junk not numeric\n   0 BS 0 inf 5 1 X\nRows\n")
        self.assertEqual(primal["X"], 5.0)
        self.assertEqual(dual["X"], 1.0)

    def test_ignores_text_outside_sections(self):
        primal, _ = ro._parse_highs_style1_table("noise\n0 BS 0 inf 9 0 Y\n")
        self.assertEqual(primal, {})   # no "Columns"/"Rows" header seen


@unittest.skipUnless(BUILT_HIGHS.is_file() or _which_any(["highs", "cbc"]),
                     "no real oracle solver available")
class RealOracleSmoke(unittest.TestCase):
    """End-to-end against the committed smoke set and its hand-verified answers."""

    @classmethod
    def setUpClass(cls):
        cls.expected = tomllib.loads((SMOKE / "expected.toml").read_text())

    def _solve(self, name: str) -> dict:
        return ro.solve(SMOKE / f"{name}.mps", backend=None, threads=1,
                        time_limit=30, with_solution=True)

    def test_feasible_bounded(self):
        r = self._solve("tiny_lp")
        exp = self.expected["tiny_lp"]
        self.assertEqual(r["status"], exp["status"])
        self.assertAlmostEqual(r["objective"], exp["objective"], delta=1e-6)

    def test_infeasible(self):
        self.assertEqual(self._solve("tiny_infeasible")["status"], "infeasible")

    def test_unbounded(self):
        self.assertIn(self._solve("tiny_unbounded")["status"],
                      ("unbounded", "infeasible_or_unbounded"))

    def test_degenerate_optimum(self):
        r = self._solve("tiny_degenerate")
        self.assertEqual(r["status"], "optimal")
        self.assertAlmostEqual(r["objective"], -2.0, delta=1e-6)

    def test_json_roundtrip_and_linked_false(self):
        r = self._solve("tiny_lp")
        blob = json.dumps(r)
        self.assertFalse(json.loads(blob)["backend"]["linked"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
