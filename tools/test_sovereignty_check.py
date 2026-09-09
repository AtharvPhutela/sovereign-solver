#!/usr/bin/env python3
"""Tests for the sovereignty check -- Build Map ticket #1.

The check is itself a piece of infrastructure the whole project's compliance
claim rests on, so it gets the same treatment as a numerical routine: it is not
enough that it returns "clean", it must be shown to go red on a real leak. Most
of these tests are therefore negative tests.

Run:  python3 tools/test_sovereignty_check.py
"""

from __future__ import annotations

import io
import contextlib
import tempfile
import textwrap
import unittest
from pathlib import Path

import sovereignty_check as sc

REPO_ROOT = Path(__file__).resolve().parent.parent

MINIMAL_POLICY = """\
schema_version = 1

[meta]
ledger = "DEPENDENCY_LEDGER.md"

[scan]
include_globs = [
  "**/CMakeLists.txt", "CMakeLists.txt", "**/*.cmake",
  "**/*.py", "**/*.cpp", "**/*.cu", "**/*.h", "**/*.hpp",
  "requirements.txt",
]
exclude_globs = [".git/**"]
vendor_dirs = ["third_party"]

[[forbidden]]
name = "scip"
display = "SCIP"
aliases = ["pyscipopt"]
category = "milp-solver"
reason = "we build the tree ourselves"

[[forbidden]]
name = "cbc"
display = "COIN-OR CBC"
category = "milp-solver"
reason = "branch-and-cut is ticket #38"

[[permitted]]
name = "cublas"
display = "cuBLAS"
category = "gpu-blas"
reason = "linear algebra"

[[restricted]]
name = "metis"
display = "METIS"
category = "graph-partitioner"
allowed_scopes = ["src/l0/**"]
reason = "ordering yes, structure detection no"
"""


class TokenisationTests(unittest.TestCase):
    """The matcher's whole justification is that grep gets these wrong."""

    def test_scipy_is_not_scip(self):
        # The canonical false positive. A substring grep fails here.
        self.assertNotIn("scip", sc.name_candidates("scipy"))
        self.assertIn("scipy", sc.name_candidates("scipy"))

    def test_versioned_soname(self):
        self.assertIn("scip", sc.name_candidates("libscip.so.8"))

    def test_linker_flag(self):
        # A substring grep for "scip" finds this, but for "cbc" would report
        # the flag text "lcbc" -- we resolve it properly instead.
        self.assertIn("scip", sc.name_candidates("-lscip"))
        self.assertIn("cbc", sc.name_candidates("-lcbc"))

    def test_distro_package_name(self):
        self.assertIn("cbc", sc.name_candidates("coinor-libcbc-dev"))

    def test_cmake_variable(self):
        self.assertIn("scip", sc.name_candidates("SCIP_DIR"))
        self.assertIn("cbc", sc.name_candidates("CBC_ROOT"))

    def test_path_component(self):
        self.assertIn("bliss", sc.name_candidates("third_party/bliss/graph.hh"))

    def test_configure_flag(self):
        self.assertIn("scip", sc.name_candidates("--with-scip"))

    def test_extensions_are_not_names(self):
        cands = sc.name_candidates("solver.cpp")
        self.assertIn("solver", cands)
        self.assertNotIn("cpp", cands)


class GlobTests(unittest.TestCase):
    def test_double_star_matches_nested_and_root(self):
        rx = sc.glob_to_regex("**/CMakeLists.txt")
        self.assertTrue(rx.match("CMakeLists.txt"))
        self.assertTrue(rx.match("src/l0/CMakeLists.txt"))

    def test_single_star_does_not_cross_slash(self):
        rx = sc.glob_to_regex("src/*.cpp")
        self.assertTrue(rx.match("src/a.cpp"))
        self.assertFalse(rx.match("src/deep/a.cpp"))

    def test_prefix_scope(self):
        rx = sc.glob_to_regex("src/l0/**")
        self.assertTrue(rx.match("src/l0/ordering/metis_order.cpp"))
        self.assertFalse(rx.match("src/l4/tree.cpp"))


class PolicyValidationTests(unittest.TestCase):
    """A broken policy must be an error, never a silent pass."""

    def _load(self, body: str):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sovereignty.toml"
            p.write_text(body)
            return sc.load_policy(p)

    def test_valid_policy_loads(self):
        policy = self._load(MINIMAL_POLICY)
        self.assertEqual(policy.lookup("scip").tier, "forbidden")
        self.assertEqual(policy.lookup("pyscipopt").name, "scip")
        self.assertEqual(policy.lookup("cublas").tier, "permitted")

    def test_entry_without_reason_is_rejected(self):
        body = MINIMAL_POLICY.replace('reason = "we build the tree ourselves"', 'reason = ""')
        with self.assertRaises(sc.PolicyError) as cm:
            self._load(body)
        self.assertIn("justified in writing", str(cm.exception))

    def test_restricted_without_scope_is_rejected(self):
        body = MINIMAL_POLICY.replace('allowed_scopes = ["src/l0/**"]\n', "")
        with self.assertRaises(sc.PolicyError):
            self._load(body)

    def test_alias_claimed_by_two_tiers_is_rejected(self):
        body = MINIMAL_POLICY + textwrap.dedent("""
            [[permitted]]
            name = "pyscipopt"
            display = "collision"
            reason = "should collide with the scip alias"
        """)
        with self.assertRaises(sc.PolicyError) as cm:
            self._load(body)
        self.assertIn("claimed by both", str(cm.exception))

    def test_wrong_schema_version_is_rejected(self):
        with self.assertRaises(sc.PolicyError):
            self._load(MINIMAL_POLICY.replace("schema_version = 1", "schema_version = 99"))

    def test_unjustified_exception_is_rejected(self):
        body = MINIMAL_POLICY + '\n[[exceptions]]\ntoken = "cbc"\n'
        with self.assertRaises(sc.PolicyError) as cm:
            self._load(body)
        self.assertIn("silent suppression", str(cm.exception))


class EndToEndTests(unittest.TestCase):
    """Full runs against a throwaway repo. These are the tests that matter."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        (self.root / "sovereignty.toml").write_text(MINIMAL_POLICY)
        (self.root / "DEPENDENCY_LEDGER.md").write_text(
            "SCIP, COIN-OR CBC, cuBLAS and METIS are all classified here."
        )
        self.addCleanup(self._td.cleanup)

    def run_check(self, *extra: str) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = sc.main(["--root", str(self.root), *extra])
        return code, buf.getvalue()

    def write(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))

    # -- the happy path ---------------------------------------------------

    def test_clean_tree_passes(self):
        self.write("CMakeLists.txt", """
            project(solver)
            find_package(CUDAToolkit REQUIRED)
            target_link_libraries(solver PRIVATE CUDA::cublas)
        """)
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_OK, out)
        self.assertIn("OK", out)

    def test_scipy_does_not_trip_the_check(self):
        # Regression guard for the false positive that would get this check
        # switched off in week two.
        self.write("tools/harness.py", "import scipy.sparse as sp\n")
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_OK, out)

    # -- the negative tests: a real leak must go red ----------------------

    def test_forbidden_in_cmake_manifest_fails(self):
        self.write("CMakeLists.txt", """
            find_package(SCIP REQUIRED)
            target_link_libraries(solver PRIVATE ${SCIP_LIBRARIES})
        """)
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)
        self.assertIn("FORBIDDEN", out)
        self.assertIn("SCIP", out)

    def test_forbidden_in_link_flags_fails(self):
        self.write("CMakeLists.txt", 'target_link_libraries(solver PRIVATE -lcbc)\n')
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)
        self.assertIn("COIN-OR CBC", out)

    def test_forbidden_in_python_import_fails(self):
        self.write("src/bind.py", "import pyscipopt\n")
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)

    def test_forbidden_in_cpp_include_fails(self):
        # Regression: an early version treated '#' as a comment marker in every
        # file type, which silently blinded the check to #include directives --
        # the most likely way a forbidden library actually enters a C++ build.
        self.write("src/l4/tree.cpp", '#include "scip/scip.h"\n')
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)
        self.assertIn("SCIP", out)

    def test_angle_bracket_include_fails(self):
        self.write("src/l4/tree.cu", "#include <cbc/CbcModel.hpp>\n")
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)

    def test_cpp_comment_discussing_a_forbidden_library_is_allowed(self):
        # "Studied, never linked" (Bible SS1.2) means prose about nauty/SCIP is
        # expected in this codebase and must not trip the check.
        self.write("src/l4/symmetry.cpp",
                   "// Individualization-Refinement, reimplemented from scratch;\n"
                   "// see SCIP's symmetry handling for the design we do NOT link.\n"
                   "int refine();\n")
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_OK, out)

    def test_python_comment_is_still_a_comment(self):
        self.write("tools/notes.py", "# we benchmark against cbc as an oracle\n")
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_OK, out)

    def _policy_with_bliss(self) -> None:
        body = MINIMAL_POLICY + textwrap.dedent("""
            [[forbidden]]
            name = "bliss"
            display = "bliss"
            category = "graph-automorphism"
            reason = "ticket #41 reimplements IR"
        """)
        (self.root / "sovereignty.toml").write_text(body)

    def test_vendored_source_drop_fails(self):
        # No manifest mentions it; someone just copied the tree in.
        (self.root / "third_party" / "bliss").mkdir(parents=True)
        self._policy_with_bliss()
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)
        self.assertIn("vendored source", out)

    def test_vendored_single_file_drop_fails(self):
        # The narrower gap: not a whole tree, one file named after the library,
        # dropped into a generic vendor dir, with content that never names it.
        (self.root / "third_party").mkdir()
        (self.root / "third_party" / "cbc_solver.cpp").write_text("int solve() { return 0; }\n")
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)
        self.assertIn("vendored source", out)
        self.assertIn("COIN-OR CBC", out)

    def test_vendored_versioned_soname_file_fails(self):
        (self.root / "extern").mkdir()
        # os.walk gives us the filename; multi-dot stem handling must still resolve it.
        (self.root / "extern" / "libglpk.so.40.4.0").write_text("")
        body = MINIMAL_POLICY + textwrap.dedent("""
            [[forbidden]]
            name = "glpk"
            display = "GLPK"
            category = "lp-milp-solver"
            reason = "complete substitute for the spine"
        """)
        (self.root / "sovereignty.toml").write_text(body)
        # extern/ is a recognised vendor dir in the real policy; add it here too.
        (self.root / "sovereignty.toml").write_text(
            (self.root / "sovereignty.toml").read_text().replace(
                'vendor_dirs = ["third_party"]', 'vendor_dirs = ["third_party", "extern"]'))
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)
        self.assertIn("GLPK", out)

    def test_vendored_benign_file_in_vendor_dir_is_fine(self):
        (self.root / "third_party").mkdir()
        (self.root / "third_party" / "README.md").write_text("bundled headers live here")
        (self.root / "third_party" / "our_helper.cpp").write_text("int help() { return 1; }\n")
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_OK, out)

    # -- the restricted tier ----------------------------------------------

    def test_restricted_inside_scope_passes(self):
        self.write("src/l0/ordering.cpp", "// nested dissection via metis\nint metis_order();\n")
        code, out = self.run_check("-v")
        self.assertEqual(code, sc.EXIT_OK, out)

    def test_restricted_outside_scope_fails(self):
        # METIS used for structure detection (ticket #14's job) rather than
        # fill-reducing ordering -- exactly the leak the tier exists to catch.
        self.write("src/l3/structure_detect.cpp", "int partition() { return metis_partition(); }\n")
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)
        self.assertIn("OUTSIDE its allowed scope", out)

    # -- exceptions --------------------------------------------------------

    def test_justified_exception_suppresses(self):
        self.write("src/crypto.cpp", "int mode = AES_CBC_MODE;\n")
        code, _ = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)  # noisy by default...

        body = MINIMAL_POLICY + textwrap.dedent("""
            [[exceptions]]
            token = "cbc"
            paths = ["src/crypto.cpp"]
            reason = "AES-CBC cipher mode, unrelated to COIN-OR CBC"
        """)
        (self.root / "sovereignty.toml").write_text(body)
        code, out = self.run_check()
        self.assertEqual(code, sc.EXIT_OK, out)  # ...quiet once justified

    def test_exception_is_path_scoped(self):
        body = MINIMAL_POLICY + textwrap.dedent("""
            [[exceptions]]
            token = "cbc"
            paths = ["src/crypto.cpp"]
            reason = "AES-CBC cipher mode"
        """)
        (self.root / "sovereignty.toml").write_text(body)
        self.write("CMakeLists.txt", "target_link_libraries(solver PRIVATE cbc)\n")
        code, _ = self.run_check()
        self.assertEqual(code, sc.EXIT_VIOLATION)

    # -- dependency graph --------------------------------------------------

    def test_depgraph_catches_what_the_manifest_hides(self):
        # The manifest is clean; a transitively-resolved target drags CBC onto
        # the link line. Manifest-only checking would call this a pass.
        self.write("CMakeLists.txt", "project(solver)\n")
        bd = self.root / "build"
        bd.mkdir()
        (bd / "CMakeCache.txt").write_text("CBC_LIBRARY:FILEPATH=/usr/lib/libcbc.so\n")
        code, out = self.run_check("--cmake-build-dir", "build")
        self.assertEqual(code, sc.EXIT_VIOLATION)
        self.assertIn("depgraph", out)

    def test_cmake_private_cache_entries_are_ignored(self):
        # Regression: find_package(Python3) writes CMake-private bookkeeping
        # into CMakeCache.txt, including a NumPy "reason failure" entry that
        # records NumPy was NOT found. Reading those as declarations made a
        # bare `cmake --build` fail on a project with no dependencies at all.
        self.write("CMakeLists.txt", "project(solver)\n")
        bd = self.root / "build"
        bd.mkdir()
        (bd / "CMakeCache.txt").write_text(
            "//METIS reason failure\n"
            "_Python3_METIS_REASON_FAILURE:INTERNAL=\n"
            "_Python3_INTERPRETER_SIGNATURE:INTERNAL=abc123\n"
        )
        code, out = self.run_check("--cmake-build-dir", "build")
        self.assertEqual(code, sc.EXIT_OK, out)

    def test_real_cache_entry_still_caught(self):
        # The private-entry filter must not become a blanket amnesty: a genuine,
        # non-underscore cache variable naming a forbidden library still fails.
        self.write("CMakeLists.txt", "project(solver)\n")
        bd = self.root / "build"
        bd.mkdir()
        (bd / "CMakeCache.txt").write_text("SCIP_LIBRARY:FILEPATH=/usr/lib/libscip.so\n")
        code, out = self.run_check("--cmake-build-dir", "build")
        self.assertEqual(code, sc.EXIT_VIOLATION, out)

    def test_missing_build_dir_is_not_an_error(self):
        self.write("CMakeLists.txt", "project(solver)\n")
        code, _ = self.run_check("--cmake-build-dir", "nonexistent")
        self.assertEqual(code, sc.EXIT_OK)

    # -- ledger drift ------------------------------------------------------

    def test_ledger_drift_is_caught(self):
        self.write("CMakeLists.txt", "project(solver)\n")
        (self.root / "DEPENDENCY_LEDGER.md").write_text("only SCIP and cuBLAS are written up here")
        code, out = self.run_check("--check-ledger")
        self.assertEqual(code, sc.EXIT_VIOLATION)
        self.assertIn("drifted", out)


class RealRepositoryTests(unittest.TestCase):
    """The check must pass on this repository, and the ledger must be current."""

    def test_repo_is_clean_and_ledger_is_current(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = sc.main(["--root", str(REPO_ROOT), "--check-ledger"])
        self.assertEqual(code, sc.EXIT_OK, buf.getvalue())

    def test_real_policy_classifies_the_known_traps(self):
        policy = sc.load_policy(REPO_ROOT / "sovereignty.toml")
        for name in ("cbc", "clp", "highs", "scip", "glpk", "cuopt", "gurobi",
                     "nauty", "bliss", "saucy", "gcg", "kahypar"):
            with self.subTest(name=name):
                self.assertEqual(policy.lookup(name).tier, "forbidden")
        for name in ("cublas", "cusparse", "cudss", "openblas", "suitesparse", "mumps"):
            with self.subTest(name=name):
                self.assertEqual(policy.lookup(name).tier, "permitted")
        for name in ("metis", "scipy", "pulp"):
            with self.subTest(name=name):
                self.assertEqual(policy.lookup(name).tier, "restricted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
