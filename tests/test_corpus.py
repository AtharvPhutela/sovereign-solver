#!/usr/bin/env python3
"""Tests for the corpus manifest and fetcher -- Build Map ticket #2.

Network-free. The parsing routines are tested against fixed sample text; the
manifest is checked for internal consistency; the committed smoke set is checked
against its reference answers; and if a real download exists, its lock file is
verified.

Run:  python3 tests/test_corpus.py
"""

from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "benchmarks"
sys.path.insert(0, str(BENCH))
import fetch_corpus as fc  # noqa: E402


SAMPLE_README = """\
                       PROBLEM SUMMARY TABLE

Name       Rows   Cols   Nonzeros    Bytes  BR      Optimal Value

25FV47      822   1571    11127      70477        5.5018458883E+03
AFIRO        28     32       88        794       -4.6475314286E+02
DEGEN2      445    534     4449      24657       -1.4351780000E+03
QAP8        913   1632     8304 (see NOTES)       2.0350000000E+02
DFL001     6072  12230    41873     353192  B     1.12664E+07 **

BOUND-TYPE TABLE

some other stuff
"""

SAMPLE_INDEX = """\
<a href="25fv47">25fv47</a>
<a href="afiro">afiro</a>
<a href="emps.c">emps.c</a>
<a href="emps.exe.gz">emps.exe.gz</a>
<a href="readme">readme</a>
<a href="kennington/index.html">kennington</a>
<a href="index.html">index.html</a>
"""


class NetlibReadmeParsing(unittest.TestCase):
    def setUp(self):
        self.ref = fc.parse_netlib_reference(SAMPLE_README)

    def test_basic_rows_captured(self):
        self.assertIn("afiro", self.ref)
        self.assertEqual(self.ref["afiro"]["rows"], 28)
        self.assertEqual(self.ref["afiro"]["cols"], 32)
        self.assertEqual(self.ref["afiro"]["nonzeros"], 88)
        self.assertAlmostEqual(self.ref["afiro"]["optimal_value"], -464.75314286)

    def test_positive_and_scientific_values(self):
        self.assertAlmostEqual(self.ref["25fv47"]["optimal_value"], 5501.8458883)
        self.assertAlmostEqual(self.ref["degen2"]["optimal_value"], -1435.178)

    def test_stops_at_bound_type_table(self):
        # nothing after "BOUND-TYPE TABLE" should be parsed as an instance
        self.assertNotIn("some", self.ref)
        self.assertNotIn("bound-type", self.ref)

    def test_every_entry_has_a_source_attribution(self):
        for name, r in self.ref.items():
            self.assertIn("MINOS", r["optimal_source"], name)


class IndexScraping(unittest.TestCase):
    def test_tooling_files_excluded(self):
        names = fc.scrape_instance_names(
            SAMPLE_INDEX,
            ["emps.c", "emps.exe.gz", "readme", "kennington", "index.html"])
        self.assertEqual(set(names), {"25fv47", "afiro"})

    def test_anything_with_a_dot_is_treated_as_tooling(self):
        names = fc.scrape_instance_names('<a href="foo.tar.gz">x</a>', [])
        self.assertEqual(names, [])


class ManifestIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (BENCH / "corpus.toml").open("rb") as fh:
            cls.m = tomllib.load(fh)

    def test_schema_version(self):
        self.assertEqual(self.m["schema_version"], 1)

    def test_exactly_one_default_set_and_it_is_netlib(self):
        defaults = [n for n, c in self.m["sets"].items() if c.get("default")]
        self.assertEqual(defaults, ["netlib_lp"])

    def test_large_sets_are_deferred(self):
        for name in ("miplib2017", "mittelmann", "milpbench", "qplib"):
            self.assertTrue(self.m["sets"][name].get("deferred"), name)

    def test_every_set_has_a_licence_and_description(self):
        for name, cfg in self.m["sets"].items():
            self.assertTrue(cfg.get("description"), name)
            # a child set inherits provenance from its parent
            if "parent" not in cfg:
                self.assertTrue(cfg.get("licence"), name)

    def test_hard_subset_is_kept_separate(self):
        hard = self.m["sets"]["miplib2017_hard"]
        self.assertEqual(hard["parent"], "miplib2017")
        self.assertTrue(hard.get("keep_separate"))


class SmokeSet(unittest.TestCase):
    """The four committed instances and their hand-verified answers."""

    @classmethod
    def setUpClass(cls):
        cls.exp = tomllib.loads((BENCH / "smoke" / "expected.toml").read_text())

    def test_schema(self):
        self.assertEqual(self.exp["schema_version"], 1)

    def test_every_expected_instance_file_exists(self):
        for name, e in self.exp.items():
            if name == "schema_version":
                continue
            self.assertTrue((BENCH / "smoke" / e["file"]).is_file(), e["file"])

    def test_status_vocabulary(self):
        allowed = {"optimal", "infeasible", "unbounded"}
        for name, e in self.exp.items():
            if name == "schema_version":
                continue
            self.assertIn(e["status"], allowed, name)

    def test_optimal_instances_have_an_objective(self):
        for name, e in self.exp.items():
            if name == "schema_version":
                continue
            if e["status"] == "optimal":
                self.assertIn("objective", e, name)
                self.assertIn("solution", e, name)

    def test_coverage_of_the_four_cases(self):
        statuses = {e["status"] for k, e in self.exp.items() if k != "schema_version"}
        self.assertEqual(statuses, {"optimal", "infeasible", "unbounded"})
        tags = {t for k, e in self.exp.items() if k != "schema_version" for t in e.get("tags", [])}
        self.assertIn("degenerate", tags)

    def test_mps_files_are_wellformed_enough(self):
        for name, e in self.exp.items():
            if name == "schema_version":
                continue
            text = (BENCH / "smoke" / e["file"]).read_text()
            self.assertIn("ROWS", text)
            self.assertIn("COLUMNS", text)
            self.assertTrue(text.rstrip().endswith("ENDATA"), e["file"])


class DownloadedCorpus(unittest.TestCase):
    """Only runs if fetch_corpus.py has been run for real."""

    LOCK = BENCH / "data" / "netlib_lp" / "_lock.toml"

    @unittest.skipUnless(LOCK.is_file(), "netlib_lp not downloaded")
    def test_verify_passes(self):
        rc = fc.main(["--verify"])
        self.assertEqual(rc, 0)

    @unittest.skipUnless(LOCK.is_file(), "netlib_lp not downloaded")
    def test_reference_values_present_for_most_instances(self):
        lock = tomllib.loads(self.LOCK.read_text())["instances"]
        with_ref = [k for k, v in lock.items() if "optimal_value" in v]
        # the readme table covers essentially the whole classic set
        self.assertGreater(len(with_ref), 0.8 * len(lock))

    @unittest.skipUnless(LOCK.is_file(), "netlib_lp not downloaded")
    def test_afiro_reference_matches_known_value(self):
        ref = tomllib.loads((BENCH / "data" / "netlib_lp" / "_reference.toml").read_text())
        self.assertAlmostEqual(ref["afiro"]["optimal_value"], -464.75314286, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
