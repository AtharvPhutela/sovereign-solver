#!/usr/bin/env python3
"""Sovereignty check -- Build Map ticket #1, gate M0.

Enforces the constraint that governs the whole project (Bible SS1.2):

    "It shall not be built upon any existing open source solver library but
     shall be built from scratch from mathematical foundation."

Reads the policy in ``sovereignty.toml`` and fails the build if a forbidden
solver library appears anywhere in the build manifests, the source tree, the
submodule list, the vendored-source directories, or -- when a configured CMake
build tree is supplied -- the resolved dependency graph.

Deliberately has zero third-party dependencies: a tool that guards our
dependency policy should not itself add dependencies. ``tomllib`` is stdlib
from Python 3.11.

Exit codes:
    0  clean
    1  violations found
    2  policy or usage error (a broken policy must never read as "clean")

Usage:
    tools/sovereignty_check.py
    tools/sovereignty_check.py --cmake-build-dir build
    tools/sovereignty_check.py --check-ledger
    tools/sovereignty_check.py --explain scip
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

EXIT_OK, EXIT_VIOLATION, EXIT_POLICY_ERROR = 0, 1, 2

# --------------------------------------------------------------------------
# Tokenisation
#
# The whole point of tokenising rather than substring-grepping: "scipy" must
# not match forbidden "scip", while "-lscip", "libscip.so.8", "SCIP_DIR" and
# "coinor-libcbc-dev" all must. Substring matching gets both of those wrong.
# --------------------------------------------------------------------------

# A run of characters that could plausibly name a dependency.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_+.\-/\\]+")

# Separators that always delimit distinct names (paths, versioned sonames).
_PATH_SPLIT_RE = re.compile(r"[/\\.:,]+")

# Separators inside a single identifier (CMake vars, package names).
_WORD_SPLIT_RE = re.compile(r"[_\-]+")

# Pieces that are file extensions or build noise, never a dependency name.
_NOISE = {
    "so", "a", "dll", "dylib", "lib", "h", "hpp", "hxx", "hh", "c", "cpp",
    "cc", "cxx", "cu", "cuh", "hip", "py", "pyi", "cmake", "txt", "json",
    "yml", "yaml", "toml", "pc", "in", "o", "obj", "md", "cfg", "ini", "sh",
}


def name_candidates(token: str) -> set[str]:
    """Every plausible dependency name a raw token could be naming."""
    raw = token.lower()
    seeds = {raw}

    # Linker flags: -lscip names scip, not "lscip".
    if raw.startswith("-l") and len(raw) > 2:
        seeds.add(raw[2:])

    out: set[str] = set()
    for seed in seeds:
        seed = seed.strip("-+._/\\")
        if not seed:
            continue
        out.add(seed)
        for part in _PATH_SPLIT_RE.split(seed):
            if not part:
                continue
            out.add(part)
            for word in _WORD_SPLIT_RE.split(part):
                if word:
                    out.add(word)

    # libfoo -> foo, so libscip.so.8 resolves to scip.
    out |= {c[3:] for c in out if c.startswith("lib") and len(c) > 3}

    # Drop leading/trailing punctuation left over, then the noise words.
    out = {c.strip("-+._") for c in out}
    return {c for c in out if c and c not in _NOISE}


# --------------------------------------------------------------------------
# Glob matching (fnmatch does not understand ``**``)
# --------------------------------------------------------------------------

def glob_to_regex(pattern: str) -> re.Pattern[str]:
    i, out = 0, []
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def any_match(patterns: list[re.Pattern[str]], relpath: str) -> bool:
    return any(p.match(relpath) for p in patterns)


def dir_is_excluded(exclude_globs: list[str], rel_dir: str) -> bool:
    """True if a directory (and thus its whole subtree) is excluded.

    A pattern like ``build-oracle/**`` is meant to exclude the directory too,
    not only its contents, so match the prefix directly.
    """
    for g in exclude_globs:
        prefix = g[:-3] if g.endswith("/**") else g
        prefix = prefix.rstrip("/")
        if "*" in prefix or "?" in prefix:
            if glob_to_regex(g).match(rel_dir):
                return True
        elif rel_dir == prefix or rel_dir.startswith(prefix + "/"):
            return True
    return False


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Entry:
    name: str
    tier: str                       # permitted | restricted | forbidden
    display: str
    reason: str
    category: str = ""
    aliases: tuple[str, ...] = ()
    allowed_scopes: tuple[str, ...] = ()

    @property
    def scope_regexes(self) -> list[re.Pattern[str]]:
        return [glob_to_regex(s) for s in self.allowed_scopes]


@dataclass
class Policy:
    entries: dict[str, Entry]                 # lookup key -> entry
    ordered: list[Entry]
    include: list[re.Pattern[str]]
    exclude: list[re.Pattern[str]]
    exclude_globs: list[str]                  # raw, for directory-prefix matching
    vendor_dirs: set[str]
    exceptions: list[dict]
    ledger_path: str

    def lookup(self, candidate: str) -> Entry | None:
        return self.entries.get(candidate)


class PolicyError(Exception):
    pass


def load_policy(path: Path) -> Policy:
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise PolicyError(f"policy file not found: {path}")
    except tomllib.TOMLDecodeError as exc:
        raise PolicyError(f"malformed policy file {path}: {exc}")

    if raw.get("schema_version") != 1:
        raise PolicyError(
            f"unsupported schema_version {raw.get('schema_version')!r}; this tool speaks version 1"
        )

    entries: dict[str, Entry] = {}
    ordered: list[Entry] = []

    for tier in ("permitted", "restricted", "forbidden"):
        for item in raw.get(tier, []):
            name = str(item.get("name", "")).strip().lower()
            if not name:
                raise PolicyError(f"[{tier}] entry with no name")

            # "Classified in writing" is the ticket's pass condition, so an
            # unjustified entry is a policy error, not a warning.
            reason = str(item.get("reason", "")).strip()
            if not reason:
                raise PolicyError(f"[{tier}] '{name}' has no reason -- every classification must be justified in writing")

            scopes = tuple(item.get("allowed_scopes", ()))
            if tier == "restricted" and not scopes:
                raise PolicyError(f"[restricted] '{name}' declares no allowed_scopes -- a restricted entry without a scope is just a permitted one")
            if tier != "restricted" and scopes:
                raise PolicyError(f"[{tier}] '{name}' declares allowed_scopes, which only the restricted tier honours")

            entry = Entry(
                name=name,
                tier=tier,
                display=str(item.get("display", name)),
                reason=reason,
                category=str(item.get("category", "")),
                aliases=tuple(str(a).lower() for a in item.get("aliases", ())),
                allowed_scopes=scopes,
            )
            ordered.append(entry)

            for key in (name, *entry.aliases):
                prior = entries.get(key)
                if prior is not None and prior.name != name:
                    raise PolicyError(
                        f"'{key}' is claimed by both '{prior.name}' ({prior.tier}) and '{name}' ({tier})"
                    )
                entries[key] = entry

    scan = raw.get("scan", {})
    exceptions = raw.get("exceptions", [])
    for exc in exceptions:
        if not str(exc.get("reason", "")).strip():
            raise PolicyError(f"exception {exc.get('token')!r} has no reason -- silent suppression is not allowed")

    return Policy(
        entries=entries,
        ordered=ordered,
        include=[glob_to_regex(g) for g in scan.get("include_globs", [])],
        exclude=[glob_to_regex(g) for g in scan.get("exclude_globs", [])],
        exclude_globs=list(scan.get("exclude_globs", [])),
        vendor_dirs=set(scan.get("vendor_dirs", [])),
        exceptions=exceptions,
        ledger_path=str(raw.get("meta", {}).get("ledger", "DEPENDENCY_LEDGER.md")),
    )


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------

@dataclass
class Finding:
    severity: str          # violation | note
    entry: Entry
    path: str
    line_no: int
    line: str
    detail: str
    source: str = "manifest"   # manifest | vendor | depgraph

    def render(self) -> str:
        loc = f"{self.path}:{self.line_no}" if self.line_no else self.path
        head = f"  {loc}\n    {self.detail}"
        if self.line:
            head += f"\n    > {self.line.strip()[:160]}"
        return head


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    permitted_hits: dict[str, int] = field(default_factory=dict)

    @property
    def violations(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "violation"]

    @property
    def notes(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "note"]


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

def excepted(policy: Policy, token: str, relpath: str) -> dict | None:
    """An exception matches if its token names the same thing the finding did.

    Comparison is by resolved policy entry, not raw string, so an exception
    keyed on the canonical name ("glpk") covers a hit on any of its aliases
    ("glpsol"), and vice versa. A token that resolves to no entry falls back to
    a plain case-insensitive match (covers collisions like AES_CBC_MODE).
    """
    tok = token.lower()
    hit_entry = policy.lookup(tok)
    for exc in policy.exceptions:
        etok = str(exc.get("token", "")).lower()
        exc_entry = policy.lookup(etok)
        same = (etok == tok) or (hit_entry is not None and exc_entry is hit_entry)
        if not same:
            continue
        paths = exc.get("paths")
        if not paths or any_match([glob_to_regex(p) for p in paths], relpath):
            return exc
    return None


def iter_scan_files(root: Path, policy: Policy):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".mypy_cache"}]
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        # Prune whole excluded subtrees so os.walk never descends a large
        # downloaded corpus or the out-of-tree oracle build.
        if rel_dir != "." and dir_is_excluded(policy.exclude_globs, rel_dir):
            dirnames[:] = []
            continue
        for fn in filenames:
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            if any_match(policy.exclude, rel):
                continue
            if not any_match(policy.include, rel):
                continue
            yield full, rel


# Comment markers are language-specific, and getting this wrong is dangerous in
# exactly one direction: in C-family files '#' opens a preprocessor directive,
# so treating it as a comment would blind the check to `#include <scip/scip.h>`
# -- the single most likely way a forbidden library actually enters a C++ build.
_C_FAMILY = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".cu", ".cuh", ".hip"}


def comment_prefixes(relpath: str) -> tuple[str, ...]:
    """Line prefixes that mark a comment in this file type.

    Comments are skipped because this project deliberately discusses forbidden
    libraries in prose -- the Bible's "studied, never linked" stance means
    source comments legitimately name nauty, GCG and friends.
    """
    name = Path(relpath).name
    suffix = Path(relpath).suffix.lower()
    if suffix in _C_FAMILY:
        return ("//", "/*", "*")
    if name == "CMakeCache.txt":
        # CMakeCache uses '//' for the doc comment above each entry, so the
        # default '#' rule would treat "//NumPy reason failure" as live text.
        return ("#", "//")
    return ("#",)


# CMake's own private cache bookkeeping: leading-underscore INTERNAL entries
# written by find_package machinery (e.g. _Python3_NumPy_REASON_FAILURE, which
# records that NumPy was *not* found). These describe CMake's search, not this
# project's dependencies, and must not be read as declarations.
_CMAKE_PRIVATE_CACHE_RE = re.compile(r"^_[A-Za-z0-9_]*:(INTERNAL|STATIC)=")


def strip_cmake_private_cache(text: str) -> str:
    """Blank CMake-private cache lines, preserving line numbering."""
    return "\n".join(
        "" if _CMAKE_PRIVATE_CACHE_RE.match(line) else line
        for line in text.splitlines()
    )


def scan_text(policy: Policy, rel: str, text: str, report: Report, source: str) -> None:
    prefixes = comment_prefixes(rel)
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(prefixes):
            continue
        for token in _TOKEN_RE.findall(line):
            for cand in name_candidates(token):
                entry = policy.lookup(cand)
                if entry is None:
                    continue

                if entry.tier == "permitted":
                    report.permitted_hits[entry.name] = report.permitted_hits.get(entry.name, 0) + 1
                    continue

                if excepted(policy, cand, rel):
                    continue

                if entry.tier == "forbidden":
                    report.findings.append(Finding(
                        severity="violation", entry=entry, path=rel, line_no=line_no,
                        line=line, source=source,
                        detail=(f"FORBIDDEN dependency '{entry.display}' ({entry.category}) "
                                f"referenced as '{token}'.\n    Why: {entry.reason}"),
                    ))
                elif entry.tier == "restricted":
                    if any_match(entry.scope_regexes, rel):
                        report.findings.append(Finding(
                            severity="note", entry=entry, path=rel, line_no=line_no,
                            line="", source=source,
                            detail=f"restricted '{entry.display}' used inside its allowed scope",
                        ))
                    else:
                        scopes = ", ".join(entry.allowed_scopes)
                        report.findings.append(Finding(
                            severity="violation", entry=entry, path=rel, line_no=line_no,
                            line=line, source=source,
                            detail=(f"RESTRICTED dependency '{entry.display}' used OUTSIDE its allowed "
                                    f"scope ({scopes}), referenced as '{token}'.\n    Why: {entry.reason}"),
                        ))


def scan_tree(policy: Policy, root: Path, report: Report) -> None:
    for full, rel in iter_scan_files(root, policy):
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"warning: could not read {rel}: {exc}", file=sys.stderr)
            continue
        report.files_scanned += 1
        scan_text(policy, rel, text, report, source="manifest")


def scan_vendor_dirs(policy: Policy, root: Path, report: Report) -> None:
    """Catch a raw source drop that no manifest mentions.

    Two shapes of drop: a whole tree (``third_party/highs/...``) and a single
    file named after the library (``third_party/cbc_solver.cpp``,
    ``extern/glpk.c``). Both are checked here by name, because a hand-copied
    file need not contain any token that self-identifies it -- an include guard
    or a bare function body would sail past the content scan.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirpath.split(os.sep):
            continue
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        # Honour exclude_globs here too: an out-of-tree oracle build
        # (build-oracle/) or a downloaded corpus may itself contain a vendored
        # third_party/ dir that is not this project's concern.
        if rel_dir != "." and dir_is_excluded(policy.exclude_globs, rel_dir):
            dirnames[:] = []
            continue
        if Path(dirpath).name not in policy.vendor_dirs:
            continue
        rel_parent = rel_dir

        def flag(display_name: str, cand: str, entry: Entry) -> None:
            rel = f"{rel_parent}/{display_name}"
            if excepted(policy, cand, rel):
                return
            report.findings.append(Finding(
                severity="violation", entry=entry, path=rel, line_no=0, line="",
                source="vendor",
                detail=(f"FORBIDDEN dependency '{entry.display}' appears as vendored source "
                        f"(named '{display_name}').\n    Why: {entry.reason}"),
            ))

        for name in (*dirnames, *filenames):
            # For a file, also try the extension-stripped stem: cbc_solver.cpp
            # -> {cbc_solver.cpp, cbc_solver, cbc, solver}. name_candidates
            # already strips extensions, but Path().stem covers multi-dot cases
            # like libglpk.so.40.
            probes = name_candidates(name) | name_candidates(Path(name).stem)
            for cand in probes:
                entry = policy.lookup(cand)
                if entry is not None and entry.tier == "forbidden":
                    flag(name, cand, entry)
                    break


def scan_cmake_build_dir(policy: Policy, build_dir: Path, report: Report,
                         repo_root: Path | None = None) -> None:
    """Scan the *resolved* dependency graph, not just declared intent.

    A manifest can be clean while a transitively-pulled target puts a forbidden
    library on the link line. CMakeCache.txt records what was actually found;
    link.txt / build.ninja record what is actually linked.

    First-party paths (references to this repo's own tree) are stripped before
    tokenizing: they are already covered by the direct source scan, and they
    would otherwise re-flag things like a custom target that shells out to
    tools/oracle/install_highs.sh. What is left is exactly the external
    references the dependency-graph pass exists to inspect.
    """
    targets = ["CMakeCache.txt", "build.ninja", "rules.ninja"]
    files: list[Path] = [build_dir / t for t in targets]
    files += list(build_dir.rglob("link.txt"))
    files += list(build_dir.rglob("*.cmake"))

    root_str = str(repo_root.resolve()) if repo_root else None

    for f in files:
        if not f.is_file():
            continue
        rel = f.relative_to(build_dir).as_posix()
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if f.name == "CMakeCache.txt":
            text = strip_cmake_private_cache(text)
        if root_str:
            text = _strip_first_party_from_depgraph(text, root_str, repo_root)
        report.files_scanned += 1
        scan_text(policy, f"<depgraph>/{rel}", text, report, source="depgraph")


def _strip_first_party_from_depgraph(text: str, root_str: str, repo_root: Path | None) -> str:
    """Remove this repo's own references from generated build files.

    The dependency-graph pass exists to catch an *external* forbidden library
    reaching the link line. A generated build file also restates our own build
    recipes -- a custom target that runs tools/oracle/install_highs.sh, a
    COMMENT string that says "HiGHS" -- and those are already covered by the
    direct source scan. Left in, they produce findings against our own
    first-party paths.

    So: drop absolute paths into the repo, drop pure-prose description lines,
    and blank any remaining token that names an existing first-party file.
    """
    out_lines = []
    for line in text.splitlines():
        stripped = line.replace(root_str + "/", " ").replace(root_str, " ")
        lead = stripped.lstrip()
        # CMake/Ninja human-readable metadata -- not dependency resolution.
        if lead.startswith(("DESC =", "COMMENT =", "description = ", "#")):
            out_lines.append("")
            continue
        if repo_root is not None:
            kept = []
            for tok in stripped.split(" "):
                cand = tok.strip().strip('"\'')
                if cand and "/" in cand and not cand.startswith("/"):
                    p = repo_root / cand
                    if p.exists() and p.is_file():
                        continue  # our own file, already scanned directly
                kept.append(tok)
            stripped = " ".join(kept)
        out_lines.append(stripped)
    return "\n".join(out_lines)


def check_ledger(policy: Policy, root: Path, report: Report) -> None:
    """The ledger is 'living' only if it cannot silently fall behind the policy."""
    ledger = root / policy.ledger_path
    if not ledger.is_file():
        report.findings.append(Finding(
            severity="violation",
            entry=Entry("ledger", "forbidden", policy.ledger_path, "missing"),
            path=policy.ledger_path, line_no=0, line="", source="manifest",
            detail=f"ledger {policy.ledger_path} is missing; ticket #1 requires the classification in writing",
        ))
        return

    text = ledger.read_text(encoding="utf-8").lower()
    for entry in policy.ordered:
        if entry.name not in text and entry.display.lower() not in text:
            report.findings.append(Finding(
                severity="violation", entry=entry, path=policy.ledger_path,
                line_no=0, line="", source="manifest",
                detail=(f"'{entry.display}' is classified {entry.tier} in sovereignty.toml "
                        f"but never appears in the ledger -- policy and ledger have drifted"),
            ))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def print_report(report: Report, policy: Policy, verbose: bool) -> None:
    print(f"sovereignty check -- {report.files_scanned} file(s) scanned, "
          f"{len(policy.ordered)} dependencies classified "
          f"({sum(1 for e in policy.ordered if e.tier == 'forbidden')} forbidden, "
          f"{sum(1 for e in policy.ordered if e.tier == 'restricted')} restricted, "
          f"{sum(1 for e in policy.ordered if e.tier == 'permitted')} permitted)")

    if verbose and report.permitted_hits:
        print("\npermitted dependencies referenced:")
        for name, count in sorted(report.permitted_hits.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<14} {count} reference(s)")

    if verbose and report.notes:
        print("\nrestricted dependencies, in scope (allowed):")
        for f in report.notes:
            print(f"  {f.path}:{f.line_no}  {f.detail}")

    violations = report.violations
    if violations:
        print(f"\nSOVEREIGNTY VIOLATION -- {len(violations)} finding(s):\n")
        for f in violations:
            print(f.render())
            print()
        print("Bible SS1.2: the solver 'shall not be built upon any existing open source")
        print("solver library but shall be built from scratch from mathematical foundation.'")
        print("Remove the dependency, or -- if this is a confirmed false positive -- add a")
        print("justified [[exceptions]] entry to sovereignty.toml.")
    else:
        print("\nOK -- no forbidden dependency in the build manifests, source tree, "
              "vendored sources, or dependency graph.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sovereignty dependency check (Build Map ticket #1)")
    parser.add_argument("--root", type=Path, default=None, help="repository root (default: parent of tools/)")
    parser.add_argument("--policy", type=Path, default=None, help="path to sovereignty.toml")
    parser.add_argument("--cmake-build-dir", type=Path, default=None,
                        help="configured CMake build tree, to scan the resolved dependency graph")
    parser.add_argument("--check-ledger", action="store_true",
                        help="also verify DEPENDENCY_LEDGER.md covers every classified dependency")
    parser.add_argument("--explain", metavar="NAME", help="print the classification of one dependency and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    policy_path = args.policy or (root / "sovereignty.toml")

    try:
        policy = load_policy(policy_path)
    except PolicyError as exc:
        print(f"policy error: {exc}", file=sys.stderr)
        return EXIT_POLICY_ERROR

    if args.explain:
        entry = policy.lookup(args.explain.strip().lower())
        if entry is None:
            print(f"'{args.explain}' is not classified in {policy_path.name}.")
            print("Unclassified is not the same as permitted -- classify it before adding it.")
            return EXIT_POLICY_ERROR
        print(f"{entry.display}  [{entry.tier.upper()}]  category={entry.category or 'n/a'}")
        if entry.aliases:
            print(f"  aliases: {', '.join(entry.aliases)}")
        if entry.allowed_scopes:
            print(f"  allowed scopes: {', '.join(entry.allowed_scopes)}")
        print(f"  reason: {entry.reason}")
        return EXIT_OK

    report = Report()
    scan_tree(policy, root, report)
    scan_vendor_dirs(policy, root, report)

    if args.cmake_build_dir:
        bd = args.cmake_build_dir if args.cmake_build_dir.is_absolute() else root / args.cmake_build_dir
        if bd.is_dir():
            scan_cmake_build_dir(policy, bd.resolve(), report, repo_root=root)
        elif args.verbose:
            print(f"note: build dir {bd} does not exist yet; skipping dependency-graph scan")

    if args.check_ledger:
        check_ledger(policy, root, report)

    print_report(report, policy, args.verbose)
    return EXIT_VIOLATION if report.violations else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
