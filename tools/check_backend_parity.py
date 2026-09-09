#!/usr/bin/env python3
"""Backend interface parity check -- Build Map ticket #3, gate M0.

The problem this exists to solve: on a machine with no CUDA toolkit and no
ROCm, `backend_cuda.cu` and `backend_hip.cpp` are never compiled. Add a method
to the Backend interface and the compiler will tell you the Host backend is
incomplete -- and say nothing at all about the two GPU backends, which quietly
fall behind until someone builds on a GPU machine weeks later.

Interface drift is the one class of GPU-backend defect that IS detectable
without a toolkit, so it gets checked here. This is not a substitute for
compiling those files; it is the part of their correctness that can be
verified from a laptop with no GPU.

Reports, per backend source:
  * pure-virtual methods of Backend with no override        -> failure
  * methods marked Unsupported / NotImplemented              -> reported, allowed
  * overrides naming a method the interface no longer has    -> failure

Exit 0 clean, 1 on drift, 2 on usage error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HEADER = REPO / "include" / "sovereign" / "backend.hpp"
SOURCES = {
    "host": REPO / "src" / "l0" / "backend_host.cpp",
    "cuda": REPO / "src" / "l0" / "backend_cuda.cu",
    "hip": REPO / "src" / "l0" / "backend_hip.cpp",
}

# `virtual <ret> name(...) = 0;` -- the methods every backend must implement.
PURE_VIRTUAL_RE = re.compile(
    r"virtual\s+[\w:<>,\s&*]+?\b(\w+)\s*\([^;]*?\)\s*(?:const\s*)?(?:noexcept\s*)?=\s*0\s*;",
    re.DOTALL)

# `virtual <ret> name(...) { ... }` -- defaulted, optional for a backend.
DEFAULTED_RE = re.compile(
    r"virtual\s+[\w:<>,\s&*]+?\b(\w+)\s*\([^;{]*?\)\s*(?:const\s*)?(?:noexcept\s*)?\{",
    re.DOTALL)

# The negative lookbehind on '~' keeps destructors out: `~CudaBackend() override`
# is an override, but not of anything in the method surface being compared.
OVERRIDE_RE = re.compile(
    r"(?<!~)\b(\w+)\s*\([^;{]*?\)\s*(?:const\s*)?(?:noexcept\s*)?override\b",
    re.DOTALL)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def interface_methods() -> tuple[set[str], set[str]]:
    if not HEADER.is_file():
        print(f"error: interface header not found: {HEADER}", file=sys.stderr)
        sys.exit(2)
    src = strip_comments(HEADER.read_text())
    return set(PURE_VIRTUAL_RE.findall(src)), set(DEFAULTED_RE.findall(src))


def overrides_in(path: Path) -> tuple[set[str], set[str]]:
    """Return (all overrides, those that decline with Unsupported/NotImplemented)."""
    src = strip_comments(path.read_text())
    names = set(OVERRIDE_RE.findall(src))

    declined: set[str] = set()
    for name in names:
        # Look at the body following each override of this name.
        for m in re.finditer(re.escape(name) + r"\s*\([^;{]*?\)\s*(?:const\s*)?"
                             r"(?:noexcept\s*)?override\s*\{", src, re.DOTALL):
            body = src[m.end(): m.end() + 400]
            if "Status::Unsupported" in body or "Status::NotImplemented" in body:
                declined.add(name)
    return names, declined


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Backend interface parity (ticket #3)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    pure, defaulted = interface_methods()
    if not pure:
        print("error: parsed zero pure-virtual methods from backend.hpp -- "
              "the regex and the header have diverged", file=sys.stderr)
        return 2

    print(f"Backend interface: {len(pure)} required, {len(defaulted)} defaulted")
    if args.verbose:
        print("  required : " + ", ".join(sorted(pure)))
        print("  defaulted: " + ", ".join(sorted(defaulted)))
    print()

    failures = 0
    for label, path in SOURCES.items():
        if not path.is_file():
            print(f"{label:<6} MISSING source file {path}")
            failures += 1
            continue

        names, declined = overrides_in(path)
        missing = sorted(pure - names)
        unknown = sorted(n for n in names if n not in pure and n not in defaulted)
        stubbed = sorted(declined)

        status = "ok" if not missing and not unknown else "DRIFT"
        print(f"{label:<6} [{status}]  {len(names & pure)}/{len(pure)} required implemented"
              + (f", {len(stubbed)} declined" if stubbed else ""))

        if missing:
            print(f"         missing override(s): {', '.join(missing)}")
            failures += 1
        if unknown:
            # An override of something the interface no longer declares would
            # not compile -- but only on a machine that compiles this file.
            print(f"         override(s) not in the interface: {', '.join(unknown)}")
            failures += 1
        if stubbed and args.verbose:
            print(f"         declines (Unsupported/NotImplemented): {', '.join(stubbed)}")

    print()
    if failures:
        print(f"FAIL: {failures} backend(s) have drifted from the interface.")
        print("Every backend must implement every pure-virtual method -- declining")
        print("with Status::Unsupported is fine, being absent is not.")
        return 1
    print("PASS: every backend implements the full interface.")
    print("NOTE: this checks the interface surface only. backend_cuda.cu and")
    print("      backend_hip.cpp are not compiled on a machine without their")
    print("      toolkits, and remain unverified until they are.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
