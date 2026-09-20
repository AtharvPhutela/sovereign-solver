#!/bin/sh
# Build the HiGHS command-line executable for use as a black-box benchmark
# oracle -- Build Map ticket #2, gate M0.
#
# WHY THIS IS SOVEREIGNTY-COMPLIANT
# --------------------------------
# HiGHS is on the FORBIDDEN list in sovereignty.toml -- as a *library*. The
# Oracle rule (Bible S1.2, Part VII; docs/dependency-ledger.md S4) permits an
# established solver as a *testing oracle* provided it is reached only as a
# black-box CLI over files on disk: never linked, never imported, never on our
# solver's link line.
#
# This script enforces that boundary physically:
#   * HiGHS source and build tree live in  build-oracle/  -- OUTSIDE the tree the
#     sovereignty check and the CMake project ever look at (it is in
#     .gitignore and in sovereignty.toml's exclude_globs).
#   * Only the compiled `highs` executable is used, and only via subprocess by
#     tools/oracle/run_oracle.py.
#   * Nothing in src/ or the solver's CMake can see any of it.
#
# If build-oracle/ were deleted, the sovereignty check and every solver test
# still pass -- the oracle is scaffolding, not a dependency.
#
# Usage:   tools/oracle/install_highs.sh [VERSION_TAG]
# Default version tag is pinned below for reproducibility.

set -eu

HIGHS_TAG="${1:-v1.11.0}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORACLE_ROOT="${REPO_ROOT}/build-oracle"
SRC_DIR="${ORACLE_ROOT}/highs-src"
BUILD_DIR="${ORACLE_ROOT}/highs-build"
PREFIX="${ORACLE_ROOT}"          # installs bin/highs, lib/, include/ under here

echo "oracle: building HiGHS ${HIGHS_TAG}"
echo "  repo root : ${REPO_ROOT}"
echo "  oracle dir: ${ORACLE_ROOT}   (gitignored, outside the scanned tree)"

command -v cmake >/dev/null 2>&1 || { echo "cmake not found" >&2; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "git not found" >&2; exit 1; }

mkdir -p "${ORACLE_ROOT}"

if [ ! -d "${SRC_DIR}/.git" ]; then
  git clone --depth 1 --branch "${HIGHS_TAG}" \
      https://github.com/ERGO-Code/HiGHS.git "${SRC_DIR}"
else
  echo "  source already present at ${SRC_DIR}"
fi

# Shared libs OFF so the only artefact anyone could accidentally link is absent;
# we want exactly one thing out of this: the executable.
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G "$(command -v ninja >/dev/null 2>&1 && echo Ninja || echo 'Unix Makefiles')" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}"

cmake --build "${BUILD_DIR}" --target highs --parallel
cmake --install "${BUILD_DIR}" >/dev/null 2>&1 || cmake --build "${BUILD_DIR}" --target install

BIN="${PREFIX}/bin/highs"
if [ ! -x "${BIN}" ]; then
  # some HiGHS versions name the CLI target differently / place it elsewhere
  BIN="$(find "${BUILD_DIR}" -name highs -type f -perm -u+x | head -n1)"
fi
[ -x "${BIN}" ] || { echo "could not locate the built highs executable" >&2; exit 1; }

mkdir -p "${PREFIX}/bin"
[ "${BIN}" = "${PREFIX}/bin/highs" ] || cp "${BIN}" "${PREFIX}/bin/highs"

echo
echo "oracle ready: ${PREFIX}/bin/highs"
"${PREFIX}/bin/highs" --version || true
echo
echo "run_oracle.py will find it automatically. To use a different oracle,"
echo "set SOLVER_ORACLE=/path/to/solver."
