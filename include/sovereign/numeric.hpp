// L0 numeric type policy -- Build Map ticket #3, gate M0. Bible S4.1, S6.3.
//
// One place decides what "a number" and "an index" mean for the whole solver.
// Everything above L0 spells its arithmetic in these names, so the
// mixed-precision work of S6.3 and a 64-bit index build are configuration
// changes rather than a sweep through the codebase.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace sov {

// --------------------------------------------------------------------------
// Scalar precision
//
// Baseline correctness is fp64 and stays fp64 (Bible S4.1). The narrower types
// are DECLARED here, not used: the mixed-precision iterative-refinement path
// (S6.3) needs a type vocabulary to grow into, and naming it now keeps the
// eventual fp32/fp16 tensor-core path from being a retrofit. Nothing in the
// solver computes in f32 or f16 today.
// --------------------------------------------------------------------------
using f64 = double;
using f32 = float;

/// The working precision of every algorithm at present.
using Real = f64;

enum class Precision : std::uint8_t {
    F64,   ///< baseline; the only precision any kernel currently runs in
    F32,   ///< reserved for the mixed-precision path (Bible S6.3)
    F16,   ///< reserved; tensor-core path, gated on conditioning estimates
};

constexpr const char* to_string(Precision p) noexcept {
    switch (p) {
        case Precision::F64: return "f64";
        case Precision::F32: return "f32";
        case Precision::F16: return "f16";
    }
    return "?";
}

constexpr std::size_t byte_width(Precision p) noexcept {
    switch (p) {
        case Precision::F64: return 8;
        case Precision::F32: return 4;
        case Precision::F16: return 2;
    }
    return 0;
}

// --------------------------------------------------------------------------
// Index width
//
// int32 is the default: it halves the bandwidth cost of the index arrays in an
// SpMV, which is memory-bound, and it is what cuSPARSE wants natively. But the
// problem statement asks for "thousands to millions of variables and
// constraints", and a large enough instance will exceed 2^31 nonzeros, so the
// width is a build policy rather than a hardcoded assumption.
//
//     cmake -DSOVEREIGN_INDEX64=ON
// --------------------------------------------------------------------------
#if defined(SOVEREIGN_INDEX64)
using Idx = std::int64_t;
inline constexpr bool kIndex64 = true;
#else
using Idx = std::int32_t;
inline constexpr bool kIndex64 = false;
#endif

inline constexpr Idx kIdxMax = std::numeric_limits<Idx>::max();

/// Largest nonzero count the current index width can address.
inline constexpr std::size_t max_addressable_nnz() noexcept {
    return static_cast<std::size_t>(kIdxMax);
}

// --------------------------------------------------------------------------
// Default tolerances
//
// These are the shared vocabulary, not per-algorithm settings: PDHG's native
// accuracy is around 1e-4 (Bible S4.2 Engine A) while the IPM is expected to
// drive far tighter, so each engine carries its own tolerance and only borrows
// these as defaults.
// --------------------------------------------------------------------------
namespace tol {

inline constexpr Real primal_feasibility = 1e-8;
inline constexpr Real dual_feasibility   = 1e-8;
inline constexpr Real relative_gap       = 1e-9;

/// Below this magnitude a stored coefficient is treated as structurally absent.
inline constexpr Real numeric_zero = 1e-14;

/// Comparison tolerance used when checking a computed result against an oracle.
inline constexpr Real oracle_relative = 1e-9;

}  // namespace tol

/// Relative-or-absolute closeness, the comparison used throughout the tests.
inline bool close(Real a, Real b, Real rel = tol::oracle_relative,
                  Real abs_tol = tol::numeric_zero) noexcept {
    const Real diff = a > b ? a - b : b - a;
    if (diff <= abs_tol) return true;
    const Real scale = (a < 0 ? -a : a) > (b < 0 ? -b : b) ? (a < 0 ? -a : a) : (b < 0 ? -b : b);
    return diff <= rel * scale;
}

}  // namespace sov
