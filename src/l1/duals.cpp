// Farkas / dual-ray & duals plumbing -- Build Map ticket #7.
#include "sovereign/duals.hpp"

#include <algorithm>
#include <cmath>

#include "sovereign/status.hpp"

namespace sov {
namespace {

/// The range a single term `mult * [lo, hi]` can occupy. `mult == 0` collapses
/// to exactly {0} regardless of lo/hi -- the term vanishes identically for
/// every value in range, not just in the limit, so this is exact, not a
/// convention.
FarkasCertificate::Range term_range(Real mult, Real lo, Real hi) {
    if (mult == 0.0) return {0.0, 0.0};
    if (mult > 0.0) return {mult * lo, mult * hi};
    return {mult * hi, mult * lo};   // negative multiplier reverses the order
}

/// Minkowski-sum accumulation: lo += term.lo, hi += term.hi, with the usual
/// "infinity absorbs a finite shift" rule and the one case that must never
/// happen for a sound certificate: opposite infinities meeting (+inf + -inf),
/// which would silently manufacture a bound out of nothing. That can only
/// happen if the certificate itself is degenerate (a multiplier paired with a
/// one-sided-infinite bound on both accumulated sides at once); guard it by
/// making the accumulator NaN, which then never wins a disjointness check.
void accumulate(FarkasCertificate::Range& acc, const FarkasCertificate::Range& term) {
    acc.lo = acc.lo + term.lo;
    acc.hi = acc.hi + term.hi;
}

}  // namespace

FarkasCertificate::Range FarkasCertificate::row_side_range(const Problem& problem) const {
    Range acc{0.0, 0.0};
    const auto lo = problem.row_lower();
    const auto hi = problem.row_upper();
    for (std::size_t i = 0; i < row_multipliers.size(); ++i)
        accumulate(acc, term_range(row_multipliers[i], lo[i], hi[i]));
    return acc;
}

FarkasCertificate::Range FarkasCertificate::column_side_range(const Problem& problem) const {
    // d = A^T y: the same computation the simplex's own `column_dot` does,
    // expressed here directly against the CSR matrix so this header has no
    // dependency on any one engine's internal representation.
    std::vector<Real> d(static_cast<std::size_t>(problem.num_cols()), 0.0);
    const CsrMatrix& A = problem.matrix();
    const auto row_ptr = A.row_ptr();
    const auto col_idx = A.col_idx();
    const auto values = A.values();
    for (Idx r = 0; r < A.rows(); ++r) {
        const Real y = row_multipliers[static_cast<std::size_t>(r)];
        if (y == 0.0) continue;
        const Idx end = row_ptr[static_cast<std::size_t>(r) + 1];
        for (Idx k = row_ptr[static_cast<std::size_t>(r)]; k < end; ++k)
            d[static_cast<std::size_t>(col_idx[static_cast<std::size_t>(k)])]
                += y * values[static_cast<std::size_t>(k)];
    }

    Range acc{0.0, 0.0};
    const auto col_lo = problem.col_lower();
    const auto col_hi = problem.col_upper();
    for (std::size_t j = 0; j < d.size(); ++j)
        accumulate(acc, term_range(d[j], col_lo[j], col_hi[j]));
    return acc;
}

bool FarkasCertificate::certifies_infeasibility(const Problem& problem, Real slack) const {
    if (row_multipliers.size() != static_cast<std::size_t>(problem.num_rows())) return false;
    const Range row_range = row_side_range(problem);
    const Range col_range = column_side_range(problem);
    if (std::isnan(row_range.lo) || std::isnan(row_range.hi)
        || std::isnan(col_range.lo) || std::isnan(col_range.hi))
        return false;
    return row_range.disjoint_from(col_range, slack);
}

Real DualSolution::complementary_slackness_violation(const Problem& problem,
                                                      std::span<const Real> x,
                                                      Real bound_tol) const {
    Real worst = 0.0;

    // A row dual must be zero unless the row's activity sits at a bound.
    if (!row_duals.empty()) {
        std::vector<Real> activity(static_cast<std::size_t>(problem.num_rows()), 0.0);
        problem.matrix().multiply(x, activity);
        const auto lo = problem.row_lower();
        const auto hi = problem.row_upper();
        for (std::size_t i = 0; i < row_duals.size(); ++i) {
            const bool at_lower = is_finite_bound(lo[i]) && std::abs(activity[i] - lo[i]) <= bound_tol;
            const bool at_upper = is_finite_bound(hi[i]) && std::abs(activity[i] - hi[i]) <= bound_tol;
            if (!at_lower && !at_upper)
                worst = std::max(worst, std::abs(row_duals[i]));
        }
    }

    // A reduced cost must be zero unless the column sits at one of its bounds.
    if (!reduced_costs.empty()) {
        const auto lo = problem.col_lower();
        const auto hi = problem.col_upper();
        for (std::size_t j = 0; j < reduced_costs.size(); ++j) {
            const bool at_lower = is_finite_bound(lo[j]) && std::abs(x[j] - lo[j]) <= bound_tol;
            const bool at_upper = is_finite_bound(hi[j]) && std::abs(x[j] - hi[j]) <= bound_tol;
            if (!at_lower && !at_upper)
                worst = std::max(worst, std::abs(reduced_costs[j]));
        }
    }

    return worst;
}

}  // namespace sov
