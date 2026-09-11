// Farkas / dual-ray & duals plumbing -- Build Map ticket #7, gate M1.
// Bible S4.2A.1 (Benders needs feasibility/optimality cuts from exactly this),
// S4.5.2 (conflict learning's Farkas-ray extraction consumes it too).
//
// PURPOSE. A clean, engine-agnostic contract for two things every continuous
// engine must be able to hand back: on an OPTIMAL solve, the row duals and
// column reduced costs; on an INFEASIBLE solve, a certificate that PROVES
// infeasibility rather than just asserting it. Simplex (#4) is the first
// producer; PDHG (#8) and the IPM (#9) implement the same contract later so
// Benders (#26) and conflict learning (#43) consume one interface, not one
// per engine.
//
// THE CERTIFICATE, DERIVED FROM SCRATCH (no borrowed Farkas-lemma code -- this
// follows directly from the two-sided-bound model in problem.hpp).
//
// The model is  row_lo <= A x <= row_hi,  col_lo <= x <= col_hi. For ANY real
// vector y (one entry per row) the algebraic identity
//
//     sum_i y_i (Ax)_i  ==  sum_j (A^T y)_j x_j
//
// holds for every x, feasible or not -- it is just reassociating a double sum.
// Call the left side S. For a FEASIBLE x, each (Ax)_i is confined to
// [row_lo_i, row_hi_i], so S is confined to the Minkowski sum, over i, of
// y_i * [row_lo_i, row_hi_i] -- call that the row-side range. Likewise each
// x_j is confined to [col_lo_j, col_hi_j], so S is ALSO confined to the
// Minkowski sum, over j, of d_j * [col_lo_j, col_hi_j] where d = A^T y -- the
// column-side range. Both ranges bound the same number S for any feasible x.
//
// If the two ranges do not overlap, no feasible x can exist: that is the
// certificate, and it is checkable by arithmetic on the ranges alone, without
// re-solving anything or trusting the engine that produced y.
//
// A revised simplex's Phase I dual is exactly such a y at an infeasible
// optimum (see src/l1/simplex.cpp) -- this header only defines the contract
// and the checker, the engines populate it.
#pragma once

#include <span>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/problem.hpp"

namespace sov {

/// A Farkas-style infeasibility certificate: one multiplier per row.
struct FarkasCertificate {
    std::vector<Real> row_multipliers;   ///< y, length num_rows(); empty = absent

    bool empty() const noexcept { return row_multipliers.empty(); }

    /// [lo, hi] the row bounds alone force on S = sum_i y_i (Ax)_i.
    struct Range {
        Real lo = -kInfinity;
        Real hi = kInfinity;
        bool disjoint_from(const Range& other, Real slack) const noexcept {
            return hi < other.lo - slack || other.hi < lo - slack;
        }
    };

    Range row_side_range(const Problem& problem) const;

    /// [lo, hi] the column bounds force on the same S, via d = A^T y.
    Range column_side_range(const Problem& problem) const;

    /// True iff the two ranges are disjoint by more than `slack` -- the actual
    /// proof. `slack` absorbs the floating-point error in the y that produced
    /// this certificate; it is not a modeling tolerance.
    bool certifies_infeasibility(const Problem& problem, Real slack = 1e-6) const;
};

/// Row duals and column reduced costs from a feasible/optimal solve, in the
/// problem's own objective sense (already sign-adjusted for Maximize).
struct DualSolution {
    std::vector<Real> row_duals;       ///< y, length num_rows()
    std::vector<Real> reduced_costs;   ///< c_j - A_j^T y, length num_cols()

    bool empty() const noexcept { return row_duals.empty() && reduced_costs.empty(); }

    /// Largest complementary-slackness violation over every row and column:
    /// a row dual must be zero unless the row is at one of its bounds, and a
    /// reduced cost must be zero unless the column is at one of its bounds.
    /// Zero (to tolerance) iff (x, *this) is a genuine primal-dual optimal
    /// pair, not merely two numbers an engine happened to report.
    Real complementary_slackness_violation(const Problem& problem, std::span<const Real> x,
                                           Real bound_tol = 1e-7) const;
};

}  // namespace sov
