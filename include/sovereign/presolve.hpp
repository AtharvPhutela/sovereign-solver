// Lightweight dual-preserving presolve -- Build Map ticket #13, gate M3.
// Bible S4.4 (L3).
//
// SCOPE (this ticket -- see HANDOFF S23 for the full derivation). Every
// reduction implemented here removes a row or column via a move whose
// postsolve rule is provably correct against the ORIGINAL problem's KKT
// conditions, not just plausible:
//   - FixedColumnMove: a column pinned to a single value (lo == hi, whether
//     that came from the model itself or from a SingletonRowMove collapsing
//     its range to a point) is substituted out, folding its contribution
//     into every row it still touched.
//   - EmptyColumnMove: a column touching no active row is resolved directly
//     from its own bounds and objective sign -- it never had a row's dual
//     to account for.
//   - RedundantRowMove: a row whose activity range (from bounds active at
//     the time, none of which this row itself produced) already sits
//     inside its own bounds is removed with a permanently-safe dual of 0.
//   - SingletonRowMove: a row with exactly one active nonzero coefficient
//     tightens that column's bound to what the row implies and is removed;
//     postsolve attributes the row's dual to the reduced cost ONLY when the
//     final point actually sits at the bound THIS row produced (see the
//     .cpp for the derivation of why an unconditional dual of 0 is wrong
//     here, unlike RedundantRowMove).
//
// General, non-removing activity-based bound tightening and multi-variable
// "forcing row" elimination (pinning several columns from one row at once)
// are deliberately NOT attempted here. Both need the SAME kind of dual-
// redistribution reasoning generalized across a longer dependency chain,
// and getting that wrong is exactly this ticket's own "Watch out": a
// silently-slightly-wrong dual that Benders (#26) and conflict learning
// (#43) would inherit without ever seeing an error. Named follow-up, not
// half-attempted -- the same discipline ticket #4 applied to sparse LU.
//
// POSTSOLVE ARCHITECTURE: a trail (Andersen & Andersen's own standard
// design, re-derived here, not copied). Every reduction is recorded, in the
// order APPLIED, as one PresolveMove. Postsolve replays the trail in
// REVERSE (LIFO): by the time any move's undo runs, every row/column it
// references was either never removed, or removed by a move LATER in
// forward order -- which, in reverse, has already been undone. This is
// exactly what lets each move's postsolve rule reason only about the single
// step it represents, never the whole pipeline at once.
#pragma once

#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/problem.hpp"
#include "sovereign/status.hpp"

namespace sov {

enum class PresolveStatus { Reduced, Infeasible, Error };

const char* to_string(PresolveStatus s) noexcept;

struct PresolveOptions {
    int max_rounds = 100;      ///< fixed-point iteration cap
    Real tolerance = 1e-9;
    bool verbose = false;
};

/// An originally- (or tightened-to-) fixed column substituted out by folding
/// its contribution into every row it still touched. `rows` lists only rows
/// that were ACTIVE at the moment this move was applied -- see the file
/// comment on why that is exactly what postsolve's reverse replay needs.
struct FixedColumnMove {
    Idx col = 0;
    Real value = 0.0;
    Real obj_coeff = 0.0;
    std::vector<std::pair<Idx, Real>> rows;   ///< (row, coefficient)
};

/// A column touching no active row, resolved directly from its own bounds
/// and objective sign -- it never has a row's dual to account for.
struct EmptyColumnMove {
    Idx col = 0;
    Real value = 0.0;
    Real obj_coeff = 0.0;
};

/// A row that can never bind, given whatever bounds are active -- none of
/// them this row's own doing -- at the time it is checked. Dual is always
/// exactly 0 in postsolve.
struct RedundantRowMove {
    Idx row = 0;
};

/// A row with exactly one active nonzero coefficient: that column's bound
/// is tightened to what the row implies and the row is removed. Both the
/// OLD (pre-this-row) and NEW (post-tightening) bounds are kept so
/// postsolve can tell whether the final point is actually attributable to
/// this row, or merely coincides with an unrelated bound.
struct SingletonRowMove {
    Idx row = 0;
    Idx col = 0;
    Real coeff = 0.0;
    Real old_lower = 0.0, old_upper = 0.0;
    Real new_lower = 0.0, new_upper = 0.0;
};

using PresolveMove =
    std::variant<FixedColumnMove, EmptyColumnMove, RedundantRowMove, SingletonRowMove>;

struct PresolveResult {
    PresolveStatus status = PresolveStatus::Error;
    Problem reduced;                  ///< meaningful only when status == Reduced

    Idx original_num_rows = 0, original_num_cols = 0;
    std::vector<Idx> row_map;         ///< reduced row index -> original row index
    std::vector<Idx> col_map;         ///< reduced col index -> original col index
    std::vector<PresolveMove> trail;  ///< in APPLIED order; postsolve replays in reverse
    Real tolerance = 1e-9;            ///< carried for postsolve's own bound comparisons

    Idx rows_removed = 0, cols_removed = 0, nonzeros_removed = 0;
    std::string message;
};

struct PostsolveResult {
    std::vector<Real> primal;          ///< length original_num_cols
    std::vector<Real> dual;            ///< length original_num_rows
    std::vector<Real> reduced_costs;   ///< length original_num_cols
};

/// Reduces `problem` to a fixed point of the move set above, or detects
/// infeasibility. See the file comment for exactly which reductions this
/// covers and which are deliberately out of scope.
PresolveResult presolve(const Problem& problem, const PresolveOptions& options = {});

/// Replays `result.trail` in reverse against a solution of `result.reduced`
/// to reconstruct primal, dual, and reduced costs for the ORIGINAL problem
/// `presolve()` was called on. `x_reduced`/`y_reduced`/`rc_reduced` must be
/// in the problem's own objective sense (the same convention Simplex/PDHG/
/// IPM already report in) and sized to `result.reduced`'s own shape.
PostsolveResult postsolve(const PresolveResult& result, std::span<const Real> x_reduced,
                          std::span<const Real> y_reduced, std::span<const Real> rc_reduced);

}  // namespace sov
