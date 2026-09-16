// Lightweight dual-preserving presolve -- Build Map ticket #13. See
// presolve.hpp for the full design (scope, the move types, and why the
// trail-replay architecture is what makes each move's postsolve rule only
// need to reason locally).
//
// THE SINGLETON-ROW DUAL FORMULA, derived here (presolve.hpp names the
// result; this is the "why"). A singleton row i ties one column j to a
// bound via coefficient a. Before row i is removed, the REDUCED problem's
// own solve reports a reduced cost rc_reduced[j] = c_j - sum_{k in reduced
// rows} a_kj y_k -- row i is simply absent from that sum. The TRUE original
// reduced cost is rc_true[j] = rc_reduced[j] - a * y_i. Two cases:
//   - x_j ends up at a bound STRICTLY TIGHTER than its pre-row-i bound (row
//     i is the actual reason x_j sits there): the ORIGINAL problem's wider
//     bound means x_j is NOT at any bound of the original column, so KKT
//     requires rc_true[j] = 0 exactly, which pins y_i = rc_reduced[j] / a.
//   - x_j does not sit at the bound row i produced (either it is interior
//     even to the tightened bound, or the tightened bound coincides with
//     the original one and some other reason binds it): row i contributed
//     nothing distinguishable, and y_i = 0 is safe.
// A RedundantRowMove needs no such case split: by construction its row was
// checked as non-binding using bounds it did not itself produce, so it is
// unconditionally never binding across the relevant region and y = 0 always
// satisfies complementary slackness for it.
#include "sovereign/presolve.hpp"

#include <algorithm>
#include <cmath>

#include "sovereign/sparse.hpp"

namespace sov {

const char* to_string(PresolveStatus s) noexcept {
    switch (s) {
        case PresolveStatus::Reduced:    return "reduced";
        case PresolveStatus::Infeasible: return "infeasible";
        case PresolveStatus::Error:      return "error";
    }
    return "unknown";
}

namespace {

/// The number of active nonzeros in the given row (over the whole row, no
/// early exit) and, when exactly one, that entry's (col, coeff).
struct RowScan {
    Idx active_count = 0;
    Idx only_col = -1;
    Real only_coeff = 0.0;
    Real min_activity = 0.0, max_activity = 0.0;
};

}  // namespace

// The whole reduction pipeline lives in one function: the state (bounds,
// active masks, running trail) is threaded through explicitly rather than
// split into a class, because every step needs all of it and nothing here
// is reused elsewhere. This is deliberately the sequential CPU correctness
// baseline (ticket #4's own precedent: build it correct and clear first,
// name the parallel/GPU version as follow-up rather than half-attempt it --
// Bible S4.4 wants this GPU-native eventually, but a naive port with an
// unverified dual story would be worse than an honest, correct CPU version).
PresolveResult presolve(const Problem& problem, const PresolveOptions& options) {
    PresolveResult result;
    result.original_num_rows = problem.num_rows();
    result.original_num_cols = problem.num_cols();
    result.tolerance = options.tolerance;

    std::string why;
    if (problem.validate(&why) != Status::Ok) {
        result.status = PresolveStatus::Error;
        result.message = "invalid problem: " + why;
        return result;
    }

    const Idx n = problem.num_cols();
    const Idx m = problem.num_rows();
    const Real tol = options.tolerance;

    const CsrMatrix& csr = problem.matrix();
    const CscMatrix csc = csr.to_csc();

    std::vector<Real> col_lower(problem.col_lower().begin(), problem.col_lower().end());
    std::vector<Real> col_upper(problem.col_upper().begin(), problem.col_upper().end());
    std::vector<Real> obj(problem.objective().begin(), problem.objective().end());
    std::vector<Real> row_lower(problem.row_lower().begin(), problem.row_lower().end());
    std::vector<Real> row_upper(problem.row_upper().begin(), problem.row_upper().end());
    std::vector<bool> col_active(static_cast<std::size_t>(n), true);
    std::vector<bool> row_active(static_cast<std::size_t>(m), true);

    std::vector<PresolveMove> trail;
    Real objective_constant_delta = 0.0;

    const bool minimize = problem.sense() == ObjSense::Minimize;

    auto crossed = [&](Real lo, Real hi) { return lo > hi + tol; };

    auto scan_row = [&](Idx i) -> RowScan {
        RowScan s;
        Real min_act = 0.0, max_act = 0.0;
        const auto begin = csr.row_ptr()[static_cast<std::size_t>(i)];
        const auto end = csr.row_ptr()[static_cast<std::size_t>(i) + 1];
        for (Idx k = begin; k < end; ++k) {
            const Idx j = csr.col_idx()[static_cast<std::size_t>(k)];
            if (!col_active[static_cast<std::size_t>(j)]) continue;
            const Real a = csr.values()[static_cast<std::size_t>(k)];
            ++s.active_count;
            s.only_col = j;
            s.only_coeff = a;

            const Real lo = col_lower[static_cast<std::size_t>(j)];
            const Real hi = col_upper[static_cast<std::size_t>(j)];
            if (a > 0.0) {
                min_act = is_finite_bound(lo) ? min_act + a * lo : -kInfinity;
                max_act = is_finite_bound(hi) ? max_act + a * hi : kInfinity;
            } else {
                min_act = is_finite_bound(hi) ? min_act + a * hi : -kInfinity;
                max_act = is_finite_bound(lo) ? max_act + a * lo : kInfinity;
            }
        }
        s.min_activity = min_act;
        s.max_activity = max_act;
        return s;
    };

    // Substitutes a fixed column (lo == hi == value) out of the model:
    // folds its contribution into every row it still touches and records
    // the move. Applies to both originally-fixed columns and ones a
    // SingletonRowMove tightened to a single point.
    auto fix_column = [&](Idx j, Real value) {
        FixedColumnMove move;
        move.col = j;
        move.value = value;
        move.obj_coeff = obj[static_cast<std::size_t>(j)];

        const auto begin = csc.col_ptr()[static_cast<std::size_t>(j)];
        const auto end = csc.col_ptr()[static_cast<std::size_t>(j) + 1];
        for (Idx k = begin; k < end; ++k) {
            const Idx i = csc.row_idx()[static_cast<std::size_t>(k)];
            if (!row_active[static_cast<std::size_t>(i)]) continue;
            const Real a = csc.values()[static_cast<std::size_t>(k)];
            move.rows.emplace_back(i, a);
            const auto ri = static_cast<std::size_t>(i);
            if (is_finite_bound(row_lower[ri])) row_lower[ri] -= a * value;
            if (is_finite_bound(row_upper[ri])) row_upper[ri] -= a * value;
        }

        objective_constant_delta += move.obj_coeff * value;
        col_active[static_cast<std::size_t>(j)] = false;
        trail.push_back(std::move(move));
    };

    bool infeasible = false;
    std::string infeasible_reason;

    for (int round = 0; round < options.max_rounds && !infeasible; ++round) {
        bool changed = false;

        // -- fixed columns (recurring: a SingletonRowMove below can create
        // a new one on any round) -----------------------------------------
        for (Idx j = 0; j < n && !infeasible; ++j) {
            if (!col_active[static_cast<std::size_t>(j)]) continue;
            const Real lo = col_lower[static_cast<std::size_t>(j)];
            const Real hi = col_upper[static_cast<std::size_t>(j)];
            if (crossed(lo, hi)) {
                infeasible = true;
                infeasible_reason = "column " + std::to_string(j) + " has crossed bounds after tightening";
                break;
            }
            if (is_finite_bound(lo) && is_finite_bound(hi) && std::abs(hi - lo) <= tol) {
                fix_column(j, lo);
                changed = true;
            }
        }
        if (infeasible) break;

        // -- empty columns (recurring: rows removed below can strand one)---
        for (Idx j = 0; j < n; ++j) {
            if (!col_active[static_cast<std::size_t>(j)]) continue;
            bool touches_active_row = false;
            const auto begin = csc.col_ptr()[static_cast<std::size_t>(j)];
            const auto end = csc.col_ptr()[static_cast<std::size_t>(j) + 1];
            for (Idx k = begin; k < end && !touches_active_row; ++k) {
                const Idx i = csc.row_idx()[static_cast<std::size_t>(k)];
                if (row_active[static_cast<std::size_t>(i)]) touches_active_row = true;
            }
            if (touches_active_row) continue;

            const Real c = obj[static_cast<std::size_t>(j)];
            const Real lo = col_lower[static_cast<std::size_t>(j)];
            const Real hi = col_upper[static_cast<std::size_t>(j)];
            const bool favor_lower = minimize ? (c > tol) : (c < -tol);
            const bool favor_upper = minimize ? (c < -tol) : (c > tol);

            Real value;
            bool resolvable = true;
            if (std::abs(c) <= tol) {
                if (is_finite_bound(lo)) value = lo;
                else if (is_finite_bound(hi)) value = hi;
                else value = 0.0;
            } else if (favor_lower && is_finite_bound(lo)) {
                value = lo;
            } else if (favor_upper && is_finite_bound(hi)) {
                value = hi;
            } else {
                // The favorable direction is unbounded -- true unboundedness
                // (or a case this scope doesn't try to prove either way).
                // Leave the column active rather than misclassify; the
                // solver downstream still sees it and handles it correctly.
                resolvable = false;
            }
            if (!resolvable) continue;

            EmptyColumnMove move{j, value, c};
            objective_constant_delta += c * value;
            col_active[static_cast<std::size_t>(j)] = false;
            trail.push_back(move);
            changed = true;
        }

        // -- rows: redundant / empty / singleton ----------------------------
        for (Idx i = 0; i < m && !infeasible; ++i) {
            if (!row_active[static_cast<std::size_t>(i)]) continue;
            const RowScan s = scan_row(i);
            const auto ri = static_cast<std::size_t>(i);
            const Real rlo = row_lower[ri], rhi = row_upper[ri];

            if (s.active_count == 0) {
                // An empty row's activity is always exactly 0; it is
                // satisfiable iff 0 lies in [rlo, rhi].
                if ((is_finite_bound(rlo) && rlo > tol) || (is_finite_bound(rhi) && rhi < -tol)) {
                    infeasible = true;
                    infeasible_reason = "row " + std::to_string(i) + " is empty but its bounds exclude 0";
                    break;
                }
                trail.push_back(RedundantRowMove{i});
                row_active[ri] = false;
                changed = true;
                continue;
            }

            const bool redundant =
                (!is_finite_bound(rlo) || s.min_activity >= rlo - tol) &&
                (!is_finite_bound(rhi) || s.max_activity <= rhi + tol);
            if (redundant) {
                trail.push_back(RedundantRowMove{i});
                row_active[ri] = false;
                changed = true;
                continue;
            }

            const bool proven_infeasible =
                (is_finite_bound(rlo) && s.max_activity < rlo - tol) ||
                (is_finite_bound(rhi) && s.min_activity > rhi + tol);
            if (proven_infeasible) {
                infeasible = true;
                infeasible_reason = "row " + std::to_string(i) + " cannot be satisfied by any "
                                    "point in the current column bounds";
                break;
            }

            if (s.active_count == 1) {
                const Idx j = s.only_col;
                const Real a = s.only_coeff;
                const auto jc = static_cast<std::size_t>(j);

                Real implied_lo = -kInfinity, implied_hi = kInfinity;
                if (a > 0.0) {
                    if (is_finite_bound(rlo)) implied_lo = rlo / a;
                    if (is_finite_bound(rhi)) implied_hi = rhi / a;
                } else {
                    if (is_finite_bound(rhi)) implied_lo = rhi / a;
                    if (is_finite_bound(rlo)) implied_hi = rlo / a;
                }

                const Real old_lower = col_lower[jc], old_upper = col_upper[jc];
                const Real new_lower = is_finite_bound(implied_lo) ? std::max(old_lower, implied_lo) : old_lower;
                const Real new_upper = is_finite_bound(implied_hi) ? std::min(old_upper, implied_hi) : old_upper;

                if (crossed(new_lower, new_upper)) {
                    infeasible = true;
                    infeasible_reason = "singleton row " + std::to_string(i)
                                        + " forces crossed bounds on column " + std::to_string(j);
                    break;
                }

                trail.push_back(SingletonRowMove{i, j, a, old_lower, old_upper, new_lower, new_upper});
                col_lower[jc] = new_lower;
                col_upper[jc] = new_upper;
                row_active[ri] = false;
                changed = true;
            }
        }

        if (!changed) break;
    }

    if (infeasible) {
        result.status = PresolveStatus::Infeasible;
        result.message = infeasible_reason;
        return result;
    }

    // -- build the reduced problem -------------------------------------------
    std::vector<Idx> row_map, col_map;
    std::vector<Idx> row_new_index(static_cast<std::size_t>(m), -1);
    std::vector<Idx> col_new_index(static_cast<std::size_t>(n), -1);
    for (Idx i = 0; i < m; ++i)
        if (row_active[static_cast<std::size_t>(i)]) {
            row_new_index[static_cast<std::size_t>(i)] = static_cast<Idx>(row_map.size());
            row_map.push_back(i);
        }
    for (Idx j = 0; j < n; ++j)
        if (col_active[static_cast<std::size_t>(j)]) {
            col_new_index[static_cast<std::size_t>(j)] = static_cast<Idx>(col_map.size());
            col_map.push_back(j);
        }

    Problem::Builder builder;
    builder.set_name(problem.name());
    builder.set_sense(problem.sense());
    builder.set_objective_name(problem.objective_name());
    builder.set_objective_constant(problem.objective_constant() + objective_constant_delta);

    for (Idx i : row_map)
        builder.add_row(problem.row_names()[static_cast<std::size_t>(i)],
                        row_lower[static_cast<std::size_t>(i)], row_upper[static_cast<std::size_t>(i)]);
    for (Idx j : col_map) {
        const Idx new_j = builder.add_column(problem.col_names()[static_cast<std::size_t>(j)]);
        builder.set_column_bounds(new_j, col_lower[static_cast<std::size_t>(j)],
                                  col_upper[static_cast<std::size_t>(j)]);
        builder.set_column_kind(new_j, problem.col_kind()[static_cast<std::size_t>(j)]);
        builder.set_objective_coefficient(new_j, obj[static_cast<std::size_t>(j)]);
    }
    for (Idx i : row_map) {
        const Idx new_i = row_new_index[static_cast<std::size_t>(i)];
        const auto begin = csr.row_ptr()[static_cast<std::size_t>(i)];
        const auto end = csr.row_ptr()[static_cast<std::size_t>(i) + 1];
        for (Idx k = begin; k < end; ++k) {
            const Idx j = csr.col_idx()[static_cast<std::size_t>(k)];
            if (!col_active[static_cast<std::size_t>(j)]) continue;
            builder.add_coefficient(new_i, col_new_index[static_cast<std::size_t>(j)],
                                    csr.values()[static_cast<std::size_t>(k)]);
        }
    }

    result.reduced = builder.finish();
    result.row_map = std::move(row_map);
    result.col_map = std::move(col_map);
    result.trail = std::move(trail);
    result.rows_removed = m - result.reduced.num_rows();
    result.cols_removed = n - result.reduced.num_cols();
    result.nonzeros_removed = problem.num_nonzeros() - result.reduced.num_nonzeros();
    result.status = PresolveStatus::Reduced;
    return result;
}

PostsolveResult postsolve(const PresolveResult& result, std::span<const Real> x_reduced,
                          std::span<const Real> y_reduced, std::span<const Real> rc_reduced) {
    PostsolveResult out;
    out.primal.assign(static_cast<std::size_t>(result.original_num_cols), 0.0);
    out.dual.assign(static_cast<std::size_t>(result.original_num_rows), 0.0);
    out.reduced_costs.assign(static_cast<std::size_t>(result.original_num_cols), 0.0);

    for (std::size_t i = 0; i < result.row_map.size(); ++i)
        out.dual[static_cast<std::size_t>(result.row_map[i])] = y_reduced[i];
    for (std::size_t j = 0; j < result.col_map.size(); ++j) {
        out.primal[static_cast<std::size_t>(result.col_map[j])] = x_reduced[j];
        out.reduced_costs[static_cast<std::size_t>(result.col_map[j])] = rc_reduced[j];
    }

    const Real tol = result.tolerance;
    for (auto it = result.trail.rbegin(); it != result.trail.rend(); ++it) {
        std::visit(
            [&](auto&& move) {
                using T = std::decay_t<decltype(move)>;
                if constexpr (std::is_same_v<T, FixedColumnMove>) {
                    out.primal[static_cast<std::size_t>(move.col)] = move.value;
                    Real rc = move.obj_coeff;
                    for (const auto& [row, coeff] : move.rows)
                        rc -= coeff * out.dual[static_cast<std::size_t>(row)];
                    out.reduced_costs[static_cast<std::size_t>(move.col)] = rc;
                } else if constexpr (std::is_same_v<T, EmptyColumnMove>) {
                    out.primal[static_cast<std::size_t>(move.col)] = move.value;
                    out.reduced_costs[static_cast<std::size_t>(move.col)] = move.obj_coeff;
                } else if constexpr (std::is_same_v<T, RedundantRowMove>) {
                    out.dual[static_cast<std::size_t>(move.row)] = 0.0;
                } else if constexpr (std::is_same_v<T, SingletonRowMove>) {
                    const auto jc = static_cast<std::size_t>(move.col);
                    const Real xj = out.primal[jc];
                    const bool at_new_lower =
                        is_finite_bound(move.new_lower) && std::abs(xj - move.new_lower) < tol;
                    const bool tightened_lower =
                        (is_finite_bound(move.new_lower) && !is_finite_bound(move.old_lower)) ||
                        (is_finite_bound(move.new_lower) && is_finite_bound(move.old_lower) &&
                         move.new_lower > move.old_lower + tol);
                    const bool at_new_upper =
                        is_finite_bound(move.new_upper) && std::abs(xj - move.new_upper) < tol;
                    const bool tightened_upper =
                        (is_finite_bound(move.new_upper) && !is_finite_bound(move.old_upper)) ||
                        (is_finite_bound(move.new_upper) && is_finite_bound(move.old_upper) &&
                         move.new_upper < move.old_upper - tol);

                    if (at_new_lower && tightened_lower) {
                        out.dual[static_cast<std::size_t>(move.row)] = out.reduced_costs[jc] / move.coeff;
                        out.reduced_costs[jc] = 0.0;
                    } else if (at_new_upper && tightened_upper) {
                        out.dual[static_cast<std::size_t>(move.row)] = out.reduced_costs[jc] / move.coeff;
                        out.reduced_costs[jc] = 0.0;
                    } else {
                        out.dual[static_cast<std::size_t>(move.row)] = 0.0;
                    }
                }
            },
            *it);
    }
    return out;
}

}  // namespace sov
