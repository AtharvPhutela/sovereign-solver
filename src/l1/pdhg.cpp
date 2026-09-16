// PDHG / PDLP engine -- Build Map ticket #8. Bible S4.2 Engine A.
// See pdhg.hpp for the full derivation of the iteration and the step-size
// choice. This file is the loop.
#include "sovereign/pdhg.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <span>
#include <utility>

#include "sovereign/scaling.hpp"

namespace sov {

const char* to_string(PdhgStatus s) noexcept {
    switch (s) {
        case PdhgStatus::Optimal:          return "optimal";
        case PdhgStatus::IterationLimit:   return "iteration_limit";
        case PdhgStatus::TimeLimit:        return "time_limit";
        case PdhgStatus::NumericalFailure: return "numerical_failure";
        case PdhgStatus::NotSolved:        return "not_solved";
        case PdhgStatus::Cancelled:        return "cancelled";
    }
    return "unknown";
}

namespace {

/// The infimum of `coeff * v` for v ranging over [lo, hi]. Matches
/// FarkasCertificate's term_range in duals.cpp (same underlying idea: a
/// linear functional's extreme value over a box) but kept local and returning
/// only the one-sided value PDHG's dual objective needs, so this file does
/// not reach into another header's private helpers for five lines of math.
Real inf_over_box(Real coeff, Real lo, Real hi) {
    if (coeff == 0.0) return 0.0;
    if (coeff > 0.0) return is_finite_bound(lo) ? coeff * lo : -kInfinity;
    return is_finite_bound(hi) ? coeff * hi : -kInfinity;
}

/// Power iteration for the largest singular value of the saddle-point
/// operator K(u, v) = v - A u  (u in R^n, v in R^m), K^T(w) = (-A^T w, w).
/// This IS the Chambolle-Pock operator this engine's iteration uses (see
/// pdhg.hpp) -- measuring it directly, rather than assuming a bound from the
/// Pock-Chambolle diagonal formula, is what makes the step size an honest
/// safety margin instead of a hopeful one.
Real estimate_operator_norm(Backend& backend, const DeviceCsr& A, Idx n, Idx m,
                            int iterations) {
    const auto nu = static_cast<std::size_t>(n);
    const auto mu = static_cast<std::size_t>(m);
    DeviceBuffer u = DeviceBuffer::upload(backend, std::vector<Real>(nu, 1.0));
    DeviceBuffer v = DeviceBuffer::upload(backend, std::vector<Real>(mu, 1.0));
    DeviceBuffer w(backend, mu);

    Real estimate = 0.0;
    for (int it = 0; it < iterations; ++it) {
        // w = K(u,v) = v - A u, then normalize.
        backend.spmv(A.view(), -1.0, u.data(), 0.0, w.data());
        backend.axpy(m, 1.0, v.data(), w.data());
        Real wn = 0.0;
        backend.norm2(m, w.data(), &wn);
        if (wn < 1e-300) break;
        backend.scale(m, 1.0 / wn, w.data());

        // (u,v) = K^T(w) = (-A^T w, w), then normalize jointly.
        backend.spmv_transpose(A.view(), -1.0, w.data(), 0.0, u.data());
        backend.copy_device_to_device(w.data(), v.data(), mu * sizeof(Real));

        Real un = 0.0, vn = 0.0;
        backend.norm2(n, u.data(), &un);
        backend.norm2(m, v.data(), &vn);
        estimate = std::sqrt(un * un + vn * vn);
        if (estimate < 1e-300) break;
        backend.scale(n, 1.0 / estimate, u.data());
        backend.scale(m, 1.0 / estimate, v.data());
    }
    return estimate;
}

/// KKT-style residual of a point (x, y) against the SCALED problem: the
/// larger of the worst row-bound violation of A x and the worst
/// complementary-slackness-style violation of the reduced cost c - A^T y.
/// Deliberately a small host-side computation -- this only runs once every
/// `restart_check_period` iterations, the "CPU decides" checkpoint against a
/// GPU-produced candidate, not part of the hot loop.
Real kkt_residual(const Problem& scaled, std::span<const Real> c,
                  std::span<const Real> x, std::span<const Real> y, Real bound_tol) {
    const Idx n = scaled.num_cols();
    const Idx m = scaled.num_rows();

    std::vector<Real> activity(static_cast<std::size_t>(m), 0.0);
    scaled.matrix().multiply(x, activity);
    Real primal_res = 0.0;
    Real dual_res = 0.0;

    // Row-dual complementary slackness: y_i must be ~0 unless row i sits at a
    // bound (>= 0 at the lower bound, <= 0 at the upper -- the same fixed-
    // point argument as the column check below, applied to s = Proj_S(s -
    // tau*y): an interior s forces tau*y = 0 at any true fixed point, so a
    // nonzero y on a slack row is only ever legitimate right at that row's
    // bound). Skipping this side of the check is what let early PDHG
    // convergence claims through on `blend`/`adlittle` while y was still
    // very wrong on non-binding rows -- the primal point looked converged
    // and the column reduced costs did too, and nothing was checking y itself.
    for (Idx i = 0; i < m; ++i) {
        const auto k = static_cast<std::size_t>(i);
        if (is_finite_bound(scaled.row_lower()[k]))
            primal_res = std::max(primal_res, scaled.row_lower()[k] - activity[k]);
        if (is_finite_bound(scaled.row_upper()[k]))
            primal_res = std::max(primal_res, activity[k] - scaled.row_upper()[k]);

        const bool at_lo = is_finite_bound(scaled.row_lower()[k])
                          && std::abs(activity[k] - scaled.row_lower()[k]) < bound_tol;
        const bool at_hi = is_finite_bound(scaled.row_upper()[k])
                          && std::abs(activity[k] - scaled.row_upper()[k]) < bound_tol;
        if (!at_lo && !at_hi) dual_res = std::max(dual_res, std::abs(y[k]));
        else if (at_lo && !at_hi) dual_res = std::max(dual_res, std::max(Real{0}, -y[k]));
        else if (at_hi && !at_lo) dual_res = std::max(dual_res, std::max(Real{0}, y[k]));
    }

    std::vector<Real> aty(static_cast<std::size_t>(n), 0.0);
    scaled.matrix().multiply_transpose(y, aty);
    for (Idx j = 0; j < n; ++j) {
        const auto k = static_cast<std::size_t>(j);
        const Real r = c[k] - aty[k];
        const Real lo = scaled.col_lower()[k], hi = scaled.col_upper()[k];
        const bool at_lo = is_finite_bound(lo) && std::abs(x[k] - lo) < bound_tol;
        const bool at_hi = is_finite_bound(hi) && std::abs(x[k] - hi) < bound_tol;
        if (!at_lo && !at_hi) dual_res = std::max(dual_res, std::abs(r));
        else if (at_lo && !at_hi) dual_res = std::max(dual_res, std::max(Real{0}, -r));
        else if (at_hi && !at_lo) dual_res = std::max(dual_res, std::max(Real{0}, r));
        // at both (a fixed column): no constraint on r.
    }
    return std::max(primal_res, dual_res);
}

/// The same residual, but against the ORIGINAL problem in its own units --
/// used only to decide convergence, not for the restart heuristic. Checking
/// in scaled space alone is not enough: unscaling divides column j's reduced
/// cost by col_scale_j, so a residual safely under tolerance in scaled space
/// can land anywhere after unscaling depending on how aggressive that
/// column's scale factor was. The tolerance the caller set is a promise about
/// real units, so convergence has to be verified there.
Real original_space_residual(const Problem& problem, Real obj_sign, const Scaling& scaling,
                             std::span<const Real> x_scaled, std::span<const Real> y_scaled,
                             Real bound_tol) {
    std::vector<Real> x(x_scaled.begin(), x_scaled.end());
    scaling.unscale_primal(x);
    std::vector<Real> y(y_scaled.begin(), y_scaled.end());
    for (Real& v : y) v *= obj_sign;
    scaling.unscale_dual(y);

    const Idx n = problem.num_cols();
    const Idx m = problem.num_rows();

    std::vector<Real> activity(static_cast<std::size_t>(m), 0.0);
    problem.matrix().multiply(x, activity);
    Real primal_res = 0.0;
    Real dual_res = 0.0;

    // Row-dual complementary slackness -- see kkt_residual's comment for why
    // this side of the check is not optional. `y` here is already in
    // original sense (obj_sign applied above), so fold it back to the
    // internal sign before comparing, same trick as the column loop below.
    for (Idx i = 0; i < m; ++i) {
        const auto k = static_cast<std::size_t>(i);
        if (is_finite_bound(problem.row_lower()[k]))
            primal_res = std::max(primal_res, problem.row_lower()[k] - activity[k]);
        if (is_finite_bound(problem.row_upper()[k]))
            primal_res = std::max(primal_res, activity[k] - problem.row_upper()[k]);

        const Real y_int = obj_sign * y[k];
        const bool at_lo = is_finite_bound(problem.row_lower()[k])
                          && std::abs(activity[k] - problem.row_lower()[k]) < bound_tol;
        const bool at_hi = is_finite_bound(problem.row_upper()[k])
                          && std::abs(activity[k] - problem.row_upper()[k]) < bound_tol;
        if (!at_lo && !at_hi) dual_res = std::max(dual_res, std::abs(y_int));
        else if (at_lo && !at_hi) dual_res = std::max(dual_res, std::max(Real{0}, -y_int));
        else if (at_hi && !at_lo) dual_res = std::max(dual_res, std::max(Real{0}, y_int));
    }

    std::vector<Real> aty(static_cast<std::size_t>(n), 0.0);
    problem.matrix().multiply_transpose(y, aty);
    for (Idx j = 0; j < n; ++j) {
        const auto k = static_cast<std::size_t>(j);
        // Original-sense reduced cost (ticket #7 convention); folded back to
        // the internal always-minimize sign so one comparison covers both
        // Minimize and Maximize -- reduced_cost_original = obj_sign * r_int.
        const Real r_int = obj_sign * (problem.objective()[k] - aty[k]);
        const Real lo = problem.col_lower()[k], hi = problem.col_upper()[k];
        const bool at_lo = is_finite_bound(lo) && std::abs(x[k] - lo) < bound_tol;
        const bool at_hi = is_finite_bound(hi) && std::abs(x[k] - hi) < bound_tol;
        if (!at_lo && !at_hi) dual_res = std::max(dual_res, std::abs(r_int));
        else if (at_lo && !at_hi) dual_res = std::max(dual_res, std::max(Real{0}, -r_int));
        else if (at_hi && !at_lo) dual_res = std::max(dual_res, std::max(Real{0}, r_int));
    }
    return std::max(primal_res, dual_res);
}

/// Unscales a (x, y) pair from PDHG's internal always-minimize, Ruiz/Pock-
/// Chambolle-scaled space back to the problem's own units and sense. Shared
/// by the ticket #11 checkpoint callback and the final result population so
/// both report the same point the same way.
std::pair<std::vector<Real>, std::vector<Real>> unscale_to_original(
        Real obj_sign, const Scaling& scaling,
        std::span<const Real> x_scaled, std::span<const Real> y_scaled) {
    std::vector<Real> x(x_scaled.begin(), x_scaled.end());
    scaling.unscale_primal(x);
    std::vector<Real> y(y_scaled.begin(), y_scaled.end());
    for (Real& v : y) v *= obj_sign;
    scaling.unscale_dual(y);
    return {std::move(x), std::move(y)};
}

}  // namespace

PdhgResult Pdhg::solve(const Problem& problem, Backend& backend,
                       const CancellationToken* cancel) const {
    PdhgResult result;
    const auto t0 = std::chrono::steady_clock::now();

    std::string why;
    if (problem.validate(&why) != Status::Ok) {
        result.status = PdhgStatus::NumericalFailure;
        result.message = "invalid problem: " + why;
        return result;
    }
    if (problem.has_crossed_bounds()) {
        // PDHG has no analogue of the simplex's Phase I here -- it would
        // iterate forever trying to satisfy a bound pair that cannot be
        // satisfied. Same rule as Simplex::run(): caught before any solve.
        result.status = PdhgStatus::NumericalFailure;
        result.message = "a bound pair is crossed (lower > upper); PDHG has no "
                         "infeasibility detection yet (tracked, not silently ignored)";
        return result;
    }

    // -- Ruiz + Pock-Chambolle front-end (ticket #6), mandatory (Bible S6.2) -
    auto [scaled, scaling] = Scaling::equilibrate(problem);
    const Idx n = scaled.num_cols();
    const Idx m = scaled.num_rows();
    const auto nu = static_cast<std::size_t>(n);
    const auto mu = static_cast<std::size_t>(m);

    DeviceCsr A = DeviceCsr::upload(backend, scaled.matrix());

    // Internal minimize convention, matching Simplex exactly so the two
    // engines' duals/reduced costs need no per-engine translation (ticket #7).
    const Real obj_sign = (scaled.sense() == ObjSense::Maximize) ? -1.0 : 1.0;
    std::vector<Real> c_host(scaled.objective().begin(), scaled.objective().end());
    for (Real& v : c_host) v *= obj_sign;

    DeviceBuffer c      = DeviceBuffer::upload(backend, c_host);
    DeviceBuffer col_lo = DeviceBuffer::upload(backend, scaled.col_lower());
    DeviceBuffer col_hi = DeviceBuffer::upload(backend, scaled.col_upper());
    DeviceBuffer row_lo = DeviceBuffer::upload(backend, scaled.row_lower());
    DeviceBuffer row_hi = DeviceBuffer::upload(backend, scaled.row_upper());

    // -- step size: measured, not assumed (see estimate_operator_norm) ------
    const Real op_norm = estimate_operator_norm(backend, A, n, m, options_.power_iterations);
    result.operator_norm_estimate = op_norm;
    const Real step = (op_norm > 1e-9) ? options_.step_size_safety / op_norm
                                       : options_.step_size_safety;
    const Real tau = step, sigma = step;

    // -- state ----------------------------------------------------------
    DeviceBuffer x(backend, nu), s(backend, mu), y(backend, mu);
    backend.project_box(n, col_lo.data(), col_hi.data(), x.data());
    backend.project_box(m, row_lo.data(), row_hi.data(), s.data());

    DeviceBuffer x_new(backend, nu), s_new(backend, mu);
    DeviceBuffer x_bar(backend, nu), s_bar(backend, mu);
    DeviceBuffer grad(backend, nu);      // scratch, n: c - A^T y
    DeviceBuffer k_of_bar(backend, mu);  // scratch, m: s_bar - A*x_bar

    DeviceBuffer x_sum(backend, nu), y_sum(backend, mu);   // Cesaro averages
    long long since_restart = 0;
    Real residual_at_last_restart = kInfinity;

    const std::size_t n_bytes = nu * sizeof(Real);
    const std::size_t m_bytes = mu * sizeof(Real);

    std::vector<Real> best_x_scaled, best_y_scaled;
    bool converged = false;

    for (result.iterations = 0; result.iterations < options_.max_iterations; ++result.iterations) {
        if (cancel != nullptr && cancel->is_cancelled()) { result.status = PdhgStatus::Cancelled; break; }
        if (options_.time_limit_seconds > 0.0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (elapsed > options_.time_limit_seconds) { result.status = PdhgStatus::TimeLimit; break; }
        }

        // x_new = Proj_X( x - tau*(c - A^T y) )
        backend.copy_device_to_device(c.data(), grad.data(), n_bytes);
        backend.spmv_transpose(A.view(), -1.0, y.data(), 1.0, grad.data());  // grad = c - A^T y
        backend.copy_device_to_device(x.data(), x_new.data(), n_bytes);
        backend.axpy(n, -tau, grad.data(), x_new.data());
        backend.project_box(n, col_lo.data(), col_hi.data(), x_new.data());

        // s_new = Proj_S( s - tau*y )
        backend.copy_device_to_device(s.data(), s_new.data(), m_bytes);
        backend.axpy(m, -tau, y.data(), s_new.data());
        backend.project_box(m, row_lo.data(), row_hi.data(), s_new.data());

        // extrapolation: x_bar = 2 x_new - x, s_bar = 2 s_new - s
        backend.copy_device_to_device(x_new.data(), x_bar.data(), n_bytes);
        backend.scale(n, 2.0, x_bar.data());
        backend.axpy(n, -1.0, x.data(), x_bar.data());
        backend.copy_device_to_device(s_new.data(), s_bar.data(), m_bytes);
        backend.scale(m, 2.0, s_bar.data());
        backend.axpy(m, -1.0, s.data(), s_bar.data());

        // y += sigma * ( s_bar - A * x_bar )
        backend.spmv(A.view(), -1.0, x_bar.data(), 0.0, k_of_bar.data());
        backend.axpy(m, 1.0, s_bar.data(), k_of_bar.data());
        backend.axpy(m, sigma, k_of_bar.data(), y.data());

        backend.copy_device_to_device(x_new.data(), x.data(), n_bytes);
        backend.copy_device_to_device(s_new.data(), s.data(), m_bytes);

        // Cesaro accumulation since the last restart.
        if (since_restart == 0) {
            backend.copy_device_to_device(x.data(), x_sum.data(), n_bytes);
            backend.copy_device_to_device(y.data(), y_sum.data(), m_bytes);
        } else {
            backend.axpy(n, 1.0, x.data(), x_sum.data());
            backend.axpy(m, 1.0, y.data(), y_sum.data());
        }
        ++since_restart;

        const bool check_now = (result.iterations + 1) % options_.restart_check_period == 0;
        if (!check_now) continue;

        const std::vector<Real> x_host = x.to_host();
        const std::vector<Real> y_host = y.to_host();
        const Real current_residual = kkt_residual(scaled, c_host, x_host, y_host, options_.tolerance);

        std::vector<Real> x_avg = x_sum.to_host();
        std::vector<Real> y_avg = y_sum.to_host();
        const Real inv = 1.0 / static_cast<Real>(since_restart);
        for (Real& v : x_avg) v *= inv;
        for (Real& v : y_avg) v *= inv;
        const Real avg_residual = kkt_residual(scaled, c_host, x_avg, y_avg, options_.tolerance);

        if (!std::isfinite(current_residual) || !std::isfinite(avg_residual)) {
            result.status = PdhgStatus::NumericalFailure;
            result.message = "non-finite KKT residual at iteration "
                            + std::to_string(result.iterations);
            return result;
        }

        const bool avg_is_better = avg_residual <= current_residual;

        // The scaled-space residual decides whether to restart (a relative,
        // same-space comparison -- unaffected by the unscaling concern
        // above); actual convergence is only declared once the ORIGINAL-space
        // residual of that same candidate is also within tolerance.
        const std::vector<Real>& candidate_x = avg_is_better ? x_avg : x_host;
        const std::vector<Real>& candidate_y = avg_is_better ? y_avg : y_host;
        const Real original_residual =
            original_space_residual(problem, obj_sign, scaling, candidate_x, candidate_y,
                                    std::max(options_.tolerance, Real{1e-6}));

        if (!std::isfinite(original_residual)) {
            result.status = PdhgStatus::NumericalFailure;
            result.message = "non-finite KKT residual (original units) at iteration "
                            + std::to_string(result.iterations);
            return result;
        }

        // Ticket #11: hand this checkpoint's candidate to whoever wants to try
        // crossing it over to an exact vertex, whether or not PDHG itself
        // considers it converged. `unscale_to_original` is exactly the
        // unscaling this function already does at the very end for the final
        // result -- the checkpoint costs nothing beyond that one extra call.
        if (options_.checkpoint) {
            auto [cx, cy] = unscale_to_original(obj_sign, scaling, candidate_x, candidate_y);
            options_.checkpoint(result.iterations, cx, cy, original_residual);
        }

        if (original_residual < options_.tolerance) {
            best_x_scaled = candidate_x;
            best_y_scaled = candidate_y;
            converged = true;
            ++result.iterations;
            break;
        }

        if (!std::isfinite(residual_at_last_restart)) residual_at_last_restart = current_residual;

        if (avg_residual <= options_.restart_sufficient_decrease * residual_at_last_restart) {
            x = DeviceBuffer::upload(backend, x_avg);
            y = DeviceBuffer::upload(backend, y_avg);
            backend.project_box(n, col_lo.data(), col_hi.data(), x.data());
            std::vector<Real> activity(mu, 0.0);
            scaled.matrix().multiply(x_avg, activity);
            s = DeviceBuffer::upload(backend, activity);
            backend.project_box(m, row_lo.data(), row_hi.data(), s.data());

            residual_at_last_restart = avg_residual;
            since_restart = 0;
            ++result.restarts;
            if (options_.verbose)
                std::fprintf(stderr, "  [pdhg] restart at iter %lld, residual %.3g\n",
                            result.iterations + 1, static_cast<double>(avg_residual));
        }
    }

    // Populating the reported point is NOT conditional on which status we
    // ended up with -- it must happen whenever the convergence check never
    // fired, whatever stopped the loop. An earlier version only did this
    // inside the "ran out of iterations" branch, so TimeLimit and Cancelled
    // (both of which can fire before iteration 0's own restart-check period
    // elapses -- e.g. a time budget consumed by setup on a large instance,
    // or an immediate race cancellation) left best_x_scaled empty, and the
    // unscale call below threw a dimension mismatch instead of reporting a
    // (loss-worthy, but not crash-worthy) partial result.
    if (best_x_scaled.empty()) {
        best_x_scaled = x.to_host();
        best_y_scaled = y.to_host();
    }
    if (converged) result.status = PdhgStatus::Optimal;
    else if (result.status == PdhgStatus::NotSolved) result.status = PdhgStatus::IterationLimit;

    // -- unscale back to the ORIGINAL problem's units and sense -------------
    auto [x_original, y_original] = unscale_to_original(obj_sign, scaling, best_x_scaled, best_y_scaled);

    result.primal = x_original;
    result.dual = y_original;
    result.objective = problem.evaluate_objective(x_original);

    result.reduced_costs.assign(static_cast<std::size_t>(problem.num_cols()), 0.0);
    {
        std::vector<Real> aty(static_cast<std::size_t>(problem.num_cols()), 0.0);
        problem.matrix().multiply_transpose(y_original, aty);
        for (Idx j = 0; j < problem.num_cols(); ++j) {
            const auto k = static_cast<std::size_t>(j);
            result.reduced_costs[k] = problem.objective()[k] - aty[k];
        }
    }

    // primal infeasibility on the ORIGINAL problem
    {
        std::vector<Real> activity(static_cast<std::size_t>(problem.num_rows()), 0.0);
        problem.matrix().multiply(x_original, activity);
        for (Idx i = 0; i < problem.num_rows(); ++i) {
            const auto k = static_cast<std::size_t>(i);
            if (is_finite_bound(problem.row_lower()[k]))
                result.primal_infeasibility = std::max(result.primal_infeasibility,
                                                       problem.row_lower()[k] - activity[k]);
            if (is_finite_bound(problem.row_upper()[k]))
                result.primal_infeasibility = std::max(result.primal_infeasibility,
                                                       activity[k] - problem.row_upper()[k]);
        }
    }

    // dual infeasibility (complementary slackness), reusing ticket #7 directly
    {
        DualSolution d;
        d.row_duals = result.dual;
        d.reduced_costs = result.reduced_costs;
        // The checker's own default bound_tol (1e-7) is calibrated for the
        // simplex's exact arithmetic. PDHG's row activity only lands within
        // its own first-order tolerance of a bound, not machine precision --
        // using the tight default here would misclassify a genuinely-binding
        // row as "not at a bound" and flag its nonzero dual as a violation
        // that was never real.
        result.dual_infeasibility = d.complementary_slackness_violation(
            problem, result.primal, std::max(options_.tolerance, Real{1e-6}));
    }

    // duality gap against the honest weak-dual bound g(y)
    {
        Real dual_obj = 0.0;
        bool finite = true;
        for (Idx j = 0; j < problem.num_cols() && finite; ++j) {
            const auto k = static_cast<std::size_t>(j);
            const Real term = inf_over_box(result.reduced_costs[k],
                                           problem.col_lower()[k], problem.col_upper()[k]);
            if (!std::isfinite(term)) { finite = false; break; }
            dual_obj += term;
        }
        for (Idx i = 0; i < problem.num_rows() && finite; ++i) {
            const auto k = static_cast<std::size_t>(i);
            const Real term = inf_over_box(result.dual[k],
                                           problem.row_lower()[k], problem.row_upper()[k]);
            if (!std::isfinite(term)) { finite = false; break; }
            dual_obj += term;
        }
        if (finite) {
            const Real denom = 1.0 + std::abs(result.objective) + std::abs(dual_obj);
            result.duality_gap = std::abs(result.objective - dual_obj) / denom;
        } else {
            result.duality_gap = kInfinity;
        }
    }

    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

}  // namespace sov
