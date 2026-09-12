// Regularized pivoting-free IPM -- Build Map ticket #9. Bible S4.2 Engine B.
// See ipm.hpp for the full derivation of the SQD system this factorizes.
#include "sovereign/ipm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#include "sovereign/scaling.hpp"

namespace sov {

const char* to_string(IpmStatus s) noexcept {
    switch (s) {
        case IpmStatus::Optimal:          return "optimal";
        case IpmStatus::IterationLimit:   return "iteration_limit";
        case IpmStatus::TimeLimit:        return "time_limit";
        case IpmStatus::NumericalFailure: return "numerical_failure";
        case IpmStatus::NotSolved:        return "not_solved";
        case IpmStatus::Cancelled:        return "cancelled";
    }
    return "unknown";
}

namespace {

constexpr Idx kMaxDenseSize = 3000;   // on the AUGMENTED (n+2m) system -- see
                                      // ipm.hpp's scope note, same discipline
                                      // as simplex.cpp's kMaxDenseRows.

/// Dense symmetric LDL^T, no pivoting, natural diagonal order. Sound only for
/// a matrix an SQD regularization has already made strictly quasi-definite
/// (Vanderbei 1995) -- this class does not check that itself, it trusts the
/// caller, exactly as the ticket's "no pivoting on the critical path" says
/// nothing pivots here by design, not by luck.
class DenseSymmetricLDL {
public:
    explicit DenseSymmetricLDL(Idx n) : n_(n), a_(static_cast<std::size_t>(n) * n),
                                        d_(static_cast<std::size_t>(n)) {}

    Real& at(Idx i, Idx j) { return a_[static_cast<std::size_t>(i) * n_ + j]; }
    Real at(Idx i, Idx j) const { return a_[static_cast<std::size_t>(i) * n_ + j]; }

    /// Factorize the LOWER triangle (including the diagonal) of whatever is
    /// currently stored -- the caller fills a symmetric matrix and only the
    /// lower half is ever read. In place: `at()` below the diagonal becomes
    /// L's strictly-lower entries after this call, `d_` holds D.
    Status factorize() {
        for (Idx k = 0; k < n_; ++k) {
            const Real dk = at(k, k);
            if (std::abs(dk) < 1e-300) return Status::NumericalFailure;
            d_[static_cast<std::size_t>(k)] = dk;
            for (Idx i = k + 1; i < n_; ++i) at(i, k) /= dk;
            for (Idx j = k + 1; j < n_; ++j) {
                const Real ljk = at(j, k);
                if (ljk == 0.0) continue;
                for (Idx i = j; i < n_; ++i)
                    at(i, j) -= at(i, k) * dk * ljk;
            }
        }
        return Status::Ok;
    }

    /// Solve K x = b in place, using the L, D stored by factorize().
    void solve(std::vector<Real>& b) const {
        // Forward: L y = b (L unit lower triangular, strictly-lower entries
        // stored in at(i,k) for i>k).
        for (Idx k = 0; k < n_; ++k) {
            const Real yk = b[static_cast<std::size_t>(k)];
            if (yk == 0.0) continue;
            for (Idx i = k + 1; i < n_; ++i)
                b[static_cast<std::size_t>(i)] -= at(i, k) * yk;
        }
        // Diagonal: D z = y.
        for (Idx k = 0; k < n_; ++k)
            b[static_cast<std::size_t>(k)] /= d_[static_cast<std::size_t>(k)];
        // Backward: L^T x = z.
        for (Idx k = n_ - 1; k >= 0; --k) {
            Real acc = b[static_cast<std::size_t>(k)];
            for (Idx i = k + 1; i < n_; ++i)
                acc -= at(i, k) * b[static_cast<std::size_t>(i)];
            b[static_cast<std::size_t>(k)] = acc;
        }
    }

private:
    Idx n_;
    std::vector<Real> a_;
    std::vector<Real> d_;
};

}  // namespace

IpmResult Ipm::solve(const Problem& problem, const CancellationToken* cancel) const {
    IpmResult result;
    const auto t0 = std::chrono::steady_clock::now();

    std::string why;
    if (problem.validate(&why) != Status::Ok) {
        result.status = IpmStatus::NumericalFailure;
        result.message = "invalid problem: " + why;
        return result;
    }
    if (problem.has_crossed_bounds()) {
        result.status = IpmStatus::NumericalFailure;
        result.message = "a bound pair is crossed (lower > upper); this engine "
                         "has no infeasibility detection, same tracked gap as PDHG";
        return result;
    }

    // Ruiz + Pock-Chambolle front-end (ticket #6), mandatory for every L1
    // engine (Bible S6.2), not just PDHG -- an interior-point method is, if
    // anything, MORE sensitive to poor scaling than a first-order method,
    // since Theta^{-1}'s entries span whatever range the raw coefficients do.
    // Skipping this here (an earlier version did) is exactly what made
    // `adlittle` -- a known badly-scaled Netlib instance -- blow mu up past
    // 1e9 instead of converging.
    auto [scaled, scaling] = Scaling::equilibrate(problem);

    const Idx n = scaled.num_cols();
    const Idx m = scaled.num_rows();
    const Idx N = n + m;          // size of z = (x, s)
    const Idx Ntot = N + m;       // augmented system size: (dz, dy)

    if (Ntot > kMaxDenseSize) {
        result.status = IpmStatus::NumericalFailure;
        result.message = "augmented system has size " + std::to_string(Ntot)
                        + "; this engine uses dense LDL^T and is limited to "
                        + std::to_string(kMaxDenseSize) + ". Sparse LDL^T "
                          "(cuDSS on GPU) is follow-up work.";
        return result;
    }

    // Internal minimize convention, matching Simplex and PDHG.
    const Real obj_sign = (scaled.sense() == ObjSense::Maximize) ? -1.0 : 1.0;

    std::vector<Real> lo(static_cast<std::size_t>(N)), hi(static_cast<std::size_t>(N));
    std::vector<Real> cz(static_cast<std::size_t>(N), 0.0);
    for (Idx j = 0; j < n; ++j) {
        const auto k = static_cast<std::size_t>(j);
        lo[k] = scaled.col_lower()[k];
        hi[k] = scaled.col_upper()[k];
        cz[k] = obj_sign * scaled.objective()[k];
    }
    for (Idx i = 0; i < m; ++i) {
        const auto k = static_cast<std::size_t>(n + i);
        lo[k] = scaled.row_lower()[static_cast<std::size_t>(i)];
        hi[k] = scaled.row_upper()[static_cast<std::size_t>(i)];
    }
    std::vector<char> has_lo(static_cast<std::size_t>(N)), has_hi(static_cast<std::size_t>(N));
    for (Idx j = 0; j < N; ++j) {
        const auto k = static_cast<std::size_t>(j);
        has_lo[k] = is_finite_bound(lo[k]) ? 1 : 0;
        has_hi[k] = is_finite_bound(hi[k]) ? 1 : 0;
    }

    // -- initial point: simple, standard, not the Mehrotra starting heuristic
    // (a documented scope choice, not an oversight -- see HANDOFF) ----------
    //
    // The both-bounds-finite case is the one that has to be built carefully:
    // g_j and h_j are NOT independent (dg_j = dz_j = -dh_j always, so their
    // SUM is fixed at hi_j - lo_j for the rest of the solve), and an earlier
    // version of this flooured g_j and h_j to at least 1.0 independently
    // without checking that sum -- which silently pushed z_j outside its own
    // [lo_j, hi_j] for any gap narrower than 2, equality rows (gap = 0)
    // included. The barrier method's own convergence check only watches
    // M z = 0 and complementarity, never "is z itself still in its box", so
    // that corruption was never caught -- it converged to a self-consistent
    // but wrong point on afiro, blend, sc50a/b, scsd1, recipe, share2b (all
    // real-corpus instances with equality rows), while the small hand-built
    // LP tests above happened not to trip it. Splitting the ACTUAL gap in
    // half, floored only when the gap itself is (near) zero, keeps g_j+h_j
    // exactly consistent by construction.
    std::vector<Real> z(static_cast<std::size_t>(N)), y(static_cast<std::size_t>(m), 0.0);
    std::vector<Real> g(static_cast<std::size_t>(N), 0.0), h(static_cast<std::size_t>(N), 0.0);
    std::vector<Real> lambda(static_cast<std::size_t>(N), 0.0), zeta(static_cast<std::size_t>(N), 0.0);
    for (Idx j = 0; j < N; ++j) {
        const auto k = static_cast<std::size_t>(j);
        if (has_lo[k] && has_hi[k]) {
            // A fixed variable or an equality row's slack (gap == 0) has no
            // true interior for a log-barrier method; presolve (#13) is what
            // actually eliminates these. Until then, a tiny artificial
            // interior is a documented approximation, not a bug -- it shrinks
            // toward the true point as mu shrinks, and is orders of magnitude
            // below this engine's own tolerance.
            const Real half_gap = std::max((hi[k] - lo[k]) / 2.0, Real{1e-8});
            g[k] = half_gap;
            h[k] = half_gap;
            z[k] = lo[k] + g[k];
        } else if (has_lo[k]) {
            g[k] = std::max(std::abs(lo[k]) * 0.1, Real{1.0});
            z[k] = lo[k] + g[k];
        } else if (has_hi[k]) {
            h[k] = std::max(std::abs(hi[k]) * 0.1, Real{1.0});
            z[k] = hi[k] - h[k];
        } else {
            z[k] = 0.0;
        }
        if (has_lo[k]) lambda[k] = 1.0;
        if (has_hi[k]) zeta[k] = 1.0;
    }

    Idx n_terms = 0;
    for (Idx j = 0; j < N; ++j) {
        const auto k = static_cast<std::size_t>(j);
        n_terms += has_lo[k] + has_hi[k];
    }
    if (n_terms == 0) {
        // Every variable free on both sides: the LP is either unbounded or
        // trivially zero-cost. Not this engine's job (no unboundedness
        // detection either, same tracked gap as crossed bounds above).
        result.status = IpmStatus::NumericalFailure;
        result.message = "every variable is free on both sides; nothing for "
                         "a barrier method to center against";
        return result;
    }

    DenseSymmetricLDL K(Ntot);
    std::vector<Real> rp(static_cast<std::size_t>(m)), rd(static_cast<std::size_t>(N));
    std::vector<Real> dz_aff(static_cast<std::size_t>(N)), dy_aff(static_cast<std::size_t>(m));
    std::vector<Real> dlambda_aff(static_cast<std::size_t>(N)), dzeta_aff(static_cast<std::size_t>(N));

    auto compute_residuals = [&]() {
        // r_p = M z = A x - s
        std::vector<Real> activity(static_cast<std::size_t>(m), 0.0);
        scaled.matrix().multiply(std::span<const Real>(z.data(), static_cast<std::size_t>(n)), activity);
        for (Idx i = 0; i < m; ++i)
            rp[static_cast<std::size_t>(i)] = activity[static_cast<std::size_t>(i)]
                                             - z[static_cast<std::size_t>(n + i)];
        // r_d = c_z - M^T y - lambda + zeta.  (M^T y)_x = A^T y, (M^T y)_s = -y.
        std::vector<Real> aty(static_cast<std::size_t>(n), 0.0);
        scaled.matrix().multiply_transpose(y, aty);
        for (Idx j = 0; j < n; ++j) {
            const auto k = static_cast<std::size_t>(j);
            rd[k] = cz[k] - aty[k] - lambda[k] + zeta[k];
        }
        for (Idx i = 0; i < m; ++i) {
            const auto k = static_cast<std::size_t>(n + i);
            rd[k] = cz[k] - (-y[static_cast<std::size_t>(i)]) - lambda[k] + zeta[k];
        }
    };

    // Best-iterate tracking. Once mu has overshot far below the target
    // tolerance (seen empirically on `adlittle`: mu reached ~1e-32 -- deep
    // into floating-point noise relative to the O(1) quantities Theta^{-1}
    // is built from) the next Newton step can be numerically unreliable and
    // actively move away from the optimum, even though the SQD system it
    // came from is theoretically sound. Reporting whichever iterate had the
    // best combined residual, rather than always the last one, is the
    // standard, robust answer to exactly this late-stage fragility -- it
    // costs one comparison and a snapshot copy per iteration.
    Real best_merit = kInfinity;
    std::vector<Real> best_z, best_y;

    for (result.iterations = 0; result.iterations < options_.max_iterations; ++result.iterations) {
        if (cancel != nullptr && cancel->is_cancelled()) { result.status = IpmStatus::Cancelled; break; }
        if (options_.time_limit_seconds > 0.0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (elapsed > options_.time_limit_seconds) { result.status = IpmStatus::TimeLimit; break; }
        }

        compute_residuals();

        Real mu = 0.0;
        for (Idx j = 0; j < N; ++j) {
            const auto k = static_cast<std::size_t>(j);
            if (has_lo[k]) mu += lambda[k] * g[k];
            if (has_hi[k]) mu += zeta[k] * h[k];
        }
        mu /= static_cast<Real>(n_terms);

        const Real primal_res = *std::max_element(rp.begin(), rp.end(),
            [](Real a, Real b) { return std::abs(a) < std::abs(b); });
        const Real dual_res = *std::max_element(rd.begin(), rd.end(),
            [](Real a, Real b) { return std::abs(a) < std::abs(b); });
        result.primal_infeasibility = std::abs(primal_res);
        result.dual_infeasibility = std::abs(dual_res);
        result.complementarity_gap = mu;

        if (options_.verbose)
            std::fprintf(stderr, "  [ipm] iter %d: primal_res=%.3g dual_res=%.3g mu=%.3g\n",
                        result.iterations, static_cast<double>(result.primal_infeasibility),
                        static_cast<double>(result.dual_infeasibility), static_cast<double>(mu));

        if (result.primal_infeasibility < options_.tolerance
            && result.dual_infeasibility < options_.tolerance
            && mu < options_.tolerance) {
            result.status = IpmStatus::Optimal;
            best_z = z;
            best_y = y;
            break;
        }
        if (!std::isfinite(mu) || !std::isfinite(result.primal_infeasibility)
            || !std::isfinite(result.dual_infeasibility)) {
            result.status = IpmStatus::NumericalFailure;
            result.message = "non-finite residual at iteration " + std::to_string(result.iterations);
            break;
        }

        const Real merit = std::max({result.primal_infeasibility, result.dual_infeasibility, mu});
        if (merit < best_merit) {
            best_merit = merit;
            best_z = z;
            best_y = y;
        }

        // Theta^{-1}_j = lambda_j/g_j + zeta_j/h_j (0 for a free z_j).
        std::vector<Real> theta_inv(static_cast<std::size_t>(N), 0.0);
        for (Idx j = 0; j < N; ++j) {
            const auto k = static_cast<std::size_t>(j);
            if (has_lo[k]) theta_inv[k] += lambda[k] / g[k];
            if (has_hi[k]) theta_inv[k] += zeta[k] / h[k];
        }

        // -- regularization schedule: shrinks with mu, floored so the
        // factorization is never asked to trust a direction that is only
        // safe because two things happen to cancel exactly (Watch out,
        // Build Map #9: this is the single most likely place to compute a
        // silently-slightly-wrong answer). Ceiling keeps it from ever
        // dominating Theta^{-1} once mu is small and precision matters.
        //
        // Backing off (retrying with a larger delta) on an actual
        // factorization breakdown is the direct, standard answer to the
        // ticket's own regularization tradeoff -- too little regularization
        // is exactly what a breakdown means happened, so the fix is more of
        // it, not giving up. `recipe` hit this right at convergence (mu
        // ~7e-9, deep enough into the barrier that some Theta^{-1} entries
        // are enormous and floating-point cancellation during elimination
        // can still produce a near-zero pivot despite the SQD guarantee
        // holding in exact arithmetic) -- one backoff step resolves it there.
        Real delta = std::clamp(mu * 1e-2, options_.min_regularization, Real{1e-6});
        bool factorized = false;
        for (int attempt = 0; attempt < 6; ++attempt, delta *= 10.0) {
            // -- assemble the SQD augmented matrix (lower triangle only) ----
            for (Idx i = 0; i < Ntot; ++i)
                for (Idx j = 0; j <= i; ++j)
                    K.at(i, j) = 0.0;
            for (Idx j = 0; j < N; ++j)
                K.at(j, j) = -(theta_inv[static_cast<std::size_t>(j)] + delta);
            for (Idx i = 0; i < m; ++i)
                K.at(N + i, N + i) = delta;
            // M block: M = [A, -I_m]. Lower-triangle placement: rows
            // N..N+m-1 (the dy block) against columns 0..N-1 (the dz block)
            // -- that IS the lower triangle here since N <= row index.
            {
                const auto row_ptr = scaled.matrix().row_ptr();
                const auto col_idx = scaled.matrix().col_idx();
                const auto values = scaled.matrix().values();
                for (Idx i = 0; i < m; ++i) {
                    const Idx end = row_ptr[static_cast<std::size_t>(i) + 1];
                    for (Idx t = row_ptr[static_cast<std::size_t>(i)]; t < end; ++t) {
                        const Idx j = col_idx[static_cast<std::size_t>(t)];
                        K.at(N + i, j) = values[static_cast<std::size_t>(t)];
                    }
                    K.at(N + i, n + i) = -1.0;
                }
            }
            if (K.factorize() == Status::Ok) { factorized = true; break; }
            if (options_.verbose)
                std::fprintf(stderr, "  [ipm] factorization broke down at delta=%.3g, "
                                     "backing off\n", static_cast<double>(delta));
        }

        if (!factorized) {
            result.status = IpmStatus::NumericalFailure;
            result.message = "SQD factorization broke down at iteration "
                            + std::to_string(result.iterations)
                            + " even after backing off regularization several times -- "
                              "a real numerical failure, not just a hard instance";
            break;
        }

        // -- predictor (affine step, sigma = 0) -----------------------------
        {
            std::vector<Real> rhs(static_cast<std::size_t>(Ntot));
            std::vector<Real> aty(static_cast<std::size_t>(n), 0.0);
            scaled.matrix().multiply_transpose(y, aty);
            for (Idx j = 0; j < n; ++j)
                rhs[static_cast<std::size_t>(j)] = cz[static_cast<std::size_t>(j)]
                                                  - aty[static_cast<std::size_t>(j)];
            for (Idx i = 0; i < m; ++i)
                rhs[static_cast<std::size_t>(n + i)] =
                    cz[static_cast<std::size_t>(n + i)] + y[static_cast<std::size_t>(i)];
            for (Idx i = 0; i < m; ++i)
                rhs[static_cast<std::size_t>(N + i)] = -rp[static_cast<std::size_t>(i)];

            K.solve(rhs);
            for (Idx j = 0; j < N; ++j) dz_aff[static_cast<std::size_t>(j)] = rhs[static_cast<std::size_t>(j)];
            for (Idx i = 0; i < m; ++i) dy_aff[static_cast<std::size_t>(i)] = rhs[static_cast<std::size_t>(N + i)];

            for (Idx j = 0; j < N; ++j) {
                const auto k = static_cast<std::size_t>(j);
                if (has_lo[k])
                    dlambda_aff[k] = -lambda[k] - (lambda[k] / g[k]) * dz_aff[k];
                if (has_hi[k])
                    dzeta_aff[k] = -zeta[k] + (zeta[k] / h[k]) * dz_aff[k];
            }
        }

        // affine step lengths (fraction-to-boundary against 1.0, not yet the
        // configured safety margin -- that only applies to the FINAL step).
        auto max_step = [&](const std::vector<Real>& val, const std::vector<Real>& dval,
                            const std::vector<char>& has, Real limit) -> Real {
            Real a = limit;
            for (Idx j = 0; j < N; ++j) {
                const auto k = static_cast<std::size_t>(j);
                if (!has[k]) continue;
                if (dval[k] < 0.0) a = std::min(a, -val[k] / dval[k]);
            }
            return a;
        };
        const Real alpha_p_aff = std::min(max_step(g, dz_aff, has_lo, 1.0),
                                          [&] { std::vector<Real> ndz(dz_aff.size());
                                                for (std::size_t i = 0; i < ndz.size(); ++i) ndz[i] = -dz_aff[i];
                                                return max_step(h, ndz, has_hi, 1.0); }());
        const Real alpha_d_aff = std::min(max_step(lambda, dlambda_aff, has_lo, 1.0),
                                          max_step(zeta, dzeta_aff, has_hi, 1.0));

        Real mu_aff = 0.0;
        for (Idx j = 0; j < N; ++j) {
            const auto k = static_cast<std::size_t>(j);
            if (has_lo[k]) mu_aff += (lambda[k] + alpha_d_aff * dlambda_aff[k])
                                    * (g[k] + alpha_p_aff * dz_aff[k]);
            if (has_hi[k]) mu_aff += (zeta[k] + alpha_d_aff * dzeta_aff[k])
                                    * (h[k] - alpha_p_aff * dz_aff[k]);
        }
        mu_aff /= static_cast<Real>(n_terms);
        Real sigma = mu > 1e-300 ? std::pow(std::clamp(mu_aff / mu, Real{0}, Real{1}), 3) : 0.0;

        // -- corrector (centering + Mehrotra second-order correction) -------
        std::vector<Real> dz_step, dy_step;
        {
            std::vector<Real> rhs(static_cast<std::size_t>(Ntot));
            std::vector<Real> aty(static_cast<std::size_t>(n), 0.0);
            scaled.matrix().multiply_transpose(y, aty);

            for (Idx j = 0; j < N; ++j) {
                const auto k = static_cast<std::size_t>(j);
                Real term = 0.0;   //  mu_g_j/g_j - mu_h_j/h_j
                if (has_lo[k]) {
                    const Real mu_g = sigma * mu - dz_aff[k] * dlambda_aff[k];
                    term += mu_g / g[k];
                }
                if (has_hi[k]) {
                    const Real mu_h = sigma * mu + dz_aff[k] * dzeta_aff[k];
                    term -= mu_h / h[k];
                }
                rhs[k] = rd[k] - term + lambda[k] - zeta[k];
            }
            for (Idx i = 0; i < m; ++i)
                rhs[static_cast<std::size_t>(N + i)] = -rp[static_cast<std::size_t>(i)];

            K.solve(rhs);
            dz_step.assign(rhs.begin(), rhs.begin() + N);
            dy_step.assign(rhs.begin() + N, rhs.end());
        }

        std::vector<Real> dlambda_step(static_cast<std::size_t>(N), 0.0);
        std::vector<Real> dzeta_step(static_cast<std::size_t>(N), 0.0);
        for (Idx j = 0; j < N; ++j) {
            const auto k = static_cast<std::size_t>(j);
            if (has_lo[k]) {
                const Real mu_g = sigma * mu - dz_aff[k] * dlambda_aff[k];
                dlambda_step[k] = mu_g / g[k] - lambda[k] - (lambda[k] / g[k]) * dz_step[k];
            }
            if (has_hi[k]) {
                const Real mu_h = sigma * mu + dz_aff[k] * dzeta_aff[k];
                dzeta_step[k] = mu_h / h[k] - zeta[k] + (zeta[k] / h[k]) * dz_step[k];
            }
        }

        // alpha = min(1, max_step_fraction * raw_boundary_step): capping
        // max_step's own `limit` at 1/fraction before the multiply gets this
        // exactly -- when the raw step is the binding one, fraction*raw comes
        // straight through; when raw is larger (or infinite, no blocking
        // bound in that direction), the cap makes fraction*(1/fraction) = 1
        // instead of overshooting.
        std::vector<Real> ndz_step(N);
        for (Idx j = 0; j < N; ++j) ndz_step[static_cast<std::size_t>(j)] = -dz_step[static_cast<std::size_t>(j)];
        const Real alpha_p = options_.max_step_fraction
                            * std::min(max_step(g, dz_step, has_lo, 1.0 / options_.max_step_fraction),
                                      max_step(h, ndz_step, has_hi, 1.0 / options_.max_step_fraction));
        const Real alpha_d = options_.max_step_fraction
                            * std::min(max_step(lambda, dlambda_step, has_lo, 1.0 / options_.max_step_fraction),
                                      max_step(zeta, dzeta_step, has_hi, 1.0 / options_.max_step_fraction));

        for (Idx j = 0; j < N; ++j) {
            const auto k = static_cast<std::size_t>(j);
            z[k] += alpha_p * dz_step[k];
            if (has_lo[k]) { g[k] += alpha_p * dz_step[k]; lambda[k] += alpha_d * dlambda_step[k]; }
            if (has_hi[k]) { h[k] -= alpha_p * dz_step[k]; zeta[k] += alpha_d * dzeta_step[k]; }
        }
        for (Idx i = 0; i < m; ++i) y[static_cast<std::size_t>(i)] += alpha_d * dy_step[static_cast<std::size_t>(i)];
    }

    if (result.status == IpmStatus::NotSolved) result.status = IpmStatus::IterationLimit;
    if (best_z.empty()) { best_z = z; best_y = y; }   // failed before iteration 0 ever ran

    // Unscale back to the ORIGINAL problem's units before reporting anything
    // -- best_z, best_y are the SCALED-space best iterate (see the
    // Scaling::equilibrate call at the top), same convention as Pdhg::solve.
    result.primal.assign(best_z.begin(), best_z.begin() + n);
    scaling.unscale_primal(result.primal);

    result.dual.resize(static_cast<std::size_t>(m));
    for (Idx i = 0; i < m; ++i) result.dual[static_cast<std::size_t>(i)] = obj_sign * best_y[static_cast<std::size_t>(i)];
    scaling.unscale_dual(result.dual);

    result.objective = problem.evaluate_objective(result.primal);

    result.reduced_costs.assign(static_cast<std::size_t>(n), 0.0);
    {
        std::vector<Real> aty(static_cast<std::size_t>(n), 0.0);
        problem.matrix().multiply_transpose(result.dual, aty);
        for (Idx j = 0; j < n; ++j) {
            const auto k = static_cast<std::size_t>(j);
            result.reduced_costs[k] = problem.objective()[k] - aty[k];
        }
    }

    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
}

}  // namespace sov
