// PDHG / PDLP engine -- Build Map ticket #8, gate M1. Bible S4.2 Engine A.
//
// PURPOSE. The first-order, GPU-native continuous engine: each iteration is
// nothing but two SpMVs and two elementwise box projections, all through the
// L0 Backend abstraction (include/sovereign/backend.hpp) -- so this same code
// runs on Host today and on Cuda/Hip unchanged once a toolkit compiles them
// (Bible Part III: the algorithm does not know which device it is on).
//
// DERIVATION (self-contained, no borrowed solver code). The model is
//     minimize   c^T x                 in the problem's own sense
//     subject to row_lo <= A x <= row_hi,   col_lo <= x <= col_hi
// Introduce a slack s := A x with the row bounds moved onto it, so the model
// becomes an EQUALITY constraint plus two independent box constraints:
//     minimize_{x in X, s in S}   c^T x     s.t.  s = A x
// where X = [col_lo,col_hi]^n, S = [row_lo,row_hi]^m. Dualizing the equality
// with multiplier y gives the saddle point
//     min_{x in X, s in S}  max_{y}   c^T x + y^T (s - A x)
// (the max over an unconstrained y forces s = Ax exactly, or the bracket is
// +-infinity -- that is what makes the equality a hard constraint, not a
// penalty). Chambolle-Pock's primal-dual hybrid gradient applied to this
// saddle point is exactly the iteration implemented in pdhg.cpp:
//     x <- Proj_X( x - tau  (c - A^T y) )
//     s <- Proj_S( s - tau  y )
//     y <- y + sigma ( (2s-s_old) - A (2x-x_old) )
// -- an SpMV-transpose, an SpMV, and two box projections, matching the
// Bible's "each iteration = SpMV + elementwise clamp" exactly. The reduced
// cost c - A^T y here is deliberately the SAME sign convention ticket #7's
// DualSolution uses (see duals.hpp), so PDHG's y and reduced costs are
// directly comparable to the simplex's without a translation layer -- this
// is what lets #10's engine race and later #26's Benders treat every engine's
// duals identically.
//
// STEP SIZES. Ruiz + Pock-Chambolle (ticket #6) is run as the mandatory
// front-end Bible S6.2 requires: PDHG iterates on the RESCALED problem, not
// the original. Convergence needs tau * sigma * ||A_scaled||^2 <= 1; rather
// than assert the rescaled operator norm is safely below 1 (Scaling's own
// diagonal formula does not account for the slack block's identity column,
// so that bound is not exact), pdhg.cpp measures the actual operator norm of
// the augmented [ -A_scaled | I ] map by power iteration -- a few extra SpMVs
// at setup, using the same primitives as the main loop, no new machinery --
// and sets tau = sigma = a safety factor over that measured value. Honest,
// not assumed.
//
// ACCURACY. Native PDHG accuracy is ~1e-4 (Bible S4.2 Engine A) -- this is
// NOT where exact vertices come from; that is crossover (#11/#12). Restarts
// (see below) shorten the tail to reach that accuracy sooner; they do not
// change the target accuracy itself.
//
// RESTARTS. PDHG has documented two-stage convergence: a fast initial phase,
// then a long, slow tail. The mitigation implemented here is the general,
// well-known "restart to the better of the running average" idea for
// primal-dual/accelerated first-order methods (self-derived from that
// principle, not lifted from any specific implementation): a Cesaro average
// of (x, y) is maintained since the last restart, and every
// `restart_check_period` iterations the KKT residual of the average is
// compared against the residual at the last restart. A sufficient decrease
// triggers a restart to the average, which resets the averaging window --
// this is what rescues the tail (Bible S10.1's fallback if it does not
// generalize is crossover, unaffected either way).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "sovereign/backend.hpp"
#include "sovereign/cancellation.hpp"
#include "sovereign/duals.hpp"
#include "sovereign/numeric.hpp"
#include "sovereign/problem.hpp"

namespace sov {

enum class PdhgStatus {
    Optimal,
    IterationLimit,
    TimeLimit,
    NumericalFailure,
    NotSolved,
    Cancelled,   ///< a CancellationToken fired (ticket #10's engine race)
};

const char* to_string(PdhgStatus s) noexcept;

struct PdhgOptions {
    /// First-order tolerance on primal infeasibility, dual (reduced-cost)
    /// infeasibility, and the normalized duality gap. ~1e-4 is PDHG's native
    /// accuracy (Bible S4.2 Engine A) -- do not expect simplex-grade digits.
    Real tolerance = 1e-4;

    // PDHG's documented slow tail (Bible S4.2 Engine A) means a harder-
    // conditioned small instance can genuinely need several hundred thousand
    // iterations to reach even 1e-4 -- adlittle needs ~394k, share2b ~774k,
    // both verified to actually reach the oracle's objective at that budget.
    // Iterations are cheap (SpMV + a clamp); this is not tuned per instance.
    long long max_iterations = 1000000;
    double time_limit_seconds = 0.0;

    /// How often (in iterations) to check convergence and the restart
    /// condition. Checking every iteration would mean an unscale + a handful
    /// of reductions every step; checking rarely delays both a converged
    /// return and a helpful restart. A few dozen is the usual range.
    long long restart_check_period = 40;

    /// A restart fires when the running average's KKT residual has fallen to
    /// at most this fraction of the residual at the last restart.
    Real restart_sufficient_decrease = 0.5;

    /// Safety factor applied to the power-iteration operator-norm estimate:
    /// tau = sigma = safety / ||K||_est. Below 1 so the *measured* estimate
    /// (itself approximate after a handful of power-iteration steps) still
    /// leaves margin against the tau*sigma*||K||^2 <= 1 requirement.
    Real step_size_safety = 0.9;

    /// Power-iteration steps for the operator-norm estimate. Cheap relative
    /// to the main loop and only paid once, at setup.
    int power_iterations = 30;

    bool verbose = false;

    /// Ticket #11 (concurrent checkpoint crossover). If set, called every
    /// `restart_check_period` iterations with the ORIGINAL-space candidate
    /// point this iteration already unscaled and checked for convergence --
    /// the callback is free of any extra unscaling work PDHG was not already
    /// doing. `original_residual` is that candidate's residual against the
    /// real problem (what the loop itself compares to `tolerance`).
    ///
    /// Runs synchronously ON THE PDHG THREAD, between one iteration and the
    /// next -- it must return promptly (hand the point to another thread and
    /// return, never solve anything here) or it becomes the very serial
    /// bottleneck concurrent crossover exists to avoid.
    std::function<void(long long iteration, const std::vector<Real>& x_original,
                       const std::vector<Real>& y_original, Real original_residual)> checkpoint;
};

struct PdhgResult {
    PdhgStatus status = PdhgStatus::NotSolved;

    Real objective = 0.0;
    std::vector<Real> primal;          ///< x, length num_cols, ORIGINAL units
    std::vector<Real> dual;            ///< y, length num_rows, ORIGINAL units
    std::vector<Real> reduced_costs;   ///< c_j - A_j^T y, length num_cols

    Real primal_infeasibility = 0.0;   ///< max row-bound violation of Ax
    Real dual_infeasibility = 0.0;     ///< max complementary-slackness violation
    Real duality_gap = 0.0;            ///< |primal_obj - dual_obj| / (1+|primal_obj|+|dual_obj|)

    long long iterations = 0;
    long long restarts = 0;
    double seconds = 0.0;
    Real operator_norm_estimate = 0.0;
    std::string message;
};

class Pdhg {
public:
    explicit Pdhg(PdhgOptions options = {}) : options_(options) {}

    /// Solve the continuous relaxation on `backend`. Integrality is ignored,
    /// same convention as Simplex::solve. `backend` may be Host today or
    /// Cuda/Hip once compiled -- the algorithm is identical either way.
    /// `cancel`, if given, is checked at the same cadence as the restart
    /// decision (ticket #10's engine race) -- a set token returns
    /// PdhgStatus::Cancelled promptly.
    PdhgResult solve(const Problem& problem, Backend& backend,
                     const CancellationToken* cancel = nullptr) const;

    const PdhgOptions& options() const noexcept { return options_; }

private:
    PdhgOptions options_;
};

}  // namespace sov
