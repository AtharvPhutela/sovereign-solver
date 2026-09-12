// Regularized pivoting-free interior-point method -- Build Map ticket #9,
// gate M1. Bible S4.2 Engine B.
//
// PURPOSE. This is the HIGH-PRECISION continuous engine: where PDHG (#8)
// settles for ~1e-4, this engine is expected to reach near-simplex accuracy.
// It is what "no pivoting on the critical path" is a claim ABOUT -- Symmetric
// Quasi-Definite (SQD) systems admit an LDL^T factorization in ANY diagonal
// order, including the natural (unpermuted) one, without a numerical pivot
// search (Vanderbei 1995). That is the whole point of regularizing the KKT
// system into SQD form rather than factoring the raw, indefinite one.
//
// DERIVATION (self-contained; standard bounded-variable primal-dual IPM
// theory, not borrowed from any specific solver's source).
//
// Same variable convention as simplex.cpp/pdhg.hpp: z = (x, s) in R^{n+m}
// with the row bounds moved onto the slack s := A x, so the model is
//     minimize   c_z^T z    s.t.  M z = 0,   lo_z <= z <= hi_z
// where M = [A, -I_m] (m x (n+m)), c_z = (c, 0).
//
// For each z_j with a finite lower bound, track g_j = z_j - lo_j >= 0 and its
// multiplier lambda_j >= 0; for each z_j with a finite upper bound, track
// h_j = hi_j - z_j >= 0 and its multiplier zeta_j >= 0. A variable can have
// either, both, or neither. The barrier-perturbed KKT conditions:
//     M z = 0
//     c_z - M^T y - lambda + zeta = 0
//     lambda_j g_j = mu   (j with a finite lower bound)
//     zeta_j  h_j = mu    (j with a finite upper bound)
//
// Linearizing (Newton's method) and eliminating dlambda, dzeta in favor of dz
// (the standard reduction -- see ipm.cpp for the exact algebra) collapses
// this to one symmetric system in (dz, dy):
//     [ -(Theta^{-1} + delta_x I)      M^T          ] [dz]   [ rhs_z ]
//     [  M                              delta_y I    ] [dy] = [ rhs_y ]
// where Theta_j^{-1} = lambda_j/g_j + zeta_j/h_j (0 for a free z_j). Without
// delta_x, delta_y this is exactly the classical IPM augmented system: (1,1)
// block negative *semi*-definite, (2,2) block zero -- quasi-definite only in
// the boundary sense, and a zero eigenvalue anywhere is what pivoting exists
// to route around. Adding delta_x > 0 to the (1,1) diagonal and delta_y > 0
// to the (2,2) diagonal makes both blocks strictly definite: this IS the SQD
// system the ticket asks for, and it is genuinely a different (regularized)
// system, not a numerical trick played on the same one -- see ipm.cpp's
// regularization schedule for how small these are kept and why that size was
// chosen empirically against the oracle rather than assumed safe.
//
// FACTORIZATION SCOPE. A dense, from-scratch, no-pivoting LDL^T of the
// (n + 2m) x (n + 2m) augmented matrix, refactorized every iteration (the
// diagonal changes every iteration; there is no analogue of the simplex's
// eta-file here). This is the same scope choice ticket #4 made for its own
// dense LU: correct and complete for the small-to-medium instances an oracle
// engine needs, O((n+2m)^3) per iteration, refused past a size guard rather
// than ground through. Sparse LDL^T (cuDSS on GPU, or a from-scratch
// Markowitz-ordered CPU version) is named follow-up work, exactly as sparse
// LU was for #4 -- not half-attempted here.
#pragma once

#include <string>
#include <vector>

#include "sovereign/cancellation.hpp"
#include "sovereign/duals.hpp"
#include "sovereign/numeric.hpp"
#include "sovereign/problem.hpp"

namespace sov {

enum class IpmStatus {
    Optimal,
    IterationLimit,
    TimeLimit,
    NumericalFailure,
    NotSolved,
    Cancelled,   ///< a CancellationToken fired (ticket #10's engine race)
};

const char* to_string(IpmStatus s) noexcept;

struct IpmOptions {
    /// Tight -- this is the high-precision engine. PDHG's ~1e-4 is the
    /// number to beat, not to match. 1e-8, not 1e-9: on `adlittle` the
    /// primal residual plateaus at ~2e-8 (a floating-point floor for that
    /// instance's conditioning, not a bug -- the objective and dual residual
    /// are both already far tighter than this by the same iteration) and a
    /// 1e-9 target never actually flips to Optimal despite the answer
    /// already being correct to that instance's achievable precision.
    Real tolerance = 5e-8;

    // 200 is enough for most small/well-conditioned instances (Netlib's
    // afiro, blend, sc50a/b, kb2, ... converge in 10-60), but several
    // harder-conditioned ones genuinely need several hundred to reach even
    // 5e-8 (bandm ~400, lotfi ~1300) -- verified to actually get there, not
    // assumed. Iterations are cheap relative to giving up early.
    int max_iterations = 1500;
    double time_limit_seconds = 0.0;

    /// Regularization floor added to the (1,1)/(2,2) SQD blocks. Shrinks with
    /// mu (see ipm.cpp) but never below this, so the factorization is never
    /// asked to trust an exactly-singular direction.
    Real min_regularization = 1e-10;

    /// Fraction-to-boundary rule: a step may close at most this fraction of
    /// the gap to a bound in one iteration (standard IPM safeguard -- a step
    /// that reaches a bound exactly makes the next iteration's Theta^{-1}
    /// infinite).
    Real max_step_fraction = 0.995;

    bool verbose = false;
};

struct IpmResult {
    IpmStatus status = IpmStatus::NotSolved;

    Real objective = 0.0;
    std::vector<Real> primal;          ///< x, length num_cols
    std::vector<Real> dual;            ///< y, length num_rows
    std::vector<Real> reduced_costs;   ///< c_j - A_j^T y, length num_cols

    Real primal_infeasibility = 0.0;
    Real dual_infeasibility = 0.0;
    Real complementarity_gap = 0.0;    ///< mu at termination

    int iterations = 0;
    double seconds = 0.0;
    std::string message;
};

class Ipm {
public:
    explicit Ipm(IpmOptions options = {}) : options_(options) {}

    /// Solve the continuous relaxation. Integrality is ignored, same
    /// convention as Simplex::solve and Pdhg::solve. Runs on the host: the
    /// factorization here is dense CPU linear algebra (see ipm.hpp's top
    /// comment on scope) -- there is no GPU path for it yet, unlike PDHG,
    /// which is why this does not take a Backend the way Pdhg::solve does.
    /// `cancel`, if given, is checked once per outer iteration (ticket #10's
    /// engine race) -- a set token returns IpmStatus::Cancelled promptly,
    /// before starting the next dense factorization rather than mid-solve.
    IpmResult solve(const Problem& problem, const CancellationToken* cancel = nullptr) const;

    const IpmOptions& options() const noexcept { return options_; }

private:
    IpmOptions options_;
};

}  // namespace sov
