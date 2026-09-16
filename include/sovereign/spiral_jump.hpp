// Spiral-axis vertex jump -- Build Map ticket #12, gate M2 (stretch, SEED /
// FRONTIER). Bible S4.3, S5.2 (Bet 2), S10 open-questions item 1.
//
// THE IDEA. PDLP's own literature documents that PDHG iterates near the
// optimal face follow (approximately) a linear recurrence z_{k+1} = M z_k + b
// whose dominant mode is often a COMPLEX-CONJUGATE eigenvalue pair: the error
// e_k := z_k - z* rotates around the fixed point z* while contracting toward
// it -- a damped spiral, not a straight-line approach. This is exactly why
// restarts (ticket #8) help: averaging over a full rotation cancels the
// rotational component. Ticket #11's checkpoint crossover treats every
// checkpoint's iterate as an independent basis guess; this ticket instead
// tries to read the spiral's geometry off a short window of checkpoints and
// solve directly for the point it is spiraling into, without waiting out any
// more of the rotation.
//
// THE METHOD (vector extrapolation, self-derived from the linear-recurrence
// assumption above -- not lifted from any specific solver's crossover code).
// If e_k = z_k - z* evolves as e_{k+1} = M e_k for a FIXED (unknown) operator
// M whose minimal polynomial has degree <= 2 -- true for a single real
// eigenvalue or a complex-conjugate PAIR, i.e. exactly the spiral case --
// then the consecutive difference vectors u_j := z_{k+j+1} - z_{k+j} satisfy
// the SAME order-2 recurrence, because differencing is linear:
//     u_{j+2} = c1 u_{j+1} + c0 u_j
// for fixed scalars c0, c1 (the recurrence's characteristic polynomial
// lambda^2 - c1 lambda - c0 shares M's eigenvalues -- a complex-conjugate
// root pair here IS the spiral's rotation rate and decay rate). Given 4
// consecutive iterates z_k..z_{k+3} (3 difference vectors u_0, u_1, u_2), c0
// and c1 are over-determined (u's are high-dimensional vectors; the scalar
// relationship is only 2 unknowns) and solved by least squares. This is
// exactly Sidi's Minimal Polynomial Extrapolation (MPE) at order p=2: with
// c_2 := 1 and gamma_j := c_j / (c_0+c_1+c_2), the point the recurrence
// converges to is
//     z* ~= gamma_0 z_k + gamma_1 z_{k+1} + gamma_2 z_{k+2}
// -- a WEIGHTED AVERAGE of the first three iterates (weights can be negative,
// which is what makes this extrapolation rather than interpolation), with no
// further iteration run at all.
//
// WHEN THIS IS TRUSTWORTHY, AND WHEN IT ISN'T. When the assumption genuinely
// holds -- M really does have a degree-2 minimal polynomial in the window
// examined -- this lands on z* to floating-point precision (see the unit
// test against a synthetic spiral). When it does not -- more than one
// significant mode is still active, the point is not yet in the asymptotic
// regime, or PDHG's true local behavior on this instance just isn't a clean
// 2D spiral -- it lands somewhere else, possibly a poor guess. Bible S10's
// own open-questions register names this directly: "may not generalize
// beyond well-behaved LPs." The mitigation is architectural, not
// statistical: like every ticket #11 checkpoint, this is fed to Simplex as
// ANOTHER warm-start guess (crossover.hpp), never trusted directly -- a bad
// jump costs pivots, never correctness.
#pragma once

#include <span>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/problem.hpp"   // kInfinity

namespace sov {

struct SpiralJumpResult {
    bool available = false;    ///< false: history too short, or the fit is ill-conditioned
    std::vector<Real> x;       ///< extrapolated primal, same space as the input history
    std::vector<Real> y;       ///< extrapolated dual, same space as the input history

    /// ||c0*u0 + c1*u1 + u2|| / max(||u2||, tiny): the least-squares fit's
    /// own residual on the difference it was NOT fit to use directly (u2
    /// only enters the right-hand side) -- how well a degree-2 recurrence
    /// actually explains this window, not just how well two free parameters
    /// can be made to fit two vectors. Near 0: a clean spiral. Near or above
    /// 1: the model does not explain this window meaningfully better than
    /// guessing u2 ~= 0 would, and the jump should be distrusted. Exposed
    /// rather than thresholded internally so a caller sets its own cutoff.
    Real fit_residual = kInfinity;
};

/// Fits the order-2 minimal polynomial extrapolation described above to a
/// window of exactly 4 consecutive (x, y) checkpoints (oldest first, i.e.
/// x_history[0]/y_history[0] is z_k) and returns the point that recurrence
/// would converge to. `available` is false (x/y left empty) when:
///   - the window is not exactly 4 points, or the per-point x/y lengths
///     are inconsistent across the window,
///   - any input value is non-finite,
///   - the 2x2 normal-equations system is too ill-conditioned to trust
///     (near-parallel difference vectors -- the degenerate case where the
///     window carries no rotational information to extract, including the
///     case where PDHG has already converged and every difference is ~0),
///   - the extrapolated point itself comes out non-finite.
SpiralJumpResult estimate_spiral_jump(std::span<const std::vector<Real>> x_history,
                                      std::span<const std::vector<Real>> y_history);

}  // namespace sov
