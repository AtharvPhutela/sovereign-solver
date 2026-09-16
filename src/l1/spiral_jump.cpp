// Spiral-axis vertex jump -- Build Map ticket #12. See spiral_jump.hpp for
// the full derivation of the order-2 minimal polynomial extrapolation this
// implements.
#include "sovereign/spiral_jump.hpp"

#include <cmath>

namespace sov {

namespace {

bool all_finite(const std::vector<Real>& v) {
    for (Real x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

/// Dot product of two (x, y)-concatenated state vectors, computed piecewise
/// so callers never actually materialize the concatenation.
Real dot_concat(const std::vector<Real>& ax, const std::vector<Real>& ay,
                const std::vector<Real>& bx, const std::vector<Real>& by) {
    Real acc = 0.0;
    for (std::size_t i = 0; i < ax.size(); ++i) acc += ax[i] * bx[i];
    for (std::size_t i = 0; i < ay.size(); ++i) acc += ay[i] * by[i];
    return acc;
}

void difference(const std::vector<Real>& a, const std::vector<Real>& b, std::vector<Real>& out) {
    out.resize(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = b[i] - a[i];
}

}  // namespace

SpiralJumpResult estimate_spiral_jump(std::span<const std::vector<Real>> x_history,
                                      std::span<const std::vector<Real>> y_history) {
    SpiralJumpResult out;
    if (x_history.size() != 4 || y_history.size() != 4) return out;

    const std::size_t nx = x_history[0].size();
    const std::size_t ny = y_history[0].size();
    for (std::size_t i = 0; i < 4; ++i) {
        if (x_history[i].size() != nx || y_history[i].size() != ny) return out;
        if (!all_finite(x_history[i]) || !all_finite(y_history[i])) return out;
    }

    // Three consecutive difference vectors u_0, u_1, u_2 (each an (x, y)
    // pair), spanning the four iterates z_k .. z_{k+3}.
    std::vector<Real> u0x, u0y, u1x, u1y, u2x, u2y;
    difference(x_history[0], x_history[1], u0x);
    difference(y_history[0], y_history[1], u0y);
    difference(x_history[1], x_history[2], u1x);
    difference(y_history[1], y_history[2], u1y);
    difference(x_history[2], x_history[3], u2x);
    difference(y_history[2], y_history[3], u2y);

    // Least squares: minimize || c0*u0 + c1*u1 + u2 ||^2 over (c0, c1).
    // Normal equations: [[a00,a01],[a01,a11]] [c0;c1] = [b0;b1].
    const Real a00 = dot_concat(u0x, u0y, u0x, u0y);
    const Real a01 = dot_concat(u0x, u0y, u1x, u1y);
    const Real a11 = dot_concat(u1x, u1y, u1x, u1y);
    const Real b0 = -dot_concat(u0x, u0y, u2x, u2y);
    const Real b1 = -dot_concat(u1x, u1y, u2x, u2y);

    // Ill-conditioning guard: reject when u0 and u1 are nearly parallel (or
    // either is ~0, e.g. PDHG has already essentially converged and there is
    // no rotational information left to extract) -- the determinant relative
    // to the diagonal entries' own scale is the right invariant here, not an
    // absolute threshold, since the iterates' own magnitude varies hugely
    // across instances.
    const Real det = a00 * a11 - a01 * a01;
    const Real scale = std::max(a00 * a11, Real{1e-300});
    if (!std::isfinite(det) || std::abs(det) < 1e-10 * scale) return out;

    const Real c0 = (b0 * a11 - b1 * a01) / det;
    const Real c1 = (a00 * b1 - a01 * b0) / det;
    const Real c2 = 1.0;
    const Real sum = c0 + c1 + c2;
    if (!std::isfinite(c0) || !std::isfinite(c1) || std::abs(sum) < 1e-8) return out;

    const Real g0 = c0 / sum, g1 = c1 / sum, g2 = c2 / sum;

    std::vector<Real> jump_x(nx), jump_y(ny);
    for (std::size_t i = 0; i < nx; ++i)
        jump_x[i] = g0 * x_history[0][i] + g1 * x_history[1][i] + g2 * x_history[2][i];
    for (std::size_t i = 0; i < ny; ++i)
        jump_y[i] = g0 * y_history[0][i] + g1 * y_history[1][i] + g2 * y_history[2][i];
    if (!all_finite(jump_x) || !all_finite(jump_y)) return out;

    // The fit's own residual on u2: || c0*u0 + c1*u1 + u2 || / ||u2||. u2 was
    // only ever used as the least-squares right-hand side, never as a basis
    // vector -- so this is a genuine out-of-sample check of whether the
    // degree-2 recurrence explains this window, not a tautology.
    Real resid_sq = 0.0, u2_norm_sq = 0.0;
    for (std::size_t i = 0; i < nx; ++i) {
        const Real r = c0 * u0x[i] + c1 * u1x[i] + u2x[i];
        resid_sq += r * r;
        u2_norm_sq += u2x[i] * u2x[i];
    }
    for (std::size_t i = 0; i < ny; ++i) {
        const Real r = c0 * u0y[i] + c1 * u1y[i] + u2y[i];
        resid_sq += r * r;
        u2_norm_sq += u2y[i] * u2y[i];
    }
    const Real fit_residual = std::sqrt(resid_sq) / std::max(std::sqrt(u2_norm_sq), Real{1e-300});
    if (!std::isfinite(fit_residual)) return out;

    out.available = true;
    out.x = std::move(jump_x);
    out.y = std::move(jump_y);
    out.fit_residual = fit_residual;
    return out;
}

}  // namespace sov
