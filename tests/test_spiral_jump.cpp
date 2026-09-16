// Spiral-axis vertex jump tests -- Build Map ticket #12.
//
// This is the standalone math check: does the order-2 extrapolation actually
// recover the fixed point of a genuine complex-eigenvalue (rotating,
// contracting) linear recurrence, independent of whether real PDHG runs
// happen to look like one? See spiral_jump.hpp for the full derivation.
#include <cmath>
#include <vector>

#include "sovereign/spiral_jump.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

/// Generates 4 consecutive iterates z_k = z_star + e_k where e_{k+1} is e_k
/// rotated by `theta` and scaled by `r` -- the canonical damped-spiral
/// recurrence a complex-conjugate eigenvalue pair r*e^{+-i theta} produces.
std::vector<std::vector<Real>> synthetic_spiral(Real zx, Real zy, Real e0x, Real e0y,
                                                Real r, Real theta) {
    std::vector<std::vector<Real>> z(4);
    Real ex = e0x, ey = e0y;
    for (int k = 0; k < 4; ++k) {
        z[static_cast<std::size_t>(k)] = {zx + ex, zy + ey};
        const Real nx = r * (std::cos(theta) * ex - std::sin(theta) * ey);
        const Real ny = r * (std::sin(theta) * ex + std::cos(theta) * ey);
        ex = nx; ey = ny;
    }
    return z;
}

}  // namespace

TEST(spiral_jump, recovers_the_exact_fixed_point_of_a_synthetic_spiral) {
    const std::vector<std::vector<Real>> x = synthetic_spiral(3.0, 4.0, 1.0, 0.0, 0.7, 0.9);
    const std::vector<std::vector<Real>> y(4);   // m = 0: an empty dual block is fine

    const SpiralJumpResult r = estimate_spiral_jump(x, y);
    CHECK(r.available);
    CHECK_NEAR(r.x[0], 3.0, 1e-6);
    CHECK_NEAR(r.x[1], 4.0, 1e-6);
    CHECK(r.fit_residual < 1e-6);
}

TEST(spiral_jump, recovers_the_fixed_point_with_a_nonempty_dual_block_too) {
    // The primal and dual blocks are extrapolated jointly (one concatenated
    // state vector, per the header's own derivation) -- give them DIFFERENT
    // spiral parameters to confirm each block is handled on its own terms,
    // not coupled incorrectly.
    const std::vector<std::vector<Real>> x = synthetic_spiral(1.0, -2.0, 0.5, 0.5, 0.6, 1.2);
    const std::vector<std::vector<Real>> y = synthetic_spiral(-5.0, 10.0, -0.3, 0.1, 0.6, 1.2);

    const SpiralJumpResult r = estimate_spiral_jump(x, y);
    CHECK(r.available);
    CHECK_NEAR(r.x[0], 1.0, 1e-6);
    CHECK_NEAR(r.x[1], -2.0, 1e-6);
    CHECK_NEAR(r.y[0], -5.0, 1e-6);
    CHECK_NEAR(r.y[1], 10.0, 1e-6);
}

TEST(spiral_jump, a_window_of_the_wrong_size_is_unavailable) {
    std::vector<std::vector<Real>> x = {{1.0}, {2.0}, {3.0}};   // only 3 points
    std::vector<std::vector<Real>> y(3);
    const SpiralJumpResult r = estimate_spiral_jump(x, y);
    CHECK(!r.available);
    CHECK(r.x.empty());
}

TEST(spiral_jump, inconsistent_lengths_across_the_window_are_unavailable) {
    std::vector<std::vector<Real>> x = {{1.0, 2.0}, {2.0}, {3.0, 4.0}, {4.0, 5.0}};
    std::vector<std::vector<Real>> y(4);
    const SpiralJumpResult r = estimate_spiral_jump(x, y);
    CHECK(!r.available);
}

TEST(spiral_jump, non_finite_history_is_rejected) {
    std::vector<std::vector<Real>> x = {{1.0}, {kInfinity}, {3.0}, {4.0}};
    std::vector<std::vector<Real>> y(4);
    const SpiralJumpResult r = estimate_spiral_jump(x, y);
    CHECK(!r.available);
}

TEST(spiral_jump, an_already_converged_sequence_has_no_rotation_to_extract) {
    // Every difference is exactly zero -- there is no information in the
    // window at all, and the method must decline rather than divide by a
    // near-zero determinant and report a meaningless "extrapolation".
    std::vector<std::vector<Real>> x = {{5.0, 5.0}, {5.0, 5.0}, {5.0, 5.0}, {5.0, 5.0}};
    std::vector<std::vector<Real>> y(4);
    const SpiralJumpResult r = estimate_spiral_jump(x, y);
    CHECK(!r.available);
}

TEST(spiral_jump, a_pure_single_real_mode_is_correctly_declared_unextractable) {
    // A single real eigenvalue (no rotation) makes every e_k a scalar
    // multiple of the SAME direction -- the three difference vectors are all
    // collinear, the 2x2 normal-equations system is exactly singular, and
    // this order-2 method has no more information to offer than plain
    // Aitken extrapolation would. Declining here (rather than returning
    // something built on a singular solve) is the honest behavior; ticket
    // #11's own checkpoint guess is what covers this case instead.
    std::vector<std::vector<Real>> x;
    Real e = 1.0;
    for (int k = 0; k < 4; ++k) {
        x.push_back({10.0 + e});
        e *= 0.5;
    }
    std::vector<std::vector<Real>> y(4);
    const SpiralJumpResult r = estimate_spiral_jump(x, y);
    CHECK(!r.available);
}

TST_MAIN()
