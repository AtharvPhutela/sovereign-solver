// Preconditioning tests -- Build Map ticket #6.
//
// The ticket's pass condition: "preconditioning measurably improves
// conditioning on a deliberately ill-scaled instance". So the central test
// builds a matrix whose coefficients span 1e-6 to 1e+6 and checks that the
// spread of row and column magnitudes collapses toward 1 -- and, just as
// important, that solving the scaled problem still gives the original answer.
#include <cmath>
#include <vector>

#include "sovereign/io.hpp"
#include "sovereign/scaling.hpp"
#include "sovereign/simplex.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

constexpr Real kTol = 1e-9;

/// A 4x4 matrix with a deliberately vicious coefficient spread.
CsrMatrix ill_scaled() {
    const std::vector<Triplet> t = {
        {0, 0, 1e6},  {0, 1, 1e-3},
        {1, 1, 1e-6}, {1, 2, 1e3},
        {2, 0, 1e-4}, {2, 2, 1e2},  {2, 3, 1.0},
        {3, 1, 1e4},  {3, 3, 1e-5},
    };
    return CsrMatrix::from_triplets(4, 4, t);
}

}  // namespace

TEST(scaling, ruiz_collapses_the_coefficient_spread) {
    const CsrMatrix A = ill_scaled();
    const Scaling::Conditioning before = Scaling::measure(A);
    CHECK(before.coeff_spread > 1e10);         // 1e6 / 1e-6

    ScalingOptions opt;
    opt.apply_pock_chambolle = false;         // isolate Ruiz
    const Scaling s = Scaling::equilibrate(A, opt);
    const CsrMatrix scaled = s.apply(A);
    const Scaling::Conditioning after = Scaling::measure(scaled);

    // Ruiz's actual guarantee: every row and column infinity-norm is driven
    // toward 1, so the SPREAD of those norms essentially vanishes. This is the
    // conditioning measure the ticket names ("row/column norms driven toward
    // unity").
    CHECK_MSG(after.row_norm_spread < 1.5,
              "row-norm spread after Ruiz: " + std::to_string(after.row_norm_spread));
    CHECK_MSG(after.col_norm_spread < 1.5,
              "col-norm spread after Ruiz: " + std::to_string(after.col_norm_spread));
    // The raw coefficient spread also improves substantially, though diagonal
    // scaling cannot fully flatten it -- a lone huge entry sharing its row and
    // column with a lone tiny one is a fixed point. A hundredfold reduction is
    // the "measurable" bar.
    CHECK_MSG(after.coeff_spread < before.coeff_spread / 100.0,
              "coeff spread only fell from " + std::to_string(before.coeff_spread)
              + " to " + std::to_string(after.coeff_spread));
}

TEST(scaling, pock_chambolle_runs_on_the_equilibrated_matrix) {
    const CsrMatrix A = ill_scaled();
    const Scaling s = Scaling::equilibrate(A, {});   // both schemes
    const CsrMatrix scaled = s.apply(A);
    const Scaling::Conditioning after = Scaling::measure(scaled);
    // Still markedly better conditioned than the input, even though
    // Pock-Chambolle tunes for PDHG step sizes rather than norm equilibration.
    CHECK(after.coeff_spread < Scaling::measure(A).coeff_spread / 100.0);
    CHECK(after.row_norm_spread < Scaling::measure(A).row_norm_spread);
    for (Real f : s.row_scale()) { CHECK(f > 0.0); CHECK(std::isfinite(f)); }
    for (Real f : s.col_scale()) { CHECK(f > 0.0); CHECK(std::isfinite(f)); }
}

TEST(scaling, all_zero_row_or_column_keeps_its_factor) {
    // Row 1 and column 1 are entirely empty. Scaling must not divide by zero
    // or emit a non-finite factor.
    const std::vector<Triplet> t = {{0, 0, 5.0}, {2, 2, 7.0}};
    const CsrMatrix A = CsrMatrix::from_triplets(3, 3, t);
    const Scaling s = Scaling::equilibrate(A, {});
    for (Real f : s.row_scale()) CHECK(std::isfinite(f) && f > 0.0);
    for (Real f : s.col_scale()) CHECK(std::isfinite(f) && f > 0.0);
    CHECK_NEAR(s.row_scale()[1], 1.0, kTol);
    CHECK_NEAR(s.col_scale()[1], 1.0, kTol);
}

TEST(scaling, apply_rejects_a_size_mismatch) {
    const Scaling s = Scaling::equilibrate(ill_scaled(), {});
    const std::vector<Triplet> one = {{0, 0, 1.0}};
    const CsrMatrix other = CsrMatrix::from_triplets(2, 2, one);
    CHECK_THROWS(s.apply(other));
}

// --------------------------------------------------------------------------
// the property that actually matters: scaling does not change the answer
// --------------------------------------------------------------------------

TEST(scaling, scaled_problem_has_the_same_optimum) {
    // A small ill-scaled LP:  min  1e3 x + 1e-3 y
    //                         s.t. 1e-4 x + 1e4 y >= 1
    //                              1e5 x - 1e-2 y <= 1e6
    //                              x, y >= 0
    const char* text = R"(Minimize
 obj: 1000 x + 0.001 y
Subject To
 c1: 0.0001 x + 10000 y >= 1
 c2: 100000 x - 0.01 y <= 1000000
End
)";
    const ReadResult r = read_lp_string(text);
    const Problem& original = r.problem;

    Simplex plain;
    const SimplexResult a = plain.solve(original);
    CHECK(a.status == SolveStatus::Optimal);

    auto [scaled_problem, scaling] = Scaling::equilibrate(original, {});
    const Scaling::Conditioning before = Scaling::measure(original.matrix());
    const Scaling::Conditioning after = Scaling::measure(scaled_problem.matrix());
    CHECK_MSG(after.coeff_spread < before.coeff_spread,
              "scaling did not improve the coefficient spread");

    Simplex on_scaled;
    SimplexResult b = on_scaled.solve(scaled_problem);
    CHECK(b.status == SolveStatus::Optimal);

    // Map the scaled solution back and confirm it is the original optimum.
    scaling.unscale_primal(b.primal);
    CHECK_NEAR(a.objective, b.objective, 1e-6);
    CHECK_NEAR(original.evaluate_objective(b.primal), a.objective, 1e-6);
    for (Idx j = 0; j < original.num_cols(); ++j)
        CHECK_NEAR(a.primal[static_cast<std::size_t>(j)],
                   b.primal[static_cast<std::size_t>(j)], 1e-6);
}

TEST(scaling, disabling_both_schemes_is_the_identity) {
    ScalingOptions off;
    off.apply_ruiz = false;
    off.apply_pock_chambolle = false;
    const Scaling s = Scaling::equilibrate(ill_scaled(), off);
    for (Real f : s.row_scale()) CHECK_NEAR(f, 1.0, kTol);
    for (Real f : s.col_scale()) CHECK_NEAR(f, 1.0, kTol);
}

TST_MAIN()
