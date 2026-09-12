// Regularized pivoting-free IPM tests -- Build Map ticket #9.
//
// Same discipline as test_simplex.cpp/test_pdhg.cpp: every expected value is
// hand-derived. Unlike PDHG, this engine is expected to reach near-simplex
// precision (Bible S4.2 Engine B) -- the tolerances here are tight, not
// PDHG's ~1e-4, because that tightness is the entire point of this engine.
#include <cmath>

#include "sovereign/duals.hpp"
#include "sovereign/io.hpp"
#include "sovereign/ipm.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

constexpr Real kTol = 1e-5;   // tight, but leaves margin over the barrier's own convergence noise

IpmResult solve_lp(const char* text, IpmOptions opt = {}) {
    const ReadResult r = read_lp_string(text);
    Ipm ipm(opt);
    return ipm.solve(r.problem);
}

}  // namespace

TEST(ipm, two_variable_optimum_matches_simplex) {
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const IpmResult r = solve_lp(text);
    CHECK(r.status == IpmStatus::Optimal);
    CHECK_NEAR(r.objective, 10.0 / 3.0, kTol);
    CHECK_NEAR(r.primal[0], 8.0 / 3.0, kTol);
    CHECK_NEAR(r.primal[1], 2.0 / 3.0, kTol);
    CHECK(r.primal_infeasibility < 1e-6);
    CHECK(r.dual_infeasibility < 1e-6);
}

TEST(ipm, minimize_with_a_binding_lower_bound) {
    const char* text = "Minimize\n obj: 3 x + 2 y\nSubject To\n"
                       " c1: x + y >= 4\nEnd\n";
    const IpmResult r = solve_lp(text);
    CHECK(r.status == IpmStatus::Optimal);
    CHECK_NEAR(r.objective, 8.0, kTol);
}

TEST(ipm, equality_row) {
    // This is the exact case that caught the real initial-point bug: an
    // equality row's slack has lo == hi (zero gap), and an earlier version
    // of the initial point pushed z outside its own box for exactly this
    // shape. Kept as its own case rather than folded into another, since
    // it is the regression test for that specific failure mode.
    const char* text = "Minimize\n obj: x + y\nSubject To\n"
                       " c1: x + y = 10\nBounds\n x <= 6\nEnd\n";
    const IpmResult r = solve_lp(text);
    CHECK(r.status == IpmStatus::Optimal);
    CHECK_NEAR(r.objective, 10.0, kTol);
}

TEST(ipm, fixed_variable) {
    // A column bound type with the same zero-gap shape as the equality-row
    // case above, but on the column side (BOUNDS FX in MPS, `x = 5` here).
    const char* text = "Minimize\n obj: x + y\nSubject To\n c1: x + y >= 0\n"
                       "Bounds\n x = 5\n y >= 1\nEnd\n";
    const IpmResult r = solve_lp(text);
    CHECK(r.status == IpmStatus::Optimal);
    CHECK_NEAR(r.primal[0], 5.0, kTol);
    CHECK_NEAR(r.objective, 6.0, kTol);
}

TEST(ipm, column_bound_forces_the_optimum) {
    const char* text = "Minimize\n obj: -x\nSubject To\n c1: x + y <= 100\n"
                       "Bounds\n x <= 3\nEnd\n";
    const IpmResult r = solve_lp(text);
    CHECK(r.status == IpmStatus::Optimal);
    CHECK_NEAR(r.objective, -3.0, kTol);
    CHECK_NEAR(r.primal[0], 3.0, kTol);
}

TEST(ipm, agrees_with_a_hand_verified_degenerate_optimum) {
    const char* text = R"(Minimize
 obj: -x1 - x2
Subject To
 c1: x1 <= 4
 c2: x1 + x2 <= 4
 c3: x2 <= 2
End
)";
    const IpmResult r = solve_lp(text);
    CHECK(r.status == IpmStatus::Optimal);
    CHECK_NEAR(r.objective, -4.0, kTol);
}

TEST(ipm, reaches_much_tighter_precision_than_pdhg) {
    // The whole point of this engine (Bible S4.2 Engine B): far tighter than
    // PDHG's ~1e-4 native accuracy.
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const IpmResult r = solve_lp(text);
    CHECK(r.status == IpmStatus::Optimal);
    CHECK(r.primal_infeasibility < 1e-6);
    CHECK(r.dual_infeasibility < 1e-6);
    CHECK(r.complementarity_gap < 1e-6);
}

TEST(ipm, reduced_costs_use_the_same_sign_convention_as_the_simplex) {
    const char* text = "Minimize\n obj: -x\nSubject To\n c1: x + y <= 100\n"
                       "Bounds\n x <= 3\nEnd\n";
    const IpmResult r = solve_lp(text);
    CHECK(r.status == IpmStatus::Optimal);
    CHECK(r.reduced_costs[0] < 1e-4);
}

TEST(ipm, complementary_slackness_holds_at_the_optimum) {
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const ReadResult rr = read_lp_string(text);
    const IpmResult r = solve_lp(text);
    CHECK(r.status == IpmStatus::Optimal);

    DualSolution d;
    d.row_duals = r.dual;
    d.reduced_costs = r.reduced_costs;
    const Real violation = d.complementary_slackness_violation(rr.problem, r.primal, 1e-5);
    CHECK_MSG(violation < 1e-4, "complementary slackness violated by " + std::to_string(violation));
}

TST_MAIN()
