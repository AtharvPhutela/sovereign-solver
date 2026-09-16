// Lightweight dual-preserving presolve tests -- Build Map ticket #13, gate M3.
//
// The Build Map's own Test step has two halves and both are checked here:
// "measure reduction in rows/cols/nonzeros" (each move type gets a
// hand-verified case confirming it actually fires) and "confirm duals
// recovered after postsolve still satisfy complementary slackness" (checked
// via ticket #7's own DualSolution::complementary_slackness_violation
// against the ORIGINAL problem -- not a bespoke check reinvented here).
#include <cmath>

#include "sovereign/duals.hpp"
#include "sovereign/io.hpp"
#include "sovereign/presolve.hpp"
#include "sovereign/simplex.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

constexpr Real kTol = 1e-7;

/// Presolves `problem`, solves the reduced model, postsolves, and confirms
/// the postsolved point is a genuine primal-dual optimal pair for the
/// ORIGINAL problem: objective matches an independent direct solve exactly,
/// the primal point is feasible, and complementary slackness holds. This is
/// the one check every move-specific test below runs, so a move that is
/// individually "tested" but produces a subtly wrong dual cannot pass.
struct Checked {
    PresolveResult presolve_result;
    SimplexResult reduced_result;
    PostsolveResult post;
    Real direct_objective;
};

Checked presolve_solve_postsolve_and_check(const Problem& problem) {
    const PresolveResult pr = presolve(problem);
    CHECK(pr.status == PresolveStatus::Reduced);

    Simplex simplex;
    const SimplexResult reduced_result = simplex.solve(pr.reduced);
    CHECK(reduced_result.status == SolveStatus::Optimal);

    const PostsolveResult post = postsolve(pr, reduced_result.primal, reduced_result.dual,
                                           reduced_result.reduced_costs);

    Simplex direct_simplex;
    const SimplexResult direct = direct_simplex.solve(problem);
    CHECK(direct.status == SolveStatus::Optimal);

    CHECK_NEAR(problem.evaluate_objective(post.primal), direct.objective, kTol);

    DualSolution d;
    d.row_duals = post.dual;
    d.reduced_costs = post.reduced_costs;
    const Real violation = d.complementary_slackness_violation(problem, post.primal, 1e-6);
    CHECK_MSG(violation < 1e-5, "complementary slackness violation " + std::to_string(violation));

    return Checked{pr, reduced_result, post, direct.objective};
}

}  // namespace

// --------------------------------------------------------------------------
// each move type, hand-verified
// --------------------------------------------------------------------------

TEST(presolve, redundant_row_is_removed_with_zero_dual) {
    // c1 can never bind (x, y both bounded well inside it); c2 is the real
    // constraint. Presolve must drop c1 entirely.
    Problem::Builder b;
    b.set_sense(ObjSense::Maximize);
    const Idx x = b.add_column("x"); b.set_column_bounds(x, 0, 5); b.set_objective_coefficient(x, 1);
    const Idx y = b.add_column("y"); b.set_column_bounds(y, 0, 5); b.set_objective_coefficient(y, 1);
    const Idx c1 = b.add_row("c1", -Real{1e300}, 100);   // x + y <= 100: never binding
    const Idx c2 = b.add_row("c2", -Real{1e300}, 8);     // x + y <= 8: the real limit
    b.add_coefficient(c1, x, 1); b.add_coefficient(c1, y, 1);
    b.add_coefficient(c2, x, 1); b.add_coefficient(c2, y, 1);
    const Problem problem = b.finish();

    const Checked c = presolve_solve_postsolve_and_check(problem);
    CHECK(c.presolve_result.rows_removed >= 1);
    CHECK_NEAR(c.direct_objective, 8.0, kTol);
}

TEST(presolve, empty_row_with_zero_in_bounds_is_redundant) {
    // c2 deliberately has TWO active columns (not a singleton) so only c1's
    // emptiness is under test here -- a singleton c2 would also tighten and
    // remove itself, which is correct but would muddy what this case checks.
    Problem::Builder b;
    const Idx x = b.add_column("x"); b.set_column_bounds(x, 0, 10); b.set_objective_coefficient(x, 1);
    const Idx y = b.add_column("y"); b.set_column_bounds(y, 0, 10); b.set_objective_coefficient(y, 1);
    const Idx c1 = b.add_row("c1", -1.0, 1.0);   // no coefficients at all: 0 in [-1, 1], redundant
    (void)c1;
    const Idx c2 = b.add_row("c2", 3.0, Real{1e300});
    b.add_coefficient(c2, x, 1);
    b.add_coefficient(c2, y, 1);
    const Problem problem = b.finish();

    const Checked c = presolve_solve_postsolve_and_check(problem);
    CHECK(c.presolve_result.rows_removed == 1);
    CHECK_NEAR(c.direct_objective, 3.0, kTol);
}

TEST(presolve, empty_row_with_zero_outside_bounds_is_infeasible) {
    Problem::Builder b;
    const Idx x = b.add_column("x"); (void)x;
    b.add_row("c1", 1.0, 2.0);   // no coefficients: activity is always 0, and 0 not in [1,2]
    const Problem problem = b.finish();

    const PresolveResult pr = presolve(problem);
    CHECK(pr.status == PresolveStatus::Infeasible);
}

TEST(presolve, singleton_row_tightens_the_bound_and_is_removed) {
    // 2x <= 10, with x's own bound much looser -- presolve must tighten x to
    // <= 5 and drop the row. The dual of the removed row must come out
    // NONZERO (x ends up exactly at the row-derived bound, strictly tighter
    // than its original one) -- this is the case that distinguishes a
    // singleton row from a redundant one.
    Problem::Builder b;
    b.set_sense(ObjSense::Maximize);
    const Idx x = b.add_column("x"); b.set_column_bounds(x, 0, 1000); b.set_objective_coefficient(x, 1);
    const Idx c1 = b.add_row("c1", -Real{1e300}, 10.0);
    b.add_coefficient(c1, x, 2.0);
    const Problem problem = b.finish();

    const Checked c = presolve_solve_postsolve_and_check(problem);
    CHECK(c.presolve_result.rows_removed == 1);
    CHECK_NEAR(c.direct_objective, 5.0, kTol);
    // The row's reconstructed dual absorbed a real reduced cost: with c_x=1
    // and coefficient 2, y_c1 = 1/2 exactly (x is unbounded from above
    // otherwise, so ALL of the reduced cost must come from this row).
    CHECK_NEAR(c.post.dual[static_cast<std::size_t>(c1)], 0.5, kTol);
}

TEST(presolve, two_singleton_rows_on_the_same_column_detect_infeasibility) {
    Problem::Builder b;
    const Idx x = b.add_column("x"); b.set_column_bounds(x, 0, 10);
    const Idx c1 = b.add_row("c1", -Real{1e300}, 1.0);   // x <= 1
    const Idx c2 = b.add_row("c2", 2.0, Real{1e300});    // x >= 2
    b.add_coefficient(c1, x, 1.0);
    b.add_coefficient(c2, x, 1.0);
    const Problem problem = b.finish();

    const PresolveResult pr = presolve(problem);
    CHECK(pr.status == PresolveStatus::Infeasible);
}

TEST(presolve, fixed_column_is_substituted_out) {
    // z is fixed at 3 (an FX-style bound) and appears in the row alongside a
    // free variable x -- presolve must fold z's contribution into the row's
    // bound and drop z as a column entirely.
    Problem::Builder b;
    b.set_sense(ObjSense::Maximize);
    const Idx x = b.add_column("x"); b.set_column_bounds(x, 0, 100); b.set_objective_coefficient(x, -1);
    const Idx z = b.add_column("z"); b.set_column_bounds(z, 3.0, 3.0); b.set_objective_coefficient(z, 2);
    const Idx c1 = b.add_row("c1", -Real{1e300}, 20.0);   // x + z <= 20  =>  x <= 17 once z is fixed
    b.add_coefficient(c1, x, 1.0);
    b.add_coefficient(c1, z, 1.0);
    const Problem problem = b.finish();

    const Checked c = presolve_solve_postsolve_and_check(problem);
    CHECK(c.presolve_result.cols_removed >= 1);
    // maximize -x + 2z with z=3 fixed: minimize x, so x -> 0 (its own lower
    // bound, unrelated to c1 at all) -> objective = 0 + 6 = 6.
    CHECK_NEAR(c.direct_objective, 6.0, kTol);
    CHECK_NEAR(c.post.primal[static_cast<std::size_t>(z)], 3.0, kTol);
}

TEST(presolve, empty_column_is_resolved_from_its_own_bounds) {
    // w touches no row at all; minimizing with a positive cost drives it to
    // its own lower bound regardless of anything else in the model. c1 has
    // TWO active columns (not a singleton) so x survives as an ordinary
    // reduced-problem column and only w's emptiness is under test here.
    Problem::Builder b;
    const Idx x = b.add_column("x"); b.set_column_bounds(x, 0, 10); b.set_objective_coefficient(x, 1);
    const Idx y = b.add_column("y"); b.set_column_bounds(y, 0, 10); b.set_objective_coefficient(y, 0);
    const Idx w = b.add_column("w"); b.set_column_bounds(w, 2.0, 9.0); b.set_objective_coefficient(w, 5);
    const Idx c1 = b.add_row("c1", 1.0, Real{1e300});
    b.add_coefficient(c1, x, 1.0);
    b.add_coefficient(c1, y, 1.0);
    const Problem problem = b.finish();

    const Checked c = presolve_solve_postsolve_and_check(problem);
    CHECK(c.presolve_result.cols_removed == 1);
    CHECK_NEAR(c.post.primal[static_cast<std::size_t>(w)], 2.0, kTol);
    // y (zero cost) absorbs c1's requirement entirely (y=1, x=0), so the
    // objective is just w's own contribution: 5 * 2 = 10.
    CHECK_NEAR(c.direct_objective, 5.0 * 2.0, kTol);
}

TEST(presolve, chained_reductions_across_rounds_still_check_out) {
    // A singleton row tightens y to exactly 4 (a fixed point); the NEXT
    // round must notice y is now fixed and substitute it out of c2, which
    // in turn may make c2 singleton/redundant too -- exercises the trail
    // composing correctly across more than one round, not just one move.
    Problem::Builder b;
    b.set_sense(ObjSense::Minimize);
    const Idx x = b.add_column("x"); b.set_column_bounds(x, 0, 100); b.set_objective_coefficient(x, 1);
    const Idx y = b.add_column("y"); b.set_column_bounds(y, 0, 100); b.set_objective_coefficient(y, 0);
    const Idx c1 = b.add_row("c1", 4.0, 4.0);           // y == 4 exactly (singleton, equality)
    b.add_coefficient(c1, y, 1.0);
    const Idx c2 = b.add_row("c2", 4.0, Real{1e300});   // x + y >= 4  =>  x >= 0 once y=4
    b.add_coefficient(c2, x, 1.0);
    b.add_coefficient(c2, y, 1.0);
    const Problem problem = b.finish();

    const Checked c = presolve_solve_postsolve_and_check(problem);
    CHECK(c.presolve_result.rows_removed >= 1);
    CHECK(c.presolve_result.cols_removed >= 1);
    CHECK_NEAR(c.direct_objective, 0.0, kTol);   // x -> 0, y fixed at 4, cost only on x
    CHECK_NEAR(c.post.primal[static_cast<std::size_t>(y)], 4.0, kTol);
}

// --------------------------------------------------------------------------
// against hand-verified LPs already used elsewhere in the suite
// --------------------------------------------------------------------------

TEST(presolve, agrees_with_direct_simplex_on_a_hand_verified_optimum) {
    // Same model as test_simplex.cpp's two_variable_optimum: objective -10/3
    // at x=8/3, y=2/3. Presolve is not expected to reduce anything here (no
    // redundant/singleton/fixed structure) -- the point is that running the
    // pipeline on an ordinary model is a correct no-op, not just that it
    // fires on contrived cases.
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const ReadResult r = read_lp_string(text);
    const Checked c = presolve_solve_postsolve_and_check(r.problem);
    CHECK_NEAR(c.direct_objective, 10.0 / 3.0, kTol);
}

TEST(presolve, agrees_with_direct_simplex_on_a_degenerate_optimum) {
    const char* text = R"(Minimize
 obj: -x1 - x2
Subject To
 c1: x1 <= 4
 c2: x1 + x2 <= 4
 c3: x2 <= 2
End
)";
    const ReadResult r = read_lp_string(text);
    const Checked c = presolve_solve_postsolve_and_check(r.problem);
    CHECK_NEAR(c.direct_objective, -4.0, kTol);
}

TST_MAIN()
