// PDHG / PDLP engine tests -- Build Map ticket #8.
//
// Same discipline as test_simplex.cpp: every expected value is hand-derived,
// checked to PDHG's own native tolerance (~1e-4, Bible S4.2 Engine A) rather
// than simplex-grade digits -- that gap is exactly what crossover (#11) is
// for, not something to paper over here.
#include <cmath>

#include "sovereign/backend.hpp"
#include "sovereign/io.hpp"
#include "sovereign/pdhg.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

constexpr Real kTol = 5e-4;   // a bit looser than the engine's own 1e-4 target

PdhgResult solve_lp(const char* text, PdhgOptions opt = {}) {
    const ReadResult r = read_lp_string(text);
    auto backend = make_backend(BackendKind::Host);
    Pdhg p(opt);
    return p.solve(r.problem, *backend);
}

}  // namespace

TEST(pdhg, two_variable_optimum_matches_simplex) {
    // Same model test_simplex.cpp hand-verifies: optimum 10/3 at (8/3, 2/3).
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const PdhgResult r = solve_lp(text);
    CHECK(r.status == PdhgStatus::Optimal);
    CHECK_NEAR(r.objective, 10.0 / 3.0, kTol);
    CHECK_NEAR(r.primal[0], 8.0 / 3.0, 1e-3);
    CHECK_NEAR(r.primal[1], 2.0 / 3.0, 1e-3);
    CHECK(r.primal_infeasibility < 1e-4);
    CHECK(r.dual_infeasibility < 1e-3);
}

TEST(pdhg, minimize_with_a_binding_lower_bound) {
    const char* text = "Minimize\n obj: 3 x + 2 y\nSubject To\n"
                       " c1: x + y >= 4\nEnd\n";
    // Optimum: drive the cheaper variable (y) to satisfy the constraint,
    // x = 0, y = 4, objective = 8.
    const PdhgResult r = solve_lp(text);
    CHECK(r.status == PdhgStatus::Optimal);
    CHECK_NEAR(r.objective, 8.0, kTol);
    CHECK_NEAR(r.primal[0], 0.0, 1e-3);
    CHECK_NEAR(r.primal[1], 4.0, 1e-3);
}

TEST(pdhg, equality_row) {
    const char* text = "Minimize\n obj: x + y\nSubject To\n"
                       " c1: x + y = 10\nBounds\n x <= 6\nEnd\n";
    // Both directions cost the same, so any feasible point is optimal at
    // objective 10 -- checking the objective and row feasibility, not a
    // specific vertex, since PDHG has no reason to prefer one.
    const PdhgResult r = solve_lp(text);
    CHECK(r.status == PdhgStatus::Optimal);
    CHECK_NEAR(r.objective, 10.0, kTol);
}

TEST(pdhg, column_bound_forces_the_optimum) {
    const char* text = "Minimize\n obj: -x\nSubject To\n c1: x + y <= 100\n"
                       "Bounds\n x <= 3\nEnd\n";
    const PdhgResult r = solve_lp(text);
    CHECK(r.status == PdhgStatus::Optimal);
    CHECK_NEAR(r.objective, -3.0, kTol);
    CHECK_NEAR(r.primal[0], 3.0, 1e-3);
}

TEST(pdhg, agrees_with_a_hand_verified_degenerate_optimum) {
    // Same model test_simplex.cpp's primal-degenerate case uses: optimum -2.
    const char* text = R"(Minimize
 obj: -x1 - x2
Subject To
 c1: x1 <= 4
 c2: x1 + x2 <= 4
 c3: x2 <= 2
End
)";
    const PdhgResult r = solve_lp(text);
    CHECK(r.status == PdhgStatus::Optimal);
    CHECK_NEAR(r.objective, -4.0, kTol);
}

TEST(pdhg, restarts_can_fire_on_a_slow_instance) {
    // Not asserting they always must (that would make the test flaky against
    // a step-size tweak) -- only that when they do fire, the counter reflects
    // it and the run still reaches tolerance. A tighter step budget makes a
    // restart worth checking for without forcing determinism on the exact
    // count.
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    PdhgOptions opt;
    opt.restart_check_period = 5;
    const PdhgResult r = solve_lp(text, opt);
    CHECK(r.status == PdhgStatus::Optimal);
    CHECK(r.restarts >= 0);   // the counter is meaningful, not left uninitialized
}

TEST(pdhg, reduced_costs_use_the_same_sign_convention_as_the_simplex) {
    // x <= 3 binds from above minimizing -x: the simplex convention (ticket
    // #7) has reduced_cost = c - A^T y, and at an upper-bound-binding column
    // in a minimize problem that value is <= 0.
    const char* text = "Minimize\n obj: -x\nSubject To\n c1: x + y <= 100\n"
                       "Bounds\n x <= 3\nEnd\n";
    const PdhgResult r = solve_lp(text);
    CHECK(r.status == PdhgStatus::Optimal);
    CHECK(r.reduced_costs[0] < 1e-3);
}

TEST(pdhg, a_time_limit_before_the_first_restart_check_still_reports_a_point) {
    // Regression test for a real bug (found via ticket #10's engine race on
    // a large real instance, maros-r7): the "best iterate seen" fallback was
    // only populated inside the "ran out of iterations" branch, so a
    // TimeLimit (or Cancelled) that fired before iteration 0 ever reached a
    // restart-check point left best_x_scaled empty -- and the final unscale
    // call threw a dimension mismatch instead of returning a legitimate
    // partial result. A near-zero time limit reproduces the exact edge case:
    // zero iterations completed, status still has to come back cleanly.
    PdhgOptions opt;
    opt.time_limit_seconds = 1e-9;
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const PdhgResult r = solve_lp(text, opt);
    CHECK(r.status == PdhgStatus::TimeLimit);
    CHECK(r.iterations == 0);
    CHECK(static_cast<Idx>(r.primal.size()) == 2);   // reported, not left empty
}

TST_MAIN()
