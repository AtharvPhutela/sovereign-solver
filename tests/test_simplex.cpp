// From-scratch simplex tests -- Build Map ticket #4.
//
// Every expected value here is derived by hand from the model, not taken from
// another solver. That is the point of an oracle: it has to be checkable
// against arithmetic, or it cannot be the thing other solvers are checked
// against.
#include <cmath>
#include <string>

#include "sovereign/io.hpp"
#include "sovereign/simplex.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

constexpr Real kTol = 1e-9;

SimplexResult solve_lp(const char* text, SimplexOptions opt = {}) {
    const ReadResult r = read_lp_string(text);
    Simplex s(opt);
    return s.solve(r.problem);
}

SimplexResult solve_mps(const char* text, SimplexOptions opt = {}) {
    const ReadResult r = read_mps_string(text);
    Simplex s(opt);
    return s.solve(r.problem);
}

}  // namespace

// --------------------------------------------------------------------------
// hand-checkable optima
// --------------------------------------------------------------------------

TEST(simplex, two_variable_optimum) {
    // max x + y  s.t.  x + 2y <= 4,  4x + 2y <= 12,  x,y >= 0
    // Vertices: (0,0)=0, (3,0)=3, (0,2)=2, and the intersection
    // x + 2y = 4, 4x + 2y = 12  =>  3x = 8, x = 8/3, y = 2/3, value 10/3.
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.objective, 10.0 / 3.0, kTol);
    CHECK_NEAR(r.primal[0], 8.0 / 3.0, 1e-7);
    CHECK_NEAR(r.primal[1], 2.0 / 3.0, 1e-7);
}

TEST(simplex, minimize_and_maximize_are_mirror_images) {
    const char* min_text = "Minimize\n obj: -x - y\nSubject To\n"
                           " c1: x + 2 y <= 4\n c2: 4 x + 2 y <= 12\nEnd\n";
    const char* max_text = "Maximize\n obj: x + y\nSubject To\n"
                           " c1: x + 2 y <= 4\n c2: 4 x + 2 y <= 12\nEnd\n";
    const SimplexResult a = solve_lp(min_text);
    const SimplexResult b = solve_lp(max_text);
    CHECK(a.status == SolveStatus::Optimal);
    CHECK(b.status == SolveStatus::Optimal);
    CHECK_NEAR(a.objective, -b.objective, kTol);
}

TEST(simplex, objective_constant_is_carried_through) {
    const char* text = "Minimize\n obj: x + 100\nSubject To\n c1: x >= 7\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.objective, 107.0, kTol);
}

TEST(simplex, equality_constraints) {
    // min x + y  s.t.  x + y = 5,  x - y = 1  =>  x = 3, y = 2, obj 5
    const char* text = "Minimize\n obj: x + y\nSubject To\n"
                       " c1: x + y = 5\n c2: x - y = 1\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.objective, 5.0, kTol);
    CHECK_NEAR(r.primal[0], 3.0, 1e-7);
    CHECK_NEAR(r.primal[1], 2.0, 1e-7);
}

TEST(simplex, ranged_row) {
    // min x  s.t.  3 <= x + y <= 8,  y <= 1  =>  x >= 2, so obj = 2.
    const char* text = "Minimize\n obj: x\nSubject To\n"
                       " c1: 3 <= x + y <= 8\n c2: y <= 1\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.objective, 2.0, 1e-7);
}

TEST(simplex, negative_lower_bounds_and_free_variables) {
    // min x + y  with  x >= -5, y free, x + y >= -8, y >= -3
    // => the optimum sits at x = -5, y = -3, objective -8.
    const char* text = R"(Minimize
 obj: x + y
Subject To
 c1: x + y >= -8
 c2: y >= -3
Bounds
 x >= -5
 y free
End
)";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.objective, -8.0, 1e-7);
}

TEST(simplex, upper_bounded_variable_needs_no_extra_row) {
    // The bounded-variable formulation handles x <= 3 as a bound, not a row.
    const char* text = "Maximize\n obj: x\nSubject To\n c1: x + y <= 100\n"
                       "Bounds\n x <= 3\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.objective, 3.0, kTol);
}

// --------------------------------------------------------------------------
// the three terminal statuses
// --------------------------------------------------------------------------

TEST(simplex, detects_infeasibility) {
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x >= 5\n c2: x <= 1\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Infeasible);
}

TEST(simplex, detects_unboundedness) {
    const char* text = "Minimize\n obj: -x\nSubject To\n c1: x - y <= 5\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Unbounded);
}

TEST(simplex, crossed_bounds_are_infeasible_without_iterating) {
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x + y >= 0\n"
                       "Bounds\n 5 <= x <= 3\nEnd\n";
    // The reader rejects a crossed bound pair outright, which is itself the
    // correct answer -- an unsolvable model should not reach the solver.
    CHECK_THROWS(read_lp_string(text));
}

TEST(simplex, empty_constraint_set) {
    // No rows at all: the optimum is at the bounds.
    const char* text = "Minimize\n obj: x\nSubject To\n c1: 0 x >= -1\n"
                       "Bounds\n x >= 2\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.objective, 2.0, kTol);
}

// --------------------------------------------------------------------------
// degeneracy and anti-cycling -- the ticket's explicit requirement
// --------------------------------------------------------------------------

TEST(simplex, degenerate_optimum_terminates) {
    // min -x - y  s.t.  x + y <= 2, x <= 1, y <= 1.
    // The optimum (1,1) has three active constraints for two variables, so any
    // optimal basis carries a zero-valued basic variable. A simplex without an
    // anti-cycling rule can stall here forever rather than merely slowly.
    const char* text = R"(Minimize
 obj: -x - y
Subject To
 c1: x + y <= 2
 c2: x <= 1
 c3: y <= 1
End
)";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.objective, -2.0, kTol);
}

TEST(simplex, beale_cycling_example_terminates) {
    // Beale's classic 1955 instance: with the naive most-negative-reduced-cost
    // rule and first-index tie-breaking it cycles through six bases forever.
    // The optimum is -1/20 at x1 = 3/4, x2 = 0, x3 = 0, x4 = 1/4... this is
    // THE canonical test that the anti-cycling guard actually works.
    const char* text = R"(Minimize
 obj: -0.75 x4 + 150 x5 - 0.02 x6 + 6 x7
Subject To
 c1: 0.25 x4 - 60 x5 - 0.04 x6 + 9 x7 + x1 = 0
 c2: 0.5 x4 - 90 x5 - 0.02 x6 + 3 x7 + x2 = 0
 c3: x6 + x3 = 1
End
)";
    SimplexOptions opt;
    opt.max_iterations = 5000;      // a cycle would blow straight past this
    const SimplexResult r = solve_lp(text, opt);
    CHECK_MSG(r.status == SolveStatus::Optimal,
              std::string("Beale's example did not terminate cleanly: ")
              + to_string(r.status));
    CHECK_NEAR(r.objective, -0.05, 1e-7);
}

TEST(simplex, bland_rule_alone_solves_the_same_problems) {
    // Bland's rule is the provably terminating configuration. It must reach the
    // same optimum as the fast path -- if the two disagree, the fast path's
    // pricing or ratio test is wrong.
    const char* text = R"(Minimize
 obj: -x - y
Subject To
 c1: x + y <= 2
 c2: x <= 1
 c3: y <= 1
End
)";
    SimplexOptions fast;
    SimplexOptions bland;
    bland.always_bland = true;
    const SimplexResult a = solve_lp(text, fast);
    const SimplexResult b = solve_lp(text, bland);
    CHECK(a.status == SolveStatus::Optimal);
    CHECK(b.status == SolveStatus::Optimal);
    CHECK_NEAR(a.objective, b.objective, 1e-9);
}

// --------------------------------------------------------------------------
// numerical guards
// --------------------------------------------------------------------------

TEST(simplex, refactorization_frequency_does_not_change_the_answer) {
    // The eta file between refactorizations is an approximation that
    // accumulates error. If the answer moves when the refactorization interval
    // changes, the update path is wrong -- which is exactly how the tiny-pivot
    // bug on `blend` first showed itself.
    const char* text = R"(Minimize
 obj: 3 a + 2 b + 4 c + d
Subject To
 c1: a + b + c + d >= 10
 c2: 2 a - b + 3 c <= 20
 c3: a + 4 b - c + 2 d = 15
 c4: a - c + d >= -5
End
)";
    Real reference = 0.0;
    for (int frequency : {1, 3, 7, 50}) {
        SimplexOptions opt;
        opt.refactor_frequency = frequency;
        const SimplexResult r = solve_lp(text, opt);
        CHECK_MSG(r.status == SolveStatus::Optimal,
                  "failed at refactor frequency " + std::to_string(frequency));
        if (frequency == 1) reference = r.objective;
        else CHECK_MSG(std::abs(r.objective - reference) < 1e-7,
                       "answer moved with refactor frequency " + std::to_string(frequency));
    }
}

TEST(simplex, wide_coefficient_range_still_solves) {
    // A spread of 1e8 between the largest and smallest coefficient is the sort
    // of conditioning the problem statement names explicitly.
    const char* text = R"(Minimize
 obj: x + y
Subject To
 c1: 1e-4 x + 1e4 y >= 1
 c2: x + y >= 0.001
End
)";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK(std::isfinite(r.objective));
    CHECK(r.primal_infeasibility < 1e-6);
}

TEST(simplex, iteration_limit_is_honoured) {
    const char* text = R"(Minimize
 obj: -x - y - z
Subject To
 c1: x + y <= 5
 c2: y + z <= 5
 c3: x + z <= 5
End
)";
    SimplexOptions opt;
    opt.max_iterations = 1;
    const SimplexResult r = solve_lp(text, opt);
    CHECK(r.status == SolveStatus::IterationLimit);
}

// --------------------------------------------------------------------------
// solution validity, not just the objective number
// --------------------------------------------------------------------------

TEST(simplex, reported_solution_is_actually_feasible) {
    // An objective that matches while the solution violates its constraints is
    // the worst outcome available: it looks right. So check the point, not the
    // number.
    const char* text = R"(Minimize
 obj: 2 a + 3 b + c
Subject To
 c1: a + b + c >= 10
 c2: a - b <= 4
 c3: 2 a + b + 3 c = 25
Bounds
 a <= 8
 b <= 9
End
)";
    const ReadResult rr = read_lp_string(text);
    const Problem& p = rr.problem;
    Simplex s;
    const SimplexResult r = s.solve(p);
    CHECK(r.status == SolveStatus::Optimal);

    // Column bounds.
    for (Idx j = 0; j < p.num_cols(); ++j) {
        CHECK(r.primal[static_cast<std::size_t>(j)] >= p.col_lower()[static_cast<std::size_t>(j)] - 1e-7);
        CHECK(r.primal[static_cast<std::size_t>(j)] <= p.col_upper()[static_cast<std::size_t>(j)] + 1e-7);
    }
    // Row activities.
    std::vector<Real> activity(static_cast<std::size_t>(p.num_rows()), 0.0);
    p.matrix().multiply(r.primal, activity);
    for (Idx i = 0; i < p.num_rows(); ++i) {
        const auto k = static_cast<std::size_t>(i);
        CHECK_MSG(activity[k] >= p.row_lower()[k] - 1e-6, "row lower bound violated");
        CHECK_MSG(activity[k] <= p.row_upper()[k] + 1e-6, "row upper bound violated");
    }
    // The reported objective must be the objective of the reported point.
    CHECK_NEAR(r.objective, p.evaluate_objective(r.primal), 1e-9);
}

TEST(simplex, mps_and_lp_forms_of_one_model_give_the_same_answer) {
    const char* lp = "Minimize\n cost: x + 2 y\nSubject To\n"
                     " r1: x + y >= 4\n r2: x - y <= 2\nEnd\n";
    const char* mps = R"(NAME T
ROWS
 N  cost
 G  r1
 L  r2
COLUMNS
    x         cost         1.0   r1           1.0
    x         r2           1.0
    y         cost         2.0   r1           1.0
    y         r2          -1.0
RHS
    RHS       r1           4.0   r2           2.0
ENDATA
)";
    const SimplexResult a = solve_lp(lp);
    const SimplexResult b = solve_mps(mps);
    CHECK(a.status == SolveStatus::Optimal);
    CHECK(b.status == SolveStatus::Optimal);
    CHECK_NEAR(a.objective, b.objective, 1e-9);
}

TEST(simplex, integrality_is_ignored_this_is_the_relaxation) {
    // Branch and bound is ticket #38. Here an integer column is just a bounded
    // one, and the answer must be the continuous optimum -- claiming otherwise
    // would be claiming a MIP solver we do not have.
    const char* text = R"(Maximize
 obj: x
Subject To
 c1: 2 x <= 3
General
 x
End
)";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.objective, 1.5, kTol);     // not 1, which the integer optimum would be
}

// --------------------------------------------------------------------------
// warm start (ticket #11: crossover's basis-guess entry point)
// --------------------------------------------------------------------------

TEST(simplex, a_correct_warm_start_basis_reaches_the_same_exact_optimum) {
    // Same model as two_variable_optimum. The optimal basis there has x, y
    // both basic (the binding constraint intersection); columns are indices
    // 0, 1 in the internal [A -I] numbering, rows' logicals are 2, 3.
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const ReadResult r = read_lp_string(text);
    WarmStart ws;
    ws.basis = {0, 1};                       // guess: x and y both basic
    ws.point = {8.0 / 3.0, 2.0 / 3.0, 4.0, 12.0};   // x, y, row activities (both binding)

    Simplex s;
    const SimplexResult result = s.solve(r.problem, nullptr, &ws);
    CHECK(result.status == SolveStatus::Optimal);
    CHECK_NEAR(result.objective, 10.0 / 3.0, kTol);
    CHECK_NEAR(result.primal[0], 8.0 / 3.0, 1e-7);
    CHECK_NEAR(result.primal[1], 2.0 / 3.0, 1e-7);
}

TEST(simplex, a_structurally_bad_warm_start_falls_back_and_still_solves) {
    // A basis guess with an out-of-range index is exactly the kind of bad
    // guess a real crossover checkpoint could never construct deliberately,
    // but this is the contract test: apply_warm_start() must reject it
    // (returning false) rather than reading out of bounds, and the solve
    // must fall back to the guaranteed-nonsingular slack basis and still
    // reach the correct answer.
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x >= 2\nEnd\n";
    const ReadResult r = read_lp_string(text);
    WarmStart bad;
    bad.basis = {5};                          // out of range: total_ = n + m = 2
    bad.point = {2.0, 2.0};

    Simplex s;
    const SimplexResult result = s.solve(r.problem, nullptr, &bad);
    CHECK(result.status == SolveStatus::Optimal);
    CHECK_NEAR(result.objective, 2.0, kTol);
}

TEST(simplex, warm_start_on_an_infeasible_model_still_proves_infeasibility) {
    // The guess must never change what the solve is ALLOWED to conclude --
    // only how fast it gets there. Phase I from a warm-started basis still
    // has to certify infeasibility exactly as it would from a cold start.
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x >= 2\n c2: x <= 1\nEnd\n";
    const ReadResult r = read_lp_string(text);
    WarmStart ws;
    ws.basis = {0, 1};                          // guess x and c1's logical both basic
    ws.point = {1.5, 1.5, 1.5};                 // a plausible-looking but infeasible guess



    Simplex s;
    const SimplexResult result = s.solve(r.problem, nullptr, &ws);
    CHECK(result.status == SolveStatus::Infeasible);
}

TST_MAIN()
