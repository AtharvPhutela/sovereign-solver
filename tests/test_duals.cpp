// Farkas / dual-ray & duals plumbing tests -- Build Map ticket #7.
//
// Two halves, matching the ticket's pass condition exactly:
//   1. On an infeasible LP, the simplex's Farkas certificate actually proves
//      infeasibility -- checked by the certificate's own arithmetic, not by
//      trusting the SolveStatus enum.
//   2. On a feasible/optimal LP, the reported duals satisfy complementary
//      slackness against the reported primal point.
#include "sovereign/duals.hpp"
#include "sovereign/io.hpp"
#include "sovereign/simplex.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

constexpr Real kTol = 1e-7;

SimplexResult solve_lp(const char* text) {
    const ReadResult r = read_lp_string(text);
    Simplex s;
    return s.solve(r.problem);
}

}  // namespace

// --------------------------------------------------------------------------
// the certificate on infeasible models
// --------------------------------------------------------------------------

TEST(duals, farkas_certifies_the_smoke_infeasible_instance) {
    // The tiny_infeasible.mps smoke fixture: min x s.t. x >= 2, x <= 1.
    // Worked by hand in duals.hpp's own comment: y = (1, -1) makes the
    // row-side range [1, inf) and the column-side range {0} -- disjoint.
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x >= 2\n c2: x <= 1\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Infeasible);
    CHECK(!r.farkas.empty());
    CHECK_MSG(r.farkas.certifies_infeasibility(read_lp_string(text).problem),
              "the reported ray does not actually certify infeasibility");
}

TEST(duals, farkas_certifies_a_two_row_range_contradiction) {
    // x + y >= 10 and x + y <= 3, both structural, no crossed column bound
    // anywhere -- infeasibility only shows up once the two rows interact.
    const char* text = "Minimize\n obj: x + y\nSubject To\n"
                       " c1: x + y >= 10\n c2: x + y <= 3\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Infeasible);
    CHECK(!r.farkas.empty());
    CHECK(r.farkas.certifies_infeasibility(read_lp_string(text).problem));
}

TEST(duals, farkas_survives_a_free_column_that_the_contradiction_does_not_touch) {
    // z is free and appears nowhere in the contradiction; a sound certificate
    // must have a zero coefficient on it (d_z == 0), or the column-side range
    // would be all of (-inf, inf) and never disjoint from anything.
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x >= 2\n c2: x <= 1\n"
                       "Bounds\n z free\nEnd\n";
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Infeasible);
    CHECK(r.farkas.certifies_infeasibility(read_lp_string(text).problem));
}

TEST(duals, an_empty_certificate_never_certifies_anything) {
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x >= 0\nEnd\n";
    const Problem p = read_lp_string(text).problem;
    FarkasCertificate empty;
    CHECK(!empty.certifies_infeasibility(p));
}

TEST(duals, a_feasible_problem_has_no_valid_certificate) {
    // Sanity check in the other direction: throwing a plausible-looking ray at
    // a FEASIBLE model must not certify anything, or the checker is unsound.
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x >= 1\n c2: x <= 5\nEnd\n";
    const Problem p = read_lp_string(text).problem;
    FarkasCertificate ray;
    ray.row_multipliers = {1.0, -1.0};
    CHECK(!ray.certifies_infeasibility(p));
}

// --------------------------------------------------------------------------
// complementary slackness on optimal models
// --------------------------------------------------------------------------

TEST(duals, complementary_slackness_holds_at_the_optimum) {
    // max x + y  s.t.  x + 2y <= 4,  4x + 2y <= 12  -- same model test_simplex
    // already hand-verifies the optimum for (10/3 at (8/3, 2/3), both rows
    // binding). Both constraints bind at the optimum, so nonzero duals there
    // are expected and the check must still report ~0 violation.
    const char* text = "Maximize\n obj: x + y\nSubject To\n"
                       " c1: x + 2 y <= 4\n c2: 4 x + 2 y <= 12\nEnd\n";
    const ReadResult rr = read_lp_string(text);
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);

    DualSolution d;
    d.row_duals = r.dual;
    d.reduced_costs = r.reduced_costs;
    const Real violation = d.complementary_slackness_violation(rr.problem, r.primal);
    CHECK_MSG(violation < kTol, "complementary slackness violated by " + std::to_string(violation));
}

TEST(duals, complementary_slackness_holds_when_a_bound_not_a_row_binds) {
    // The optimum here is forced by the column bound (x <= 3), while c1 is
    // slack -- exercises the reduced-cost half of the check with a genuinely
    // inactive row.
    const char* text = "Minimize\n obj: -x\nSubject To\n c1: x + y <= 100\n"
                       "Bounds\n x <= 3\nEnd\n";
    const ReadResult rr = read_lp_string(text);
    const SimplexResult r = solve_lp(text);
    CHECK(r.status == SolveStatus::Optimal);
    CHECK_NEAR(r.primal[0], 3.0, kTol);

    DualSolution d;
    d.row_duals = r.dual;
    d.reduced_costs = r.reduced_costs;
    const Real violation = d.complementary_slackness_violation(rr.problem, r.primal);
    CHECK_MSG(violation < kTol, "complementary slackness violated by " + std::to_string(violation));
}

TST_MAIN()
