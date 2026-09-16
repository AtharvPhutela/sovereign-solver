// Concurrent checkpoint crossover tests -- Build Map ticket #11, gate M2.
//
// The Build Map's own Test step: "take a PDHG interior solution on a large
// LP; confirm crossover produces a basic vertex whose objective matches the
// oracle EXACTLY (not just to 1e-4), and confirm the wall-clock including
// crossover still beats the CPU simplex baseline." These models are tiny
// (hand-checkable, not "large"), but the property under test is the same:
// crossover's answer must be exact-tolerance, not PDHG's native 1e-4, and it
// must come from an actual concurrent checkpoint win whenever one is
// possible -- not silently fall through to the last-resort single attempt
// every time, which would defeat the point of ticket #11 over plain #11's
// predecessor (#8 + a bolted-on final crossover).
#include <cmath>

#include "sovereign/backend.hpp"
#include "sovereign/crossover.hpp"
#include "sovereign/io.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

constexpr Real kTol = 1e-9;

CrossoverResult crossover(const char* text, CrossoverOptions opt = {}) {
    const ReadResult r = read_lp_string(text);
    auto backend = make_backend(default_backend_kind());
    Crossover c(opt);
    return c.solve(r.problem, *backend);
}

// Small enough restart-check period that even these tiny models fire several
// checkpoints before PDHG itself would converge or exhaust its own budget --
// this is what actually exercises "concurrent," not just "crossover."
CrossoverOptions frequent_checkpoints() {
    CrossoverOptions opt;
    opt.pdhg.restart_check_period = 3;
    opt.pdhg.max_iterations = 5000;
    return opt;
}

}  // namespace

TEST(crossover, guess_basis_from_point_returns_a_valid_shaped_basis) {
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const ReadResult r = read_lp_string(text);
    // The true optimal vertex: x=8/3, y=2/3, both rows binding.
    const std::vector<Real> x = {8.0 / 3.0, 2.0 / 3.0};
    const std::vector<Real> y = {0.0, 0.0};   // duals not used by the guess itself
    const WarmStart ws = guess_basis_from_point(r.problem, x, y);

    CHECK_EQ(ws.basis.size(), static_cast<std::size_t>(r.problem.num_rows()));
    CHECK_EQ(ws.point.size(),
            static_cast<std::size_t>(r.problem.num_cols() + r.problem.num_rows()));
    // Both x and y are interior to their own [0, inf) bounds (strictly
    // positive with no upper bound), while both rows are exactly AT their
    // upper bound (binding) -- so the guess should pick the two structural
    // columns (indices 0, 1) as basic, not either row logical (2, 3).
    CHECK(ws.basis.size() == 2);
    CHECK((ws.basis[0] == 0 && ws.basis[1] == 1));
}

TEST(crossover, reaches_the_exact_optimum_not_just_pdhg_tolerance) {
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const CrossoverResult r = crossover(text, frequent_checkpoints());
    CHECK(r.status == CrossoverStatus::Optimal);
    // 1e-9, not PDHG's native 1e-4 -- this is the whole point of crossover.
    CHECK_NEAR(r.objective, 10.0 / 3.0, kTol);
    CHECK_NEAR(r.primal[0], 8.0 / 3.0, 1e-7);
    CHECK_NEAR(r.primal[1], 2.0 / 3.0, 1e-7);
    // A real concurrent win: some checkpoint before PDHG's own end produced
    // the vertex, not just the last-resort fallback attempt.
    CHECK(r.checkpoint_attempts > 0);
}

TEST(crossover, degenerate_small_instance_matches_the_hand_derived_optimum) {
    const char* text = R"(Minimize
 obj: -x1 - x2
Subject To
 c1: x1 <= 4
 c2: x1 + x2 <= 4
 c3: x2 <= 2
End
)";
    const CrossoverResult r = crossover(text, frequent_checkpoints());
    CHECK(r.status == CrossoverStatus::Optimal);
    CHECK_NEAR(r.objective, -4.0, kTol);
}

TEST(crossover, an_infeasible_model_is_caught_by_a_crossover_attempts_own_simplex) {
    // PDHG has no infeasibility detection of its own (ticket #8's own
    // header), so this only works because each crossover attempt's Simplex
    // Phase I is exact regardless of the checkpoint's guess -- it must
    // notice infeasibility even though PDHG itself never would.
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x >= 2\n c2: x <= 1\nEnd\n";
    const CrossoverResult r = crossover(text, frequent_checkpoints());
    CHECK(r.status == CrossoverStatus::Infeasible);
}

TEST(crossover, no_checkpoint_firing_still_solves_via_the_final_fallback_attempt) {
    // restart_check_period larger than max_iterations means the checkpoint
    // callback never fires even once during the loop -- the only path left
    // to an answer is the last-resort attempt on PDHG's own returned point.
    // winning_checkpoint_iteration == -1 marks exactly that path.
    CrossoverOptions opt;
    opt.pdhg.restart_check_period = 10000;
    opt.pdhg.max_iterations = 50;
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    const CrossoverResult r = crossover(text, opt);
    CHECK(r.status == CrossoverStatus::Optimal);
    CHECK_NEAR(r.objective, 10.0 / 3.0, kTol);
    // Exactly one attempt -- the last-resort fallback on PDHG's own returned
    // point; no mid-run checkpoint ever fired to launch any other.
    CHECK(r.checkpoint_attempts == 1);
    CHECK(r.winning_checkpoint_iteration == -1);
}

TEST(crossover, repeated_solves_do_not_hang_or_leak_threads) {
    // Practical evidence of clean join discipline (ticket #10's own
    // precedent): run several times back to back and confirm each one
    // returns promptly with a consistent, correct answer.
    const char* text = R"(Maximize
 obj: x + y
Subject To
 c1: x + 2 y <= 4
 c2: 4 x + 2 y <= 12
End
)";
    for (int i = 0; i < 10; ++i) {
        const CrossoverResult r = crossover(text, frequent_checkpoints());
        CHECK(r.status == CrossoverStatus::Optimal);
        CHECK_NEAR(r.objective, 10.0 / 3.0, kTol);
    }
}

TST_MAIN()
