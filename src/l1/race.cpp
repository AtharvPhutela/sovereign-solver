// Concurrent engine race -- Build Map ticket #10. See race.hpp for the
// design rationale (validity, cancellation, why this is engine-agnostic).
#include "sovereign/race.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "sovereign/backend.hpp"
#include "sovereign/cancellation.hpp"
#include "sovereign/ipm.hpp"
#include "sovereign/pdhg.hpp"
#include "sovereign/simplex.hpp"

namespace sov {
namespace {

/// "Valid" means this engine is making a claim the race harness should
/// trust as a terminal answer -- not merely that solve() returned. An
/// engine that ran out of iterations, timed out, hit a numerical failure,
/// or was cancelled has said nothing usable, and must never win the race
/// just for finishing quickly.
bool is_valid(SolveStatus s) {
    return s == SolveStatus::Optimal || s == SolveStatus::Infeasible || s == SolveStatus::Unbounded;
}
bool is_valid(PdhgStatus s) { return s == PdhgStatus::Optimal; }
bool is_valid(IpmStatus s) { return s == IpmStatus::Optimal; }

RaceResult from(const SimplexResult& r) {
    RaceResult out;
    out.status = to_string(r.status);
    out.objective = r.objective;
    out.primal = r.primal;
    out.dual = r.dual;
    return out;
}
RaceResult from(const PdhgResult& r) {
    RaceResult out;
    out.status = to_string(r.status);
    out.objective = r.objective;
    out.primal = r.primal;
    out.dual = r.dual;
    return out;
}
RaceResult from(const IpmResult& r) {
    RaceResult out;
    out.status = to_string(r.status);
    out.objective = r.objective;
    out.primal = r.primal;
    out.dual = r.dual;
    return out;
}

/// Shared state every worker thread reports into. One CancellationToken per
/// entrant (not one shared token) because the harness cancels the LOSERS
/// specifically, not everyone -- the winner's own token is never touched.
struct SharedState {
    std::mutex mtx;
    std::condition_variable cv;
    std::optional<RaceResult> winner;
    int remaining = 0;
    int launched = 0;
};

}  // namespace

RaceResult race_solve(const Problem& problem, const RaceOptions& options) {
    const auto t0 = std::chrono::steady_clock::now();

    struct Entrant { RaceEngine engine; bool enabled; };
    const Entrant entrants[] = {
        {RaceEngine::Simplex, options.use_simplex},
        {RaceEngine::Pdhg,    options.use_pdhg},
        {RaceEngine::Ipm,     options.use_ipm},
    };

    SharedState shared;
    CancellationToken tokens[3];   // indexed the same as `entrants`
    std::vector<std::thread> workers;

    // Both counts are fixed BEFORE any thread exists, precisely so nothing
    // ever touches them without holding shared.mtx: incrementing them
    // per-launch in the loop below (an earlier version did) races against
    // an already-running worker's `--shared.remaining` under lock the
    // moment one entrant finishes before the loop reaches the next one --
    // caught by ThreadSanitizer, not by inspection.
    for (const Entrant& e : entrants)
        if (e.enabled) { ++shared.remaining; ++shared.launched; }

    auto report = [&](std::size_t idx, RaceResult result, bool valid) {
        std::lock_guard<std::mutex> lock(shared.mtx);
        --shared.remaining;
        if (valid && !shared.winner.has_value()) {
            result.has_winner = true;
            result.winner = entrants[idx].engine;
            result.seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            shared.winner = std::move(result);
            // Cancel every OTHER entrant. The winner's own token (already
            // past its solve, or about to be past it) is left alone.
            for (std::size_t j = 0; j < 3; ++j)
                if (j != idx) tokens[j].cancel();
        }
        shared.cv.notify_all();
    };

    for (std::size_t idx = 0; idx < 3; ++idx) {
        if (!entrants[idx].enabled) continue;

        workers.emplace_back([&, idx]() {
            try {
                switch (entrants[idx].engine) {
                    case RaceEngine::Simplex: {
                        SimplexOptions opt;
                        opt.time_limit_seconds = options.time_limit_seconds;
                        SimplexResult r = Simplex(opt).solve(problem, &tokens[idx]);
                        report(idx, from(r), is_valid(r.status));
                        break;
                    }
                    case RaceEngine::Pdhg: {
                        PdhgOptions opt;
                        opt.time_limit_seconds = options.time_limit_seconds;
                        auto backend = make_backend(default_backend_kind());
                        PdhgResult r = Pdhg(opt).solve(problem, *backend, &tokens[idx]);
                        report(idx, from(r), is_valid(r.status));
                        break;
                    }
                    case RaceEngine::Ipm: {
                        IpmOptions opt;
                        opt.time_limit_seconds = options.time_limit_seconds;
                        IpmResult r = Ipm(opt).solve(problem, &tokens[idx]);
                        report(idx, from(r), is_valid(r.status));
                        break;
                    }
                }
            } catch (const std::exception& e) {
                // Simplex::solve throws on a structurally invalid problem;
                // PDHG/IPM report failure as a status instead. A thread that
                // lets an exception escape its top-level function is
                // std::terminate, not a lost race -- caught here so an
                // invalid model loses the race cleanly instead of crashing
                // every other entrant with it.
                RaceResult failed;
                failed.status = "error";
                failed.message = e.what();
                report(idx, std::move(failed), false);
            }
        });
    }

    {
        std::unique_lock<std::mutex> lock(shared.mtx);
        shared.cv.wait(lock, [&] { return shared.winner.has_value() || shared.remaining == 0; });
    }

    // Every thread is joined before returning, win or lose -- the losers'
    // tokens are already set above, so this is a short wait for whichever
    // iteration each was mid-way through to notice and unwind, not a wait
    // for them to actually finish solving.
    for (std::thread& t : workers) t.join();

    if (shared.winner.has_value()) {
        RaceResult result = std::move(*shared.winner);
        result.engines_launched = shared.launched;
        result.engines_completed = shared.launched;   // all joined by now
        return result;
    }

    RaceResult result;
    result.has_winner = false;
    result.engines_launched = shared.launched;
    result.engines_completed = shared.launched;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    result.message = shared.launched == 0
        ? "no engine was enabled -- nothing raced"
        : "no entrant reached a valid terminal status";
    return result;
}

}  // namespace sov
