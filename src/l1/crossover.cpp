// Concurrent checkpoint crossover -- Build Map ticket #11. See crossover.hpp
// for the full design rationale (why concurrent, why a basis guess is safe).
#include "sovereign/crossover.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>

#include "sovereign/cancellation.hpp"

namespace sov {

const char* to_string(CrossoverStatus s) noexcept {
    switch (s) {
        case CrossoverStatus::Optimal:    return "optimal";
        case CrossoverStatus::Infeasible: return "infeasible";
        case CrossoverStatus::Unbounded:  return "unbounded";
        case CrossoverStatus::Failed:     return "failed";
        case CrossoverStatus::NotSolved:  return "not_solved";
    }
    return "unknown";
}

WarmStart guess_basis_from_point(const Problem& problem, std::span<const Real> x,
                                 std::span<const Real> y) {
    (void)y;   // not needed for the guess itself -- kept in the signature because
               // a caller reaching in from a dual-aware engine (IPM) has it in
               // hand, and a future refinement (pricing ties by reduced cost)
               // would want it without an interface change.
    const Idx n = problem.num_cols();
    const Idx m = problem.num_rows();
    const Idx total = n + m;

    std::vector<Real> activity(static_cast<std::size_t>(m), 0.0);
    problem.matrix().multiply(x, activity);

    WarmStart ws;
    ws.point.resize(static_cast<std::size_t>(total));
    for (Idx j = 0; j < n; ++j) ws.point[static_cast<std::size_t>(j)] = x[static_cast<std::size_t>(j)];
    for (Idx i = 0; i < m; ++i)
        ws.point[static_cast<std::size_t>(n + i)] = activity[static_cast<std::size_t>(i)];

    // Margin to the nearer bound: larger = more interior = a better basic
    // guess. A variable with no finite bound at all is treated as maximally
    // interior -- it MUST be basic for the guessed value to be representable
    // at all (a nonbasic variable is pinned to a bound; a free nonbasic
    // variable is pinned to 0, which would silently discard the guess).
    std::vector<Real> margin(static_cast<std::size_t>(total));
    for (Idx j = 0; j < total; ++j) {
        const auto k = static_cast<std::size_t>(j);
        const Real lo = (j < n) ? problem.col_lower()[k] : problem.row_lower()[k - static_cast<std::size_t>(n)];
        const Real hi = (j < n) ? problem.col_upper()[k] : problem.row_upper()[k - static_cast<std::size_t>(n)];
        const Real v = ws.point[k];
        const bool has_lo = is_finite_bound(lo);
        const bool has_hi = is_finite_bound(hi);
        if (!has_lo && !has_hi) margin[k] = kInfinity;
        else if (has_lo && has_hi) margin[k] = std::min(v - lo, hi - v);
        else if (has_lo) margin[k] = v - lo;
        else margin[k] = hi - v;
    }

    std::vector<Idx> order(static_cast<std::size_t>(total));
    std::iota(order.begin(), order.end(), Idx{0});
    const auto m_count = static_cast<std::ptrdiff_t>(m);
    std::partial_sort(order.begin(), order.begin() + m_count, order.end(),
                      [&](Idx a, Idx b) {
                          return margin[static_cast<std::size_t>(a)] > margin[static_cast<std::size_t>(b)];
                      });

    ws.basis.assign(order.begin(), order.begin() + m_count);
    std::sort(ws.basis.begin(), ws.basis.end());   // deterministic ordering, cosmetic only
    return ws;
}

namespace {

/// Same rule as ticket #10's race harness (race.cpp): a status is "valid"
/// only when it is a claim Simplex is actually entitled to make about the
/// ORIGINAL problem. Phase I/II are exact regardless of how good the warm
/// start's guess was, so Infeasible/Unbounded from a crossover attempt are
/// just as trustworthy as Optimal -- only IterationLimit/TimeLimit/
/// NumericalFailure/Cancelled say nothing usable.
bool is_valid(SolveStatus s) {
    return s == SolveStatus::Optimal || s == SolveStatus::Infeasible || s == SolveStatus::Unbounded;
}

CrossoverStatus from_solve_status(SolveStatus s) {
    switch (s) {
        case SolveStatus::Optimal:    return CrossoverStatus::Optimal;
        case SolveStatus::Infeasible: return CrossoverStatus::Infeasible;
        case SolveStatus::Unbounded:  return CrossoverStatus::Unbounded;
        default:                      return CrossoverStatus::Failed;
    }
}

CrossoverResult from(const SimplexResult& r, long long checkpoint_iteration) {
    CrossoverResult out;
    out.status = from_solve_status(r.status);
    out.objective = r.objective;
    out.primal = r.primal;
    out.dual = r.dual;
    out.reduced_costs = r.reduced_costs;
    out.winning_checkpoint_iteration = checkpoint_iteration;
    return out;
}

/// Shared state across the PDHG thread and every crossover-attempt thread it
/// spawns from a checkpoint. CancellationToken is neither copyable nor
/// movable (it wraps a std::atomic<bool>), and attempts are spawned an
/// unknown number of times at an unknown cadence -- a std::deque gives each
/// token a stable address across growth without needing either property,
/// unlike std::vector.
struct SharedState {
    std::mutex mtx;
    std::condition_variable cv;
    std::optional<CrossoverResult> winner;
    std::deque<CancellationToken> attempt_tokens;
    std::vector<std::thread> attempt_threads;
    long long launched_attempts = 0;
    long long finished_attempts = 0;
    bool pdhg_finished = false;
};

}  // namespace

CrossoverResult Crossover::solve(const Problem& problem, Backend& backend) const {
    const auto t0 = std::chrono::steady_clock::now();

    std::string why;
    if (problem.validate(&why) != Status::Ok) {
        CrossoverResult result;
        result.status = CrossoverStatus::Failed;
        result.message = "invalid problem: " + why;
        return result;
    }

    SharedState shared;
    CancellationToken pdhg_token;

    auto report = [&](SimplexResult r, long long checkpoint_iteration) {
        std::lock_guard<std::mutex> lock(shared.mtx);
        ++shared.finished_attempts;
        if (is_valid(r.status) && !shared.winner.has_value()) {
            shared.winner = from(r, checkpoint_iteration);
            // Cancel PDHG and every OTHER in-flight attempt. The winner's own
            // token is already past its solve; cancelling it too is harmless.
            pdhg_token.cancel();
            for (CancellationToken& tok : shared.attempt_tokens) tok.cancel();
        }
        shared.cv.notify_all();
    };

    PdhgOptions pdhg_opt = options_.pdhg;
    pdhg_opt.checkpoint = [&](long long iteration, const std::vector<Real>& x,
                              const std::vector<Real>& y, Real /*residual*/) {
        // Runs synchronously on the PDHG thread -- must return promptly.
        // Building the warm-start guess is pure host-side arithmetic over
        // `problem` (read-only, safe to touch without the lock) and is cheap
        // next to a checkpoint's own KKT-residual computation; only the
        // shared bookkeeping needs the mutex.
        std::unique_lock<std::mutex> lock(shared.mtx);
        if (shared.winner.has_value()) return;   // already won; stop spawning attempts
        ++shared.launched_attempts;
        shared.attempt_tokens.emplace_back();
        CancellationToken* tok = &shared.attempt_tokens.back();
        lock.unlock();

        WarmStart ws = guess_basis_from_point(problem, x, y);

        lock.lock();
        shared.attempt_threads.emplace_back([&, ws = std::move(ws), tok, iteration]() {
            try {
                Simplex simplex(options_.crossover_simplex);
                SimplexResult r = simplex.solve(problem, tok, &ws);
                report(std::move(r), iteration);
            } catch (const std::exception&) {
                // problem was already validated once above; Simplex::solve
                // only throws on that structural check, so this should be
                // unreachable. Caught anyway so an attempt thread can never
                // escape an exception into std::terminate -- it just loses.
                report(SimplexResult{}, iteration);
            }
        });
        lock.unlock();
    };

    PdhgResult pdhg_result;
    std::thread pdhg_thread([&]() {
        Pdhg pdhg(pdhg_opt);
        PdhgResult r = pdhg.solve(problem, backend, &pdhg_token);
        {
            std::lock_guard<std::mutex> lock(shared.mtx);
            pdhg_result = std::move(r);
            shared.pdhg_finished = true;
        }
        shared.cv.notify_all();
    });

    // Wait for either a winner, or PDHG finished with every attempt it ever
    // launched (there cannot be more after that -- checkpoints only fire
    // from inside PDHG's own loop, which has now returned) also finished.
    {
        std::unique_lock<std::mutex> lock(shared.mtx);
        shared.cv.wait(lock, [&] {
            return shared.winner.has_value()
                || (shared.pdhg_finished && shared.finished_attempts >= shared.launched_attempts);
        });
    }

    // Last resort: PDHG stopped (converged, ran out of budget, or was
    // cancelled by an earlier winner) without any checkpoint attempt
    // winning. If it has a real point to offer, try ONE more crossover
    // attempt directly on it before declaring failure -- this is the design
    // degrading gracefully to an ordinary, non-concurrent crossover exactly
    // when concurrency had nothing left to overlap with.
    if (!shared.winner.has_value() && !pdhg_result.primal.empty()) {
        WarmStart ws = guess_basis_from_point(problem, pdhg_result.primal, pdhg_result.dual);
        Simplex simplex(options_.crossover_simplex);
        SimplexResult r = simplex.solve(problem, nullptr, &ws);
        ++shared.launched_attempts;
        ++shared.finished_attempts;
        if (is_valid(r.status)) shared.winner = from(r, -1);
    }

    pdhg_thread.join();
    // Copy out and join outside the lock: an in-flight attempt thread's
    // final act is `report()`, which takes shared.mtx -- joining while
    // holding it would deadlock against exactly that.
    std::vector<std::thread> attempts_to_join;
    {
        std::lock_guard<std::mutex> lock(shared.mtx);
        attempts_to_join = std::move(shared.attempt_threads);
    }
    for (std::thread& t : attempts_to_join)
        if (t.joinable()) t.join();

    CrossoverResult final_result;
    if (shared.winner.has_value()) {
        final_result = std::move(*shared.winner);
    } else {
        final_result.status = CrossoverStatus::Failed;
        final_result.message = "no crossover attempt (including the final PDHG iterate) reached "
                               "a valid terminal status; PDHG stopped with '"
                              + std::string(to_string(pdhg_result.status)) + "'";
    }
    final_result.checkpoint_attempts = shared.launched_attempts;
    final_result.pdhg_seconds = pdhg_result.seconds;
    final_result.total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return final_result;
}

}  // namespace sov
