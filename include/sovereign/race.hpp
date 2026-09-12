// Concurrent engine race -- Build Map ticket #10, gate M1. Bible S4.2,
// Part III principle "concurrency by default."
//
// PURPOSE. Launch every continuous engine on the SAME model at once; return
// the answer from whichever reaches a genuinely trustworthy terminal state
// first; cancel the rest. "First to a *valid* answer" is the whole point --
// a fast wrong answer must lose, so validity is checked per engine against
// what that engine can actually promise (see is_valid_* in race.cpp), not
// just "did solve() return".
//
// WHY THIS IS GENERAL (Build Map's own requirement: "reused verbatim by the
// decomposition race #25"). The harness only needs three things from an
// engine: run it with a CancellationToken, ask whether its terminal status
// is one this harness should trust, and read back objective/primal/dual in
// a common shape. Nothing here is specific to simplex/PDHG/IPM being the
// three engines racing today -- #25 racing a decomposition mode against the
// monolithic solve is the same shape of problem with different contestants.
//
// CLEAN CANCELLATION (the ticket's explicit "Watch out"). Every engine
// checks a CancellationToken once per iteration/pivot (see cancellation.hpp)
// and returns Cancelled promptly. The race sets every LOSING engine's token
// the moment a winner is accepted, then joins every worker thread before
// returning -- so "cancel" here means "every thread actually stops and its
// resources are released before solve() returns," not "we stopped waiting
// for it." No leaked memory, no dangling threads, by construction.
#pragma once

#include <string>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/problem.hpp"

namespace sov {

enum class RaceEngine { Simplex, Pdhg, Ipm };

constexpr const char* to_string(RaceEngine e) noexcept {
    switch (e) {
        case RaceEngine::Simplex: return "simplex";
        case RaceEngine::Pdhg:    return "pdhg";
        case RaceEngine::Ipm:     return "ipm";
    }
    return "?";
}

struct RaceOptions {
    bool use_simplex = true;
    bool use_pdhg = true;
    bool use_ipm = true;

    /// Shared wall-clock budget applied to every entrant (0 = each engine's
    /// own default). A generous default matters here specifically: a race
    /// with no budget at all can hang forever if every entrant is disabled
    /// or misconfigured, which is a config error, not a "no valid answer"
    /// outcome -- see RaceResult::message when that happens.
    double time_limit_seconds = 60.0;
};

struct RaceResult {
    bool has_winner = false;
    RaceEngine winner{};             ///< meaningful only if has_winner
    std::string status;              ///< the winning engine's own status string
    Real objective = 0.0;
    std::vector<Real> primal;
    std::vector<Real> dual;

    double seconds = 0.0;            ///< wall-clock until the winner was accepted
    int engines_launched = 0;
    int engines_completed = 0;       ///< includes cancelled/failed entrants
    std::string message;             ///< set when has_winner is false
};

/// Races the enabled engines on `problem`, blocking until either a valid
/// winner is accepted or every entrant has finished without producing one.
/// Every launched thread is joined before this returns, win or lose.
RaceResult race_solve(const Problem& problem, const RaceOptions& options = {});

}  // namespace sov
