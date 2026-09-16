// Concurrent checkpoint crossover -- Build Map ticket #11, gate M2. Bible
// S4.3, S5.2 (Bet 2).
//
// PURPOSE. PDHG (#8) and the IPM (#9) only ever reach an INTERIOR point --
// PDHG's native accuracy is ~1e-4, and neither engine terminates on an exact
// basic vertex the way simplex does. Branch-and-bound (#38) and everything
// downstream of it needs an exact vertex. Classical crossover gets one by
// treating the interior point as a basis GUESS and finishing with ordinary
// simplex pivots -- but "classical crossover is essentially simplex again"
// (the ticket's own "Watch out"), so running it once, serially, after PDHG
// has fully converged deletes the entire GPU win (Amdahl's Law).
//
// THE FIX: run crossover CONCURRENTLY with PDHG, not after it. Every
// checkpoint PDHG itself already computes (ticket #8's restart-check
// cadence -- an unscaled candidate point it evaluates for convergence every
// `restart_check_period` iterations, whether or not it decides to restart)
// is handed, via PdhgOptions::checkpoint, to a freshly launched crossover
// attempt in its own thread. Crossover attempts and PDHG's own continuing
// iteration race each other exactly like ticket #10's engine race: the
// first attempt to reach a verified exact vertex wins, and every other
// in-flight attempt -- including PDHG itself, if it is still running -- is
// cancelled. A late, well-converged checkpoint is not required to win; an
// EARLY, rougher checkpoint that happens to guess a basis simplex can
// finish from quickly is just as good a winner, and often faster.
//
// BASIS CONSTRUCTION (guess_basis_from_point, below). The m variables
// (structural columns or row logicals, in the internal [A -I] indexing
// ticket #4 already uses) sitting furthest from their own nearer bound are
// guessed basic; the rest are guessed nonbasic, snapped to whichever bound
// their guessed value sits closer to. This is a GUESS, not a claim: Simplex
// itself (ticket #4's own Phase I "dual push" restores feasibility from
// whatever basis it is handed, Phase II "primal push" then optimizes)
// verifies and corrects it exactly as it would from any other starting
// basis, and falls back to the guaranteed-nonsingular slack basis if the
// guess is singular. A bad guess costs pivots; it never costs correctness.
#pragma once

#include <functional>
#include <span>
#include <string>
#include <vector>

#include "sovereign/backend.hpp"
#include "sovereign/numeric.hpp"
#include "sovereign/pdhg.hpp"
#include "sovereign/problem.hpp"
#include "sovereign/simplex.hpp"

namespace sov {

enum class CrossoverStatus {
    Optimal,      ///< a crossover attempt produced an exact optimal vertex
    Infeasible,   ///< a crossover attempt's own simplex proved infeasibility
    Unbounded,    ///< a crossover attempt's own simplex proved unboundedness
    Failed,       ///< no attempt (including PDHG's own final iterate) reached
                  ///< a trustworthy terminal status
    NotSolved,
};

const char* to_string(CrossoverStatus s) noexcept;

struct CrossoverOptions {
    /// PDHG's own tuning (tolerance, restart cadence, iteration/time
    /// budget). `checkpoint` is overwritten internally -- Crossover::solve
    /// is what installs it.
    PdhgOptions pdhg;

    /// Options for EACH crossover attempt's own Simplex solve. Every
    /// checkpoint gets its own attempt with these same options; they are
    /// independent solves (own factorization, own pivot count), so a large
    /// instance with many checkpoints launches many concurrent attempts --
    /// each one individually bounded by these limits (and, once a real GPU
    /// is available, cheap relative to the GPU iteration they overlap).
    SimplexOptions crossover_simplex;

    bool verbose = false;
};

struct CrossoverResult {
    CrossoverStatus status = CrossoverStatus::NotSolved;

    Real objective = 0.0;
    std::vector<Real> primal;
    std::vector<Real> dual;
    std::vector<Real> reduced_costs;

    /// The PDHG iteration whose checkpoint produced the winning attempt, or
    /// -1 if the winner was the single fallback attempt run directly on
    /// PDHG's own returned point after every checkpoint attempt and PDHG
    /// itself had already finished without an earlier winner.
    long long winning_checkpoint_iteration = -1;

    long long checkpoint_attempts = 0;    ///< crossover attempts launched in total

    double pdhg_seconds = 0.0;      ///< PDHG's own reported wall time
    double total_seconds = 0.0;     ///< wall time for the whole concurrent solve

    std::string message;
};

/// Builds a WarmStart basis guess for `problem` from an interior point
/// (x, y). `x` must have length num_cols, `y` length num_rows. Standalone
/// and pure (no threading, no solving) so it is independently testable and
/// usable directly against a single IPM or PDHG result without the
/// concurrent checkpoint harness.
WarmStart guess_basis_from_point(const Problem& problem, std::span<const Real> x,
                                 std::span<const Real> y);

class Crossover {
public:
    explicit Crossover(CrossoverOptions options = {}) : options_(options) {}

    /// Runs PDHG on `backend` and, concurrently, a crossover attempt from
    /// every checkpoint PDHG reaches -- see the file comment above. Blocks
    /// until either a valid winner is accepted or every attempt (including
    /// the final fallback on PDHG's own returned point) has been tried
    /// without producing one. Every launched thread is joined before this
    /// returns, win or lose (same discipline as ticket #10's race_solve).
    CrossoverResult solve(const Problem& problem, Backend& backend) const;

private:
    CrossoverOptions options_;
};

}  // namespace sov
