// From-scratch revised simplex -- Build Map ticket #4, gate M0. Bible S4.2 Engine C.
//
// PURPOSE. This is the correctness oracle we own. Its first job is not speed;
// it is to be an independent answer that the GPU engines of Phase 2 can be
// checked against, the way the external CLI oracle (#2) checks it. The ticket
// is explicit: "Build it correct and clear, not fast", because Bible Part II
// says chasing the saturated CPU curve is the trap.
//
// WHICH SIMPLEX. The Build Map names a revised *dual* simplex. This implements
// a revised *primal* simplex with a bounded-variable formulation and an
// explicit Phase I. The reason is the oracle role: dual simplex earns its keep
// on re-optimization after a bound change -- which is why it is the industry
// default inside branch-and-bound -- but it needs a dual-feasible starting
// basis, and manufacturing one for an arbitrary MPS instance adds machinery
// whose bugs would be indistinguishable from the bugs it is meant to catch.
// The primal two-phase method starts from a basis that always exists (the
// slacks), so correctness is checkable end to end. The dual variant belongs
// with ticket #38, where warm-starting is what actually needs it.
//
// ANTI-CYCLING is built in from the start, not bolted on. Degenerate vertices
// are the norm on the instances the problem statement grades against, and a
// simplex without a cycling guard does not merely run slowly there -- it never
// terminates. The scheme is a largest-improvement rule that falls back to
// Bland's rule once a degenerate stall is detected, then returns to the fast
// rule once progress resumes. Bland's rule guarantees termination; the
// switching keeps the usual case from paying for it.
#pragma once

#include <string>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/problem.hpp"

namespace sov {

enum class SolveStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    NumericalFailure,
    NotSolved,
};

const char* to_string(SolveStatus s) noexcept;

struct SimplexOptions {
    /// Primal feasibility tolerance: how far a variable may sit outside its
    /// bounds and still count as feasible.
    Real primal_tolerance = 1e-7;
    /// Reduced-cost tolerance: how negative a reduced cost must be to be worth
    /// pivoting on.
    Real dual_tolerance = 1e-7;
    /// A pivot element smaller than this in absolute terms is rejected.
    Real pivot_tolerance = 1e-9;

    /// ...and one smaller than this fraction of the largest entry in the same
    /// pivot column is rejected too. The relative test is the one that matters:
    /// an absolute threshold passes a pivot that is tiny *for its column*, the
    /// resulting basis is near-singular, and the damage only surfaces later as
    /// a failed refactorization or -- worse -- as a plausible wrong answer.
    Real relative_pivot_tolerance = 1e-7;

    long long max_iterations = 1000000;
    double time_limit_seconds = 0.0;    ///< 0 = no limit

    /// Rebuild the basis factorization from scratch every N pivots. Product-form
    /// updates accumulate error; periodic refactorization is what bounds it.
    int refactor_frequency = 100;

    /// Number of consecutive degenerate (zero-progress) pivots before switching
    /// to Bland's rule. Small enough to catch a cycle quickly, large enough that
    /// ordinary degeneracy does not pay Bland's slower pricing.
    int degenerate_pivots_before_bland = 40;

    /// Skip the switching heuristic and use Bland's rule throughout. Slow, but
    /// it is the provably terminating configuration, so it is what a cycling
    /// investigation should reach for.
    bool always_bland = false;

    bool verbose = false;
};

struct SimplexResult {
    SolveStatus status = SolveStatus::NotSolved;

    /// Objective in the problem's own sense, including its constant term.
    /// Meaningful only when status is Optimal.
    Real objective = 0.0;

    std::vector<Real> primal;    ///< column values, length num_cols
    std::vector<Real> dual;      ///< row duals, length num_rows
    std::vector<Real> reduced_costs;

    long long iterations = 0;
    long long phase1_iterations = 0;
    long long refactorizations = 0;
    long long bland_switches = 0;   ///< how often anti-cycling had to engage
    double seconds = 0.0;

    Real primal_infeasibility = 0.0;
    Real dual_infeasibility = 0.0;
    std::string message;
};

class Simplex {
public:
    explicit Simplex(SimplexOptions options = {}) : options_(options) {}

    /// Solve the continuous relaxation. Integrality is ignored -- branch and
    /// bound is ticket #38; here an integer column is just a bounded one.
    SimplexResult solve(const Problem& problem);

    const SimplexOptions& options() const noexcept { return options_; }

private:
    SimplexOptions options_;
};

}  // namespace sov
