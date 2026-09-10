// From-scratch revised simplex -- Build Map ticket #4. Bible S4.2 Engine C.
//
// COMPUTATIONAL FORM. Every row lo_i <= a_i^T x <= hi_i gets a logical variable
// s_i = a_i^T x, so the system becomes
//
//     [ A  -I ] [ x ; s ] = 0,      lo_col <= x <= hi_col,  lo_row <= s <= hi_row
//
// Every variable then has a (possibly infinite) bound pair and the right-hand
// side is zero. Two consequences make this the right choice for an oracle:
// a starting basis always exists (the logicals, whose basis matrix is -I), and
// ranges, equalities and free rows are all just bound pairs rather than
// separate cases.
//
// LINEAR ALGEBRA. Dense LU of the basis with partial pivoting, refactorized
// periodically, with product-form (eta) updates in between. That is the
// textbook revised simplex, and the dense basis is a deliberate scope choice:
// it costs O(m^2) memory and O(m^3) per refactorization, which is fine for the
// small and medium Netlib instances an oracle needs and hopeless beyond a few
// thousand rows. Sparse LU with Markowitz pivoting is the real answer and is
// named as follow-up work rather than half-attempted here. The size guard
// below refuses instead of quietly grinding.
#include "sovereign/simplex.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>

namespace sov {

const char* to_string(SolveStatus s) noexcept {
    switch (s) {
        case SolveStatus::Optimal:          return "optimal";
        case SolveStatus::Infeasible:       return "infeasible";
        case SolveStatus::Unbounded:        return "unbounded";
        case SolveStatus::IterationLimit:   return "iteration_limit";
        case SolveStatus::TimeLimit:        return "time_limit";
        case SolveStatus::NumericalFailure: return "numerical_failure";
        case SolveStatus::NotSolved:        return "not_solved";
    }
    return "unknown";
}

namespace {

/// Beyond this many rows the dense basis stops being reasonable: the
/// factorization is O(m^3) and the storage O(m^2). Refusing with a clear
/// message beats appearing to work.
constexpr Idx kMaxDenseRows = 3000;

enum class NonbasicState : std::uint8_t { AtLower, AtUpper, Free };

// --------------------------------------------------------------------------
// Basis factorization: dense LU + product-form updates
// --------------------------------------------------------------------------

class BasisFactorization {
public:
    explicit BasisFactorization(Idx m) : m_(m), lu_(static_cast<std::size_t>(m) * m),
                                         perm_(static_cast<std::size_t>(m)),
                                         work_(static_cast<std::size_t>(m)) {}

    Idx eta_count() const noexcept { return static_cast<Idx>(etas_.size()); }

    /// Rebuild from the basis columns. Returns NumericalFailure if the basis
    /// turns out to be singular, which the caller must treat as a real failure
    /// rather than retrying -- a singular basis means the pivot choice that
    /// produced it was unsound.
    template <class ColumnLoader>
    Status factorize(std::span<const Idx> basis, ColumnLoader&& load_column) {
        etas_.clear();
        std::fill(lu_.begin(), lu_.end(), Real{0});

        // Column-major: entry (i, j) at lu_[j*m + i]. Column-major suits the
        // elimination loop below, which sweeps down a column at a time.
        for (Idx j = 0; j < m_; ++j) {
            load_column(basis[static_cast<std::size_t>(j)], work_);
            for (Idx i = 0; i < m_; ++i)
                lu_[static_cast<std::size_t>(j) * m_ + i] = work_[static_cast<std::size_t>(i)];
        }
        std::iota(perm_.begin(), perm_.end(), Idx{0});

        for (Idx k = 0; k < m_; ++k) {
            // Partial pivoting. Without it a zero or tiny pivot on a perfectly
            // good basis destroys the factorization.
            Idx best = k;
            Real best_mag = std::abs(at(k, k));
            for (Idx i = k + 1; i < m_; ++i) {
                const Real mag = std::abs(at(i, k));
                if (mag > best_mag) { best_mag = mag; best = i; }
            }
            if (best_mag < 1e-12) return Status::NumericalFailure;

            if (best != k) {
                for (Idx j = 0; j < m_; ++j) std::swap(at(k, j), at(best, j));
                std::swap(perm_[static_cast<std::size_t>(k)],
                          perm_[static_cast<std::size_t>(best)]);
            }

            const Real pivot = at(k, k);
            for (Idx i = k + 1; i < m_; ++i) {
                const Real factor = at(i, k) / pivot;
                if (factor == 0.0) continue;
                at(i, k) = factor;                        // store L below the diagonal
                for (Idx j = k + 1; j < m_; ++j) at(i, j) -= factor * at(k, j);
            }
        }
        return Status::Ok;
    }

    /// Solve B z = v in place.
    ///
    /// B = B0 * E1 * ... * Ek, so z = Ek^-1 ... E1^-1 B0^-1 v: the LU solve
    /// first, then the etas in the order they were created.
    void ftran(std::vector<Real>& v) const {
        // Apply the row permutation, then forward and back substitution.
        for (Idx i = 0; i < m_; ++i)
            work_[static_cast<std::size_t>(i)] = v[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])];
        for (Idx i = 0; i < m_; ++i) {
            Real acc = work_[static_cast<std::size_t>(i)];
            for (Idx j = 0; j < i; ++j) acc -= at(i, j) * work_[static_cast<std::size_t>(j)];
            work_[static_cast<std::size_t>(i)] = acc;
        }
        for (Idx i = m_ - 1; i >= 0; --i) {
            Real acc = work_[static_cast<std::size_t>(i)];
            for (Idx j = i + 1; j < m_; ++j) acc -= at(i, j) * work_[static_cast<std::size_t>(j)];
            work_[static_cast<std::size_t>(i)] = acc / at(i, i);
        }
        std::copy(work_.begin(), work_.end(), v.begin());

        for (const Eta& e : etas_) {
            const auto p = static_cast<std::size_t>(e.position);
            const Real vp = v[p] / e.values[p];
            v[p] = vp;
            if (vp == 0.0) continue;
            for (Idx i = 0; i < m_; ++i) {
                if (static_cast<std::size_t>(i) == p) continue;
                v[static_cast<std::size_t>(i)] -= e.values[static_cast<std::size_t>(i)] * vp;
            }
        }
    }

    /// Solve B^T z = v in place: the etas transposed in reverse order, then the
    /// transposed LU solve.
    void btran(std::vector<Real>& v) const {
        for (auto it = etas_.rbegin(); it != etas_.rend(); ++it) {
            const Eta& e = *it;
            const auto p = static_cast<std::size_t>(e.position);
            Real dot = 0.0;
            for (Idx i = 0; i < m_; ++i)
                dot += e.values[static_cast<std::size_t>(i)] * v[static_cast<std::size_t>(i)];
            v[p] = (v[p] - (dot - e.values[p] * v[p])) / e.values[p];
        }

        // (P A)= L U  =>  A^T = U^T L^T P, so solve U^T then L^T, then unpermute.
        for (Idx i = 0; i < m_; ++i) {
            Real acc = v[static_cast<std::size_t>(i)];
            for (Idx j = 0; j < i; ++j) acc -= at(j, i) * v[static_cast<std::size_t>(j)];
            v[static_cast<std::size_t>(i)] = acc / at(i, i);
        }
        for (Idx i = m_ - 1; i >= 0; --i) {
            Real acc = v[static_cast<std::size_t>(i)];
            for (Idx j = i + 1; j < m_; ++j) acc -= at(j, i) * v[static_cast<std::size_t>(j)];
            v[static_cast<std::size_t>(i)] = acc;
        }
        for (Idx i = 0; i < m_; ++i)
            work_[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])] =
                v[static_cast<std::size_t>(i)];
        std::copy(work_.begin(), work_.end(), v.begin());
    }

    void push_eta(Idx position, const std::vector<Real>& direction) {
        etas_.push_back(Eta{position, direction});
    }

private:
    Real& at(Idx i, Idx j) { return lu_[static_cast<std::size_t>(j) * m_ + i]; }
    const Real& at(Idx i, Idx j) const { return lu_[static_cast<std::size_t>(j) * m_ + i]; }

    struct Eta { Idx position; std::vector<Real> values; };

    Idx m_;
    std::vector<Real> lu_;
    std::vector<Idx> perm_;
    mutable std::vector<Real> work_;
    std::vector<Eta> etas_;
};

// --------------------------------------------------------------------------
// The solver
// --------------------------------------------------------------------------

class SimplexImpl {
public:
    SimplexImpl(const Problem& p, const SimplexOptions& opt)
        : problem_(p), opt_(opt),
          n_(p.num_cols()), m_(p.num_rows()), total_(p.num_cols() + p.num_rows()),
          csc_(p.matrix().to_csc()),
          factor_(p.num_rows()) {}

    SimplexResult run();

private:
    void setup();
    void load_column(Idx j, std::vector<Real>& dense) const;
    Real column_dot(Idx j, const std::vector<Real>& y) const;
    void recompute_basic_values();
    Status refactorize();

    /// Total bound violation across basic variables.
    Real primal_infeasibility() const;

    bool phase1_costs();      ///< returns true if already feasible
    Idx price(bool phase1, Real* best_reduced, Real* direction);
    bool ratio_test(Idx entering, Real direction, bool phase1,
                    Idx* leaving_position, Real* step, bool* bound_flip);

    SolveStatus iterate(bool phase1, long long* iterations);

    const Problem& problem_;
    SimplexOptions opt_;
    Idx n_, m_, total_;
    CscMatrix csc_;
    BasisFactorization factor_;

    Real objective_sign_ = 1.0;   ///< +1 minimize, -1 maximize (we always minimize)

    std::vector<Real> lower_, upper_, cost_, x_;
    std::vector<Idx> basis_;              ///< variable index at each basis position
    std::vector<Idx> basis_position_;     ///< -1 if nonbasic
    std::vector<NonbasicState> state_;
    std::vector<Real> phase1_cost_;

    // Bland's rule only guarantees termination against a FIXED objective, and
    // the Phase I objective is not fixed: it is the gradient of a piecewise
    // linear infeasibility sum, recomputed each iteration. During a degenerate
    // stall the set of violated variables can shuffle without the total
    // changing, the objective moves under Bland's feet, and the guarantee
    // evaporates -- which is exactly how `brandy` cycled forever with the
    // anti-cycling rule supposedly engaged. So the cost vector is frozen for
    // the duration of an anti-cycling episode.
    std::vector<Real> frozen_phase1_cost_;
    bool phase1_cost_frozen_ = false;

    // Scratch, allocated once. The inner loop must not touch the allocator.
    mutable std::vector<Real> scratch_column_, direction_, duals_, basic_costs_;

    long long pivots_since_refactor_ = 0;
    int consecutive_degenerate_ = 0;
    bool bland_ = false;

    // EXPAND anti-degeneracy (Gill, Murray, Saunders & Wright 1989). A working
    // feasibility tolerance that GROWS by expand_rate_ every pivot and is reset
    // at each refactorization. Because it is strictly increasing between
    // resets, the relaxed feasible region is different every iteration, so a
    // basis cannot recur -- the ratio test always has room for a strictly
    // positive step, and the method cannot cycle even on a fully degenerate
    // vertex. This is the real termination guarantee; Bland's rule is kept as a
    // cheap secondary net. Without EXPAND, an absolute pivot tolerance large
    // enough to be safe also excludes small-but-real pivots from the ratio
    // test, which quietly voids Bland's guarantee -- `brandy` cycles forever.
    Real expand_tol_ = 0.0;
    Real expand_rate_ = 0.0;

    // Cleared for the optimality-verification endgame in run(): with expansion
    // off the ratio test is exact, so a solution "optimal" only under the
    // relaxed tolerance is refined to a true vertex. Without this the relaxed
    // slack leaks into the reported objective by ~1e-5 (scsd1, wood1p).
    bool allow_expand_ = true;

    // Progress-stall net. Tracks the best merit seen (infeasibility in phase 1,
    // objective in phase 2); if it does not improve for a long run of pivots
    // the solve is refactorized once and, failing that, abandoned with
    // NumericalFailure -- never a wrong Optimal.
    Real best_merit_ = kInfinity;
    long long stall_pivots_ = 0;

    SimplexResult result_;
    std::chrono::steady_clock::time_point start_;
};

void SimplexImpl::load_column(Idx j, std::vector<Real>& dense) const {
    std::fill(dense.begin(), dense.end(), Real{0});
    if (j < n_) {
        const auto begin = csc_.col_ptr()[static_cast<std::size_t>(j)];
        const auto end = csc_.col_ptr()[static_cast<std::size_t>(j) + 1];
        for (Idx k = begin; k < end; ++k)
            dense[static_cast<std::size_t>(csc_.row_idx()[static_cast<std::size_t>(k)])] =
                csc_.values()[static_cast<std::size_t>(k)];
    } else {
        // Logical column for row (j - n): the -I block.
        dense[static_cast<std::size_t>(j - n_)] = -1.0;
    }
}

Real SimplexImpl::column_dot(Idx j, const std::vector<Real>& y) const {
    if (j >= n_) return -y[static_cast<std::size_t>(j - n_)];
    Real acc = 0.0;
    const auto begin = csc_.col_ptr()[static_cast<std::size_t>(j)];
    const auto end = csc_.col_ptr()[static_cast<std::size_t>(j) + 1];
    for (Idx k = begin; k < end; ++k)
        acc += csc_.values()[static_cast<std::size_t>(k)]
             * y[static_cast<std::size_t>(csc_.row_idx()[static_cast<std::size_t>(k)])];
    return acc;
}

void SimplexImpl::setup() {
    lower_.resize(static_cast<std::size_t>(total_));
    upper_.resize(static_cast<std::size_t>(total_));
    cost_.assign(static_cast<std::size_t>(total_), 0.0);
    x_.assign(static_cast<std::size_t>(total_), 0.0);
    state_.assign(static_cast<std::size_t>(total_), NonbasicState::AtLower);
    basis_position_.assign(static_cast<std::size_t>(total_), Idx{-1});

    // We always minimize internally; a maximization problem is minimized with
    // negated costs and the objective is reported in the original sense.
    objective_sign_ = (problem_.sense() == ObjSense::Maximize) ? -1.0 : 1.0;

    for (Idx j = 0; j < n_; ++j) {
        lower_[static_cast<std::size_t>(j)] = problem_.col_lower()[static_cast<std::size_t>(j)];
        upper_[static_cast<std::size_t>(j)] = problem_.col_upper()[static_cast<std::size_t>(j)];
        cost_[static_cast<std::size_t>(j)] =
            objective_sign_ * problem_.objective()[static_cast<std::size_t>(j)];
    }
    for (Idx i = 0; i < m_; ++i) {
        // The logical equals the row activity, so it inherits the row's bounds.
        lower_[static_cast<std::size_t>(n_ + i)] = problem_.row_lower()[static_cast<std::size_t>(i)];
        upper_[static_cast<std::size_t>(n_ + i)] = problem_.row_upper()[static_cast<std::size_t>(i)];
    }

    // Nonbasic variables start at a finite bound; a free variable starts at zero.
    for (Idx j = 0; j < total_; ++j) {
        const auto k = static_cast<std::size_t>(j);
        if (is_finite_bound(lower_[k])) {
            state_[k] = NonbasicState::AtLower;
            x_[k] = lower_[k];
        } else if (is_finite_bound(upper_[k])) {
            state_[k] = NonbasicState::AtUpper;
            x_[k] = upper_[k];
        } else {
            state_[k] = NonbasicState::Free;
            x_[k] = 0.0;
        }
    }

    // Starting basis: all logicals. Its basis matrix is -I, always nonsingular,
    // which is the whole reason for the [A -I] formulation.
    basis_.resize(static_cast<std::size_t>(m_));
    for (Idx i = 0; i < m_; ++i) {
        basis_[static_cast<std::size_t>(i)] = n_ + i;
        basis_position_[static_cast<std::size_t>(n_ + i)] = i;
    }

    scratch_column_.assign(static_cast<std::size_t>(m_), 0.0);
    direction_.assign(static_cast<std::size_t>(m_), 0.0);
    duals_.assign(static_cast<std::size_t>(m_), 0.0);
    basic_costs_.assign(static_cast<std::size_t>(m_), 0.0);
    phase1_cost_.assign(static_cast<std::size_t>(total_), 0.0);

    // EXPAND starts at half the feasibility tolerance and is scheduled to reach
    // the full tolerance by the next refactorization, then resets.
    const int window = std::max(1, opt_.refactor_frequency);
    expand_tol_ = 0.5 * opt_.primal_tolerance;
    expand_rate_ = 0.5 * opt_.primal_tolerance / window;
}

Status SimplexImpl::refactorize() {
    const Status s = factor_.factorize(
        basis_, [this](Idx j, std::vector<Real>& dense) { load_column(j, dense); });
    pivots_since_refactor_ = 0;
    ++result_.refactorizations;

    // EXPAND reset. The tolerance has been growing all window; drop it back and
    // clean up the small controlled infeasibilities EXPAND deliberately let
    // accumulate -- any basic variable within the tolerance of a bound is
    // snapped exactly onto it. Doing this only at refactor time keeps the inner
    // loop cheap while bounding the drift.
    if (s == Status::Ok) {
        const Real clean = opt_.primal_tolerance;
        for (Idx i = 0; i < m_; ++i) {
            const auto j = static_cast<std::size_t>(basis_[static_cast<std::size_t>(i)]);
            if (is_finite_bound(lower_[j]) && std::abs(x_[j] - lower_[j]) < clean) x_[j] = lower_[j];
            else if (is_finite_bound(upper_[j]) && std::abs(x_[j] - upper_[j]) < clean) x_[j] = upper_[j];
        }
        expand_tol_ = allow_expand_ ? 0.5 * opt_.primal_tolerance : 0.0;
    }
    return s;
}

void SimplexImpl::recompute_basic_values() {
    // M x = 0, so B x_B = -N x_N. Build the right-hand side from the nonbasic
    // values, then FTRAN.
    std::vector<Real> rhs(static_cast<std::size_t>(m_), 0.0);
    for (Idx j = 0; j < total_; ++j) {
        if (basis_position_[static_cast<std::size_t>(j)] >= 0) continue;
        const Real xj = x_[static_cast<std::size_t>(j)];
        if (xj == 0.0) continue;
        if (j >= n_) {
            rhs[static_cast<std::size_t>(j - n_)] -= -1.0 * xj;
        } else {
            const auto begin = csc_.col_ptr()[static_cast<std::size_t>(j)];
            const auto end = csc_.col_ptr()[static_cast<std::size_t>(j) + 1];
            for (Idx k = begin; k < end; ++k)
                rhs[static_cast<std::size_t>(csc_.row_idx()[static_cast<std::size_t>(k)])] -=
                    csc_.values()[static_cast<std::size_t>(k)] * xj;
        }
    }
    factor_.ftran(rhs);
    for (Idx i = 0; i < m_; ++i)
        x_[static_cast<std::size_t>(basis_[static_cast<std::size_t>(i)])] =
            rhs[static_cast<std::size_t>(i)];
}

Real SimplexImpl::primal_infeasibility() const {
    Real total = 0.0;
    for (Idx i = 0; i < m_; ++i) {
        const auto j = static_cast<std::size_t>(basis_[static_cast<std::size_t>(i)]);
        const Real v = x_[j];
        if (v < lower_[j] - opt_.primal_tolerance) total += lower_[j] - v;
        else if (v > upper_[j] + opt_.primal_tolerance) total += v - upper_[j];
    }
    return total;
}

bool SimplexImpl::phase1_costs() {
    // The Phase I objective is the sum of bound violations. Its gradient with
    // respect to a basic variable is -1 when the variable sits below its lower
    // bound (raising it helps) and +1 when above its upper bound. Nonbasic
    // variables are at a bound and so contribute nothing.
    std::fill(phase1_cost_.begin(), phase1_cost_.end(), Real{0});
    bool feasible = true;
    for (Idx i = 0; i < m_; ++i) {
        const auto j = static_cast<std::size_t>(basis_[static_cast<std::size_t>(i)]);
        const Real v = x_[j];
        if (v < lower_[j] - opt_.primal_tolerance) { phase1_cost_[j] = -1.0; feasible = false; }
        else if (v > upper_[j] + opt_.primal_tolerance) { phase1_cost_[j] = 1.0; feasible = false; }
    }
    return feasible;
}

Idx SimplexImpl::price(bool phase1, Real* best_reduced, Real* direction) {
    const std::vector<Real>& costs =
        phase1 ? (phase1_cost_frozen_ ? frozen_phase1_cost_ : phase1_cost_) : cost_;

    for (Idx i = 0; i < m_; ++i)
        basic_costs_[static_cast<std::size_t>(i)] =
            costs[static_cast<std::size_t>(basis_[static_cast<std::size_t>(i)])];
    duals_ = basic_costs_;
    factor_.btran(duals_);

    Idx best = -1;
    Real best_value = 0.0;
    *direction = 0.0;

    for (Idx j = 0; j < total_; ++j) {
        const auto k = static_cast<std::size_t>(j);
        if (basis_position_[k] >= 0) continue;
        if (lower_[k] == upper_[k]) continue;          // fixed: never enters

        const Real reduced = costs[k] - column_dot(j, duals_);

        // A variable at its lower bound can only increase, so it helps only
        // when its reduced cost is negative; at its upper bound, only decrease
        // helps. A free variable may move either way.
        Real dir = 0.0;
        if (state_[k] == NonbasicState::AtLower) {
            if (reduced < -opt_.dual_tolerance) dir = 1.0;
        } else if (state_[k] == NonbasicState::AtUpper) {
            if (reduced > opt_.dual_tolerance) dir = -1.0;
        } else {
            if (reduced < -opt_.dual_tolerance) dir = 1.0;
            else if (reduced > opt_.dual_tolerance) dir = -1.0;
        }
        if (dir == 0.0) continue;

        if (bland_) {
            // Bland's rule: the lowest-index eligible column, always. Slow, but
            // it is what makes termination provable, so the cycling escape uses
            // exactly this and nothing cleverer.
            best = j;
            best_value = reduced;
            *direction = dir;
            break;
        }
        // Dantzig pricing otherwise: steepest reduced cost.
        if (std::abs(reduced) > std::abs(best_value)) {
            best = j;
            best_value = reduced;
            *direction = dir;
        }
    }
    *best_reduced = best_value;
    return best;
}

bool SimplexImpl::ratio_test(Idx entering, Real direction, bool phase1,
                             Idx* leaving_position, Real* step, bool* bound_flip) {
    load_column(entering, scratch_column_);
    direction_ = scratch_column_;
    factor_.ftran(direction_);      // direction_ = B^-1 * column

    *leaving_position = -1;
    *bound_flip = false;

    // Absolute pivot floor ONLY. Bland's termination proof needs every
    // nonzero-pivot row in the running; a floor that scales with the column's
    // largest entry silently drops small-but-real pivots, and if the excluded
    // row is the lowest-indexed one at the minimum ratio, Bland is no longer
    // Bland. Numerical safety instead comes from EXPAND (below) plus the Harris
    // second pass preferring the largest pivot on ties.
    const Real zero_pivot = opt_.pivot_tolerance;

    // The gap between basic variable i and the bound it is heading toward, and
    // |pivot|. Returns false when i cannot block (pivot ~0, or the bound in
    // that direction is infinite).
    auto blocker = [&](Idx i, Real* gap, Real* absd, bool* toward_lower) -> bool {
        const Real d = direction_[static_cast<std::size_t>(i)];
        *absd = std::abs(d);
        if (*absd < zero_pivot) return false;

        const auto j = static_cast<std::size_t>(basis_[static_cast<std::size_t>(i)]);
        const Real rate = -d * direction;             // d(x_Bi)/dt; |rate| == |d|
        const Real v = x_[j];

        Real lo = lower_[j], hi = upper_[j];
        if (phase1) {
            // In Phase I an infeasible basic variable is allowed to travel to
            // the bound it is violating; arriving there removes the
            // infeasibility, which is the point of the phase.
            if (v < lo - opt_.primal_tolerance) { hi = lo; lo = -kInfinity; }
            else if (v > hi + opt_.primal_tolerance) { lo = hi; hi = kInfinity; }
        }

        if (rate < 0.0) {
            if (!is_finite_bound(lo)) return false;
            *gap = v - lo;
            *toward_lower = true;
        } else {
            if (!is_finite_bound(hi)) return false;
            *gap = hi - v;
            *toward_lower = false;
        }
        if (*gap < 0.0) *gap = 0.0;    // already marginally past (EXPAND drift)
        return true;
    };

    // Pass 1 (Harris): the largest step for which every basic variable stays
    // within EXPAND's relaxed tolerance of its bound. Because expand_tol_ > 0
    // this maximum is strictly positive even at a fully degenerate vertex,
    // which is what makes progress -- and therefore termination -- guaranteed.
    Real theta_max = kInfinity;
    for (Idx i = 0; i < m_; ++i) {
        Real gap = 0.0, absd = 0.0;
        bool toward_lower = false;
        if (!blocker(i, &gap, &absd, &toward_lower)) continue;
        theta_max = std::min(theta_max, (gap + expand_tol_) / absd);
    }

    // The entering variable's own range competes: reaching the opposite bound
    // is a flip that changes no basis.
    const auto e = static_cast<std::size_t>(entering);
    const Real flip_range = (is_finite_bound(lower_[e]) && is_finite_bound(upper_[e]))
                          ? upper_[e] - lower_[e] : kInfinity;

    if (!std::isfinite(theta_max) && !std::isfinite(flip_range))
        return false;                                  // unbounded in this direction

    if (flip_range <= theta_max) {
        *bound_flip = true;
        *step = flip_range;
        return true;
    }

    // Pass 2: among rows whose TRUE (un-relaxed) ratio is within theta_max,
    // pick the leaving row.
    Idx chosen = -1;
    Real chosen_absd = 0.0;
    Real chosen_true_ratio = 0.0;
    for (Idx i = 0; i < m_; ++i) {
        Real gap = 0.0, absd = 0.0;
        bool toward_lower = false;
        if (!blocker(i, &gap, &absd, &toward_lower)) continue;
        const Real true_ratio = gap / absd;
        if (true_ratio > theta_max) continue;

        if (bland_) {
            // Bland: the eligible row whose BASIC VARIABLE has the smallest
            // index. With price()'s lowest-index entering choice, this is what
            // makes termination provable in exact arithmetic; EXPAND covers the
            // inexact case.
            if (chosen < 0 || basis_[static_cast<std::size_t>(i)]
                            < basis_[static_cast<std::size_t>(chosen)]) {
                chosen = i; chosen_absd = absd; chosen_true_ratio = true_ratio;
            }
        } else {
            // Harris: the largest pivot, which keeps the factorization healthy
            // -- this is the safeguard that made the relative pivot floor
            // unnecessary and fixed the singular basis on `blend`.
            if (absd > chosen_absd) {
                chosen = i; chosen_absd = absd; chosen_true_ratio = true_ratio;
            }
        }
    }

    if (chosen < 0) return false;

    // EXPAND step: at least expand_tol_ / |pivot| so it is strictly positive,
    // at most theta_max so no variable leaves the relaxed region. The leaving
    // variable ends up to expand_tol_ past its bound and is snapped back by the
    // caller; the residue is cleaned at the next refactorization.
    Real s = std::max(chosen_true_ratio, expand_tol_ / chosen_absd);
    if (s > theta_max) s = theta_max;

    *leaving_position = chosen;
    *step = s;
    return true;
}

SolveStatus SimplexImpl::iterate(bool phase1, long long* iterations) {
    // Set immediately after a rebuild, cleared by the next successful pivot.
    // It is what stops the recovery path below from looping forever on a
    // failure that is genuine rather than drift-induced.
    bool fresh_factorization = false;

    // Progress-stall net: if the merit function (infeasibility in phase 1, the
    // internal objective in phase 2) does not improve for this many pivots,
    // something is numerically wrong -- refactorize once, and if it still will
    // not move, stop with NumericalFailure rather than spin or, worse, return a
    // wrong Optimal. EXPAND makes a real stall very unlikely, so this only
    // fires on genuine trouble.
    const long long stall_limit =
        std::max<long long>(5000, 40 * (static_cast<long long>(m_) + n_));
    best_merit_ = kInfinity;
    stall_pivots_ = 0;
    bool stall_recovered = false;

    auto merit = [&]() -> Real {
        if (phase1) return primal_infeasibility();
        Real acc = 0.0;
        for (Idx j = 0; j < total_; ++j)
            acc += cost_[static_cast<std::size_t>(j)] * x_[static_cast<std::size_t>(j)];
        return acc;
    };

    while (true) {
        if (opt_.max_iterations > 0 && result_.iterations >= opt_.max_iterations)
            return SolveStatus::IterationLimit;
        if (opt_.time_limit_seconds > 0.0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_).count();
            if (elapsed > opt_.time_limit_seconds) return SolveStatus::TimeLimit;
        }

        if (phase1 && phase1_costs()) return SolveStatus::Optimal;   // feasible: phase done

        // EXPAND: the working feasibility tolerance grows every pivot. This is
        // the monotone quantity that makes the basis sequence non-repeating.
        // Held at zero during the verification endgame, where the ratio test
        // must be exact.
        expand_tol_ = allow_expand_
            ? std::min(expand_tol_ + expand_rate_, opt_.primal_tolerance)
            : 0.0;

        {
            const Real m = merit();
            if (m < best_merit_ - 1e-9 * std::max(Real{1}, std::abs(best_merit_))) {
                best_merit_ = m;
                stall_pivots_ = 0;
                stall_recovered = false;
            } else if (++stall_pivots_ > stall_limit) {
                if (!stall_recovered) {
                    if (refactorize() != Status::Ok) {
                        result_.message = "singular basis while recovering from a progress stall";
                        return SolveStatus::NumericalFailure;
                    }
                    recompute_basic_values();
                    stall_pivots_ = 0;
                    stall_recovered = true;
                } else {
                    result_.message =
                        (phase1 ? "phase 1 " : "phase 2 ")
                        + std::string("made no progress for ") + std::to_string(stall_limit)
                        + " pivots even after a fresh factorization (infeasibility "
                        + std::to_string(primal_infeasibility()) + ")";
                    return SolveStatus::NumericalFailure;
                }
            }
        }

        Real reduced = 0.0, direction = 0.0;
        const Idx entering = price(phase1, &reduced, &direction);
        if (entering < 0) {
            if (phase1 && phase1_cost_frozen_) {
                // The frozen Phase I objective is optimal, which says nothing
                // about the real one. Thaw and re-derive from the current point
                // before concluding anything. If the live gradient turns out to
                // match the frozen one, there genuinely is no improving column
                // and the fall-through below is the right answer.
                const bool unchanged = (frozen_phase1_cost_ == phase1_cost_);
                phase1_cost_frozen_ = false;
                bland_ = opt_.always_bland;
                consecutive_degenerate_ = 0;
                if (!unchanged) continue;
            }
            return SolveStatus::Optimal;                             // no improving column
        }

        Idx leaving_position = -1;
        Real step = 0.0;
        bool bound_flip = false;
        if (!ratio_test(entering, direction, phase1, &leaving_position, &step, &bound_flip)) {
            // Phase I *cannot* be unbounded: its objective is a sum of bound
            // violations and so is bounded below by zero. Finding no blocking
            // variable therefore means the search direction is wrong, and the
            // overwhelmingly likely cause is a stale eta file rather than a
            // logic error -- this is exactly how `brandy` failed, at iteration
            // 274 with a hundred pivots of accumulated drift behind it.
            //
            // So rebuild the factorization from the basis, recompute the basic
            // values from scratch, and try again. Only a second failure, on
            // freshly computed numbers, is a real one.
            //
            // The same guard covers Phase II's unboundedness claim: drift can
            // fake that too, and reporting "unbounded" for a bounded problem is
            // a wrong answer rather than a slow one.
            if (!fresh_factorization) {
                if (refactorize() != Status::Ok) {
                    result_.message = "singular basis while recovering from a ratio-test failure";
                    return SolveStatus::NumericalFailure;
                }
                recompute_basic_values();
                fresh_factorization = true;
                continue;
            }
            if (phase1) {
                result_.message = "phase 1 ratio test found no blocking variable at iteration "
                                + std::to_string(result_.iterations)
                                + ", on a freshly computed factorization";
                return SolveStatus::NumericalFailure;
            }
            return SolveStatus::Unbounded;
        }
        fresh_factorization = false;

        // Degeneracy bookkeeping. A run of zero-length steps is the signature
        // of a stall, and a stall that never breaks is a cycle.
        if (step <= opt_.primal_tolerance) {
            ++consecutive_degenerate_;
            if (!bland_ && !opt_.always_bland
                && consecutive_degenerate_ >= opt_.degenerate_pivots_before_bland) {
                bland_ = true;
                ++result_.bland_switches;
                if (phase1) {
                    frozen_phase1_cost_ = phase1_cost_;
                    phase1_cost_frozen_ = true;
                }
                if (opt_.verbose)
                    std::fprintf(stderr, "  [simplex] degenerate stall after %lld pivots; "
                                         "switching to Bland's rule\n", result_.iterations);
            }
        } else {
            consecutive_degenerate_ = 0;
            if (bland_ && !opt_.always_bland) {
                // Progress resumed, so the expensive guarantee can be dropped
                // again. Staying in Bland's rule for the rest of the solve
                // would cost far more than the stall it resolved.
                bland_ = false;
                phase1_cost_frozen_ = false;
            }
        }

        const Real signed_step = step * direction;
        const auto e = static_cast<std::size_t>(entering);

        // Move the basic variables along the edge.
        for (Idx i = 0; i < m_; ++i) {
            const Real d = direction_[static_cast<std::size_t>(i)];
            if (d == 0.0) continue;
            x_[static_cast<std::size_t>(basis_[static_cast<std::size_t>(i)])] -= d * signed_step;
        }
        x_[e] += signed_step;

        if (bound_flip) {
            state_[e] = (state_[e] == NonbasicState::AtLower) ? NonbasicState::AtUpper
                                                              : NonbasicState::AtLower;
            x_[e] = (state_[e] == NonbasicState::AtLower) ? lower_[e] : upper_[e];
        } else {
            const auto leaving = basis_[static_cast<std::size_t>(leaving_position)];
            const auto l = static_cast<std::size_t>(leaving);

            // Snap the leaving variable exactly onto the bound it reached.
            // Leaving it a rounding error away is how a "feasible" basis
            // quietly stops being feasible over thousands of pivots.
            const Real dist_lo = is_finite_bound(lower_[l]) ? std::abs(x_[l] - lower_[l]) : kInfinity;
            const Real dist_hi = is_finite_bound(upper_[l]) ? std::abs(x_[l] - upper_[l]) : kInfinity;
            if (dist_lo <= dist_hi) { state_[l] = NonbasicState::AtLower; x_[l] = lower_[l]; }
            else                    { state_[l] = NonbasicState::AtUpper; x_[l] = upper_[l]; }

            basis_position_[l] = -1;
            basis_[static_cast<std::size_t>(leaving_position)] = entering;
            basis_position_[e] = leaving_position;

            factor_.push_eta(leaving_position, direction_);
            ++pivots_since_refactor_;

            if (pivots_since_refactor_ >= opt_.refactor_frequency) {
                if (refactorize() != Status::Ok) {
                    result_.message = "refactorization found a singular basis at iteration "
                                    + std::to_string(result_.iterations);
                    return SolveStatus::NumericalFailure;
                }
                recompute_basic_values();
                fresh_factorization = true;
            }
        }

        ++result_.iterations;
        ++(*iterations);
    }
}

SimplexResult SimplexImpl::run() {
    start_ = std::chrono::steady_clock::now();

    if (m_ > kMaxDenseRows) {
        result_.status = SolveStatus::NumericalFailure;
        result_.message = "instance has " + std::to_string(m_) + " rows; this oracle uses a "
                          "dense basis factorization and is limited to "
                        + std::to_string(kMaxDenseRows) + ". Sparse LU is follow-up work.";
        return result_;
    }
    if (problem_.has_crossed_bounds()) {
        result_.status = SolveStatus::Infeasible;
        result_.message = "a bound pair is crossed (lower > upper)";
        return result_;
    }

    setup();
    if (refactorize() != Status::Ok) {
        result_.status = SolveStatus::NumericalFailure;
        result_.message = "the starting basis is singular, which should be impossible";
        return result_;
    }
    recompute_basic_values();

    // -- Phase I -----------------------------------------------------------
    bland_ = opt_.always_bland;
    long long phase1_iterations = 0;
    if (!phase1_costs()) {
        if (opt_.verbose)
            std::fprintf(stderr, "  [simplex] phase 1: infeasibility %.6g\n",
                         primal_infeasibility());
        const SolveStatus s = iterate(true, &phase1_iterations);
        result_.phase1_iterations = phase1_iterations;
        if (s != SolveStatus::Optimal) {
            result_.status = s;
            result_.seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_).count();
            return result_;
        }
        // Refactorize before Phase II: Phase I's pivots are chosen to reduce
        // infeasibility, not for numerical quality, so the eta file at this
        // point is the worst it will be all solve.
        if (refactorize() != Status::Ok) {
            result_.status = SolveStatus::NumericalFailure;
            result_.message = "singular basis at the phase 1 / phase 2 boundary";
            return result_;
        }
        recompute_basic_values();

        if (primal_infeasibility() > 1e-6) {
            result_.status = SolveStatus::Infeasible;
            result_.primal_infeasibility = primal_infeasibility();
            result_.seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_).count();
            return result_;
        }
    }
    result_.phase1_iterations = phase1_iterations;

    // -- Phase II ----------------------------------------------------------
    consecutive_degenerate_ = 0;
    bland_ = opt_.always_bland;
    long long phase2_iterations = 0;
    SolveStatus s = iterate(false, &phase2_iterations);

    // -- EXPAND endgame: verify optimality with the ratio test made exact ----
    //
    // The relaxed ratio test can declare optimality at a point a few 1e-7
    // outside a true bound. Snapping the leaving variable each pivot bounds
    // that, but the residue still accumulates into the reported objective by
    // ~1e-5 on staircase-degenerate instances (scsd1, wood1p). So: switch the
    // relaxation off, rebuild exactly, and if the point is not genuinely
    // feasible-and-optimal, refine it with a bounded burst of exact pivots.
    // The starting point is already almost there, so each burst is short.
    if (s == SolveStatus::Optimal) {
        allow_expand_ = false;
        expand_tol_ = 0.0;
        for (int round = 0; round < 5; ++round) {
            if (refactorize() != Status::Ok) {
                s = SolveStatus::NumericalFailure;
                result_.message = "singular basis during optimality verification";
                break;
            }
            recompute_basic_values();

            if (primal_infeasibility() > opt_.primal_tolerance) {
                long long it = 0;
                consecutive_degenerate_ = 0;
                s = iterate(true, &it);            // pull back to true feasibility
                result_.phase1_iterations += it;
                if (s != SolveStatus::Optimal) break;
                continue;                          // and re-verify
            }

            Real red = 0.0, dir = 0.0;
            if (price(false, &red, &dir) < 0) break;   // no improving column: truly optimal

            long long it = 0;
            consecutive_degenerate_ = 0;
            s = iterate(false, &it);               // exact re-optimization
            if (s != SolveStatus::Optimal) break;
        }
    }

    result_.status = s;
    result_.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_).count();

    if (s == SolveStatus::Optimal || s == SolveStatus::IterationLimit
        || s == SolveStatus::TimeLimit) {
        result_.primal.assign(x_.begin(), x_.begin() + static_cast<std::size_t>(n_));
        result_.objective = problem_.evaluate_objective(result_.primal);
        result_.primal_infeasibility = primal_infeasibility();

        // Row duals, in the internal minimize form; flip back for maximize so
        // the reported signs match the problem the caller handed us.
        for (Idx i = 0; i < m_; ++i)
            basic_costs_[static_cast<std::size_t>(i)] =
                cost_[static_cast<std::size_t>(basis_[static_cast<std::size_t>(i)])];
        duals_ = basic_costs_;
        factor_.btran(duals_);
        result_.dual.resize(static_cast<std::size_t>(m_));
        for (Idx i = 0; i < m_; ++i)
            result_.dual[static_cast<std::size_t>(i)] =
                objective_sign_ * duals_[static_cast<std::size_t>(i)];

        result_.reduced_costs.resize(static_cast<std::size_t>(n_));
        for (Idx j = 0; j < n_; ++j)
            result_.reduced_costs[static_cast<std::size_t>(j)] =
                objective_sign_ * (cost_[static_cast<std::size_t>(j)] - column_dot(j, duals_));
    }
    return result_;
}

}  // namespace

SimplexResult Simplex::solve(const Problem& problem) {
    std::string why;
    if (problem.validate(&why) != Status::Ok)
        throw Error("Simplex::solve was handed an invalid problem: " + why);
    SimplexImpl impl(problem, options_);
    return impl.run();
}

}  // namespace sov
