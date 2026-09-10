// The optimization problem, as the solver sees it -- Build Map ticket #5.
//
// One canonical internal form, reached by every reader:
//
//     optimize   c^T x + d          (minimize or maximize)
//     subject to row_lo <= A x <= row_hi
//                col_lo <=  x  <= col_hi
//                x_j integral for j in the integer set
//
// TWO-SIDED ROW BOUNDS ARE THE POINT. MPS says a row is <=, >=, or =, and then
// the RANGES section turns any of them two-sided. Carrying (lo, hi) per row
// makes RANGES a natural assignment instead of a special case, and it is the
// form the presolve (#13) and both continuous engines want anyway. Infinite
// bounds mark the one-sided cases, so nothing is lost.
#pragma once

#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/sparse.hpp"
#include "sovereign/status.hpp"

namespace sov {

inline constexpr Real kInfinity = std::numeric_limits<Real>::infinity();

inline bool is_finite_bound(Real v) noexcept {
    return v > -kInfinity && v < kInfinity;
}

enum class ObjSense { Minimize, Maximize };

constexpr const char* to_string(ObjSense s) noexcept {
    return s == ObjSense::Minimize ? "minimize" : "maximize";
}

enum class VarKind : std::uint8_t { Continuous, Integer };

/// Row classification, kept for reporting and for matching what an oracle
/// says about a file. The solver itself only reads the bounds.
enum class RowType : std::uint8_t { Free, LessEqual, GreaterEqual, Equality, Range };

class Problem {
public:
    Problem() = default;

    // -- shape -----------------------------------------------------------
    Idx num_rows() const noexcept { return matrix_.rows(); }
    Idx num_cols() const noexcept { return matrix_.cols(); }
    Idx num_nonzeros() const noexcept { return matrix_.nnz(); }

    const CsrMatrix& matrix() const noexcept { return matrix_; }

    // -- objective -------------------------------------------------------
    ObjSense sense() const noexcept { return sense_; }
    std::span<const Real> objective() const noexcept { return objective_; }

    /// Additive constant. MPS expresses it as an RHS entry on the objective
    /// row, negated -- see the reader.
    Real objective_constant() const noexcept { return objective_constant_; }

    /// c^T x + d, in the problem's own sense.
    Real evaluate_objective(std::span<const Real> x) const;

    // -- bounds ----------------------------------------------------------
    std::span<const Real> row_lower() const noexcept { return row_lower_; }
    std::span<const Real> row_upper() const noexcept { return row_upper_; }
    std::span<const Real> col_lower() const noexcept { return col_lower_; }
    std::span<const Real> col_upper() const noexcept { return col_upper_; }

    RowType row_type(Idx i) const noexcept;

    // -- integrality -----------------------------------------------------
    std::span<const VarKind> col_kind() const noexcept { return col_kind_; }
    Idx num_integer_columns() const noexcept;
    bool is_mip() const noexcept { return num_integer_columns() > 0; }

    // -- names -----------------------------------------------------------
    const std::string& name() const noexcept { return name_; }
    const std::string& objective_name() const noexcept { return objective_name_; }
    std::span<const std::string> row_names() const noexcept { return row_names_; }
    std::span<const std::string> col_names() const noexcept { return col_names_; }

    /// Free rows present in the source file beyond the objective row. Kept as
    /// rows with infinite bounds so row counts match what a reader of the same
    /// file reports; recorded separately because they carry no constraint.
    Idx num_free_rows() const noexcept;

    // -- checks ----------------------------------------------------------
    /// Structural consistency: array lengths, lo <= hi everywhere, finite
    /// objective coefficients, and a valid matrix.
    Status validate(std::string* why = nullptr) const;

    /// True when some bound pair is crossed (lo > hi), which makes the problem
    /// trivially infeasible before any algorithm runs.
    bool has_crossed_bounds() const noexcept;

    /// A one-line summary for logs and for diffing against an oracle's report.
    std::string summary() const;

    // -- construction ----------------------------------------------------
    // Readers fill a Builder and hand it over; Problem itself stays immutable
    // once built, because a model that can be edited after validation is a
    // model whose validation means nothing.
    class Builder;

private:
    friend class Builder;

    std::string name_;
    std::string objective_name_;
    ObjSense sense_ = ObjSense::Minimize;

    CsrMatrix matrix_;
    std::vector<Real> objective_;
    Real objective_constant_ = 0.0;

    std::vector<Real> row_lower_, row_upper_;
    std::vector<Real> col_lower_, col_upper_;
    std::vector<VarKind> col_kind_;

    std::vector<std::string> row_names_, col_names_;
};

/// Incremental assembly. Readers add rows and columns by name, then finish().
class Problem::Builder {
public:
    void set_name(std::string n) { name_ = std::move(n); }
    void set_sense(ObjSense s) { sense_ = s; }
    void set_objective_name(std::string n) { objective_name_ = std::move(n); }
    void set_objective_constant(Real d) { objective_constant_ = d; }

    /// Register a row. Returns its index; a repeated name is an error.
    Idx add_row(std::string name, Real lower, Real upper);

    /// Register a column with default bounds [0, +inf) and continuous kind.
    Idx add_column(std::string name);

    Idx find_row(const std::string& name) const;      ///< -1 if absent
    Idx find_column(const std::string& name) const;   ///< -1 if absent

    void set_row_bounds(Idx row, Real lower, Real upper);
    void set_column_bounds(Idx col, Real lower, Real upper);
    void set_column_kind(Idx col, VarKind kind);
    void set_objective_coefficient(Idx col, Real value);

    /// Accumulates: a repeated (row, col) pair sums, which is what MPS means.
    void add_coefficient(Idx row, Idx col, Real value);

    Real column_lower(Idx col) const { return col_lower_.at(static_cast<std::size_t>(col)); }
    Real column_upper(Idx col) const { return col_upper_.at(static_cast<std::size_t>(col)); }
    VarKind column_kind(Idx col) const { return col_kind_.at(static_cast<std::size_t>(col)); }
    Idx num_rows() const noexcept { return static_cast<Idx>(row_names_.size()); }
    Idx num_columns() const noexcept { return static_cast<Idx>(col_names_.size()); }

    Problem finish();

private:
    std::string name_, objective_name_;
    ObjSense sense_ = ObjSense::Minimize;
    Real objective_constant_ = 0.0;

    std::vector<std::string> row_names_, col_names_;
    std::vector<Real> row_lower_, row_upper_;
    std::vector<Real> col_lower_, col_upper_;
    std::vector<VarKind> col_kind_;
    std::vector<Real> objective_;
    std::vector<Triplet> entries_;

    // Name lookup. MPS is name-addressed throughout -- COLUMNS, RHS, RANGES
    // and BOUNDS all refer to rows and columns by name, and instances carry
    // tens of thousands of them, so a linear scan would dominate parse time.
    std::unordered_map<std::string, Idx> row_index_, col_index_;
};

}  // namespace sov
