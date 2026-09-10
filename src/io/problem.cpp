// Problem model implementation -- Build Map ticket #5.
#include "sovereign/problem.hpp"

#include <cmath>
#include <sstream>

namespace sov {

// --------------------------------------------------------------------------
// Problem
// --------------------------------------------------------------------------

Real Problem::evaluate_objective(std::span<const Real> x) const {
    if (static_cast<Idx>(x.size()) != num_cols())
        throw Error(Status::DimensionMismatch, "evaluate_objective");
    Real acc = objective_constant_;
    for (std::size_t j = 0; j < objective_.size(); ++j) acc += objective_[j] * x[j];
    return acc;
}

RowType Problem::row_type(Idx i) const noexcept {
    const auto k = static_cast<std::size_t>(i);
    const Real lo = row_lower_[k], hi = row_upper_[k];
    const bool lo_fin = is_finite_bound(lo), hi_fin = is_finite_bound(hi);
    if (!lo_fin && !hi_fin) return RowType::Free;
    if (lo_fin && hi_fin) return (lo == hi) ? RowType::Equality : RowType::Range;
    return hi_fin ? RowType::LessEqual : RowType::GreaterEqual;
}

Idx Problem::num_integer_columns() const noexcept {
    Idx n = 0;
    for (VarKind k : col_kind_) n += (k == VarKind::Integer) ? 1 : 0;
    return n;
}

Idx Problem::num_free_rows() const noexcept {
    Idx n = 0;
    for (Idx i = 0; i < num_rows(); ++i) n += (row_type(i) == RowType::Free) ? 1 : 0;
    return n;
}

bool Problem::has_crossed_bounds() const noexcept {
    for (std::size_t i = 0; i < row_lower_.size(); ++i)
        if (row_lower_[i] > row_upper_[i]) return true;
    for (std::size_t j = 0; j < col_lower_.size(); ++j)
        if (col_lower_[j] > col_upper_[j]) return true;
    return false;
}

Status Problem::validate(std::string* why) const {
    auto fail = [&](Status s, const std::string& msg) {
        if (why) *why = msg;
        return s;
    };

    const auto rows = static_cast<std::size_t>(num_rows());
    const auto cols = static_cast<std::size_t>(num_cols());

    if (row_lower_.size() != rows || row_upper_.size() != rows)
        return fail(Status::InvalidArgument, "row bound arrays do not match row count");
    if (row_names_.size() != rows)
        return fail(Status::InvalidArgument, "row name array does not match row count");
    if (col_lower_.size() != cols || col_upper_.size() != cols)
        return fail(Status::InvalidArgument, "column bound arrays do not match column count");
    if (col_kind_.size() != cols || col_names_.size() != cols)
        return fail(Status::InvalidArgument, "column kind/name array does not match column count");
    if (objective_.size() != cols)
        return fail(Status::InvalidArgument, "objective length does not match column count");

    if (!std::isfinite(objective_constant_))
        return fail(Status::NumericalFailure, "objective constant is not finite");
    for (std::size_t j = 0; j < cols; ++j) {
        if (!std::isfinite(objective_[j]))
            return fail(Status::NumericalFailure, "non-finite objective coefficient in column "
                                                  + col_names_[j]);
        if (std::isnan(col_lower_[j]) || std::isnan(col_upper_[j]))
            return fail(Status::NumericalFailure, "NaN bound on column " + col_names_[j]);
        if (col_lower_[j] > col_upper_[j])
            return fail(Status::InvalidArgument, "crossed bounds on column " + col_names_[j]);
    }
    for (std::size_t i = 0; i < rows; ++i) {
        if (std::isnan(row_lower_[i]) || std::isnan(row_upper_[i]))
            return fail(Status::NumericalFailure, "NaN bound on row " + row_names_[i]);
        if (row_lower_[i] > row_upper_[i])
            return fail(Status::InvalidArgument, "crossed bounds on row " + row_names_[i]);
    }

    // An integer variable with fractional bounds is legal but almost always a
    // parse error, so it is worth surfacing rather than silently tightening.
    for (std::size_t j = 0; j < cols; ++j) {
        if (col_kind_[j] != VarKind::Integer) continue;
        for (Real b : {col_lower_[j], col_upper_[j]}) {
            if (is_finite_bound(b) && std::abs(b - std::round(b)) > 1e-9)
                return fail(Status::InvalidArgument,
                            "integer column " + col_names_[j] + " has a fractional bound");
        }
    }

    return matrix_.validate(why);
}

std::string Problem::summary() const {
    std::ostringstream os;
    os << (name_.empty() ? "<unnamed>" : name_) << ": "
       << num_rows() << " rows, " << num_cols() << " cols, "
       << num_nonzeros() << " nonzeros, " << to_string(sense_);
    const Idx ints = num_integer_columns();
    if (ints > 0) os << ", " << ints << " integer";
    const Idx frees = num_free_rows();
    if (frees > 0) os << ", " << frees << " free row(s)";
    if (objective_constant_ != 0.0) os << ", objective constant " << objective_constant_;
    return os.str();
}

// --------------------------------------------------------------------------
// Problem::Builder
// --------------------------------------------------------------------------

Idx Problem::Builder::add_row(std::string name, Real lower, Real upper) {
    const Idx index = static_cast<Idx>(row_names_.size());
    auto [it, inserted] = row_index_.emplace(name, index);
    if (!inserted)
        throw Error("duplicate row name '" + name + "'");
    row_names_.push_back(std::move(name));
    row_lower_.push_back(lower);
    row_upper_.push_back(upper);
    return index;
}

Idx Problem::Builder::add_column(std::string name) {
    const Idx index = static_cast<Idx>(col_names_.size());
    auto [it, inserted] = col_index_.emplace(name, index);
    if (!inserted)
        throw Error("duplicate column name '" + name + "'");
    col_names_.push_back(std::move(name));
    // MPS default: a column is nonnegative and unbounded above unless BOUNDS
    // says otherwise. Getting this default wrong changes every instance that
    // omits a BOUNDS section, which is most of Netlib.
    col_lower_.push_back(0.0);
    col_upper_.push_back(kInfinity);
    col_kind_.push_back(VarKind::Continuous);
    objective_.push_back(0.0);
    return index;
}

Idx Problem::Builder::find_row(const std::string& name) const {
    const auto it = row_index_.find(name);
    return it == row_index_.end() ? Idx{-1} : it->second;
}

Idx Problem::Builder::find_column(const std::string& name) const {
    const auto it = col_index_.find(name);
    return it == col_index_.end() ? Idx{-1} : it->second;
}

void Problem::Builder::set_row_bounds(Idx row, Real lower, Real upper) {
    row_lower_.at(static_cast<std::size_t>(row)) = lower;
    row_upper_.at(static_cast<std::size_t>(row)) = upper;
}

void Problem::Builder::set_column_bounds(Idx col, Real lower, Real upper) {
    col_lower_.at(static_cast<std::size_t>(col)) = lower;
    col_upper_.at(static_cast<std::size_t>(col)) = upper;
}

void Problem::Builder::set_column_kind(Idx col, VarKind kind) {
    col_kind_.at(static_cast<std::size_t>(col)) = kind;
}

void Problem::Builder::set_objective_coefficient(Idx col, Real value) {
    objective_.at(static_cast<std::size_t>(col)) = value;
}

void Problem::Builder::add_coefficient(Idx row, Idx col, Real value) {
    entries_.push_back(Triplet{row, col, value});
}

Problem Problem::Builder::finish() {
    Problem p;
    p.name_ = std::move(name_);
    p.objective_name_ = std::move(objective_name_);
    p.sense_ = sense_;
    p.objective_constant_ = objective_constant_;
    p.objective_ = std::move(objective_);
    p.row_lower_ = std::move(row_lower_);
    p.row_upper_ = std::move(row_upper_);
    p.col_lower_ = std::move(col_lower_);
    p.col_upper_ = std::move(col_upper_);
    p.col_kind_ = std::move(col_kind_);
    p.row_names_ = std::move(row_names_);
    p.col_names_ = std::move(col_names_);
    p.matrix_ = CsrMatrix::from_triplets(static_cast<Idx>(p.row_names_.size()),
                                         static_cast<Idx>(p.col_names_.size()),
                                         entries_);
    entries_.clear();
    return p;
}

}  // namespace sov
