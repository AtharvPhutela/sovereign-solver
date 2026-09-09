// L0 sparse container implementation -- Build Map ticket #3.
#include "sovereign/sparse.hpp"

#include <cmath>
#include <map>

namespace sov {
namespace {

bool finite(Real v) noexcept { return std::isfinite(v); }

}  // namespace

// --------------------------------------------------------------------------
// CsrMatrix
// --------------------------------------------------------------------------

CsrMatrix::CsrMatrix(Idx rows, Idx cols, std::vector<Idx> row_ptr,
                     std::vector<Idx> col_idx, std::vector<Real> values)
    : rows_(rows), cols_(cols), row_ptr_(std::move(row_ptr)),
      col_idx_(std::move(col_idx)), values_(std::move(values)) {
    if (rows_ < 0 || cols_ < 0)
        throw Error("CsrMatrix: negative dimensions");
    if (static_cast<Idx>(row_ptr_.size()) != rows_ + 1)
        throw Error("CsrMatrix: row_ptr must have rows+1 entries");
    if (col_idx_.size() != values_.size())
        throw Error("CsrMatrix: col_idx and values length mismatch");
}

CsrMatrix CsrMatrix::from_triplets(Idx rows, Idx cols, std::span<const Triplet> entries) {
    if (rows < 0 || cols < 0) throw Error("from_triplets: negative dimensions");

    // Ordered map keyed by (row, col) gives duplicate summation and ascending
    // column order within a row in one pass. Fine at assembly time; the hot
    // paths never touch it.
    std::map<std::pair<Idx, Idx>, Real> acc;
    for (const Triplet& t : entries) {
        if (t.row < 0 || t.row >= rows || t.col < 0 || t.col >= cols)
            throw Error("from_triplets: entry out of range");
        if (!finite(t.value))
            throw Error("from_triplets: non-finite coefficient");
        acc[{t.row, t.col}] += t.value;   // MPS permits repeated (row, col)
    }

    std::vector<Idx> row_ptr(static_cast<std::size_t>(rows) + 1, 0);
    std::vector<Idx> col_idx;
    std::vector<Real> values;
    col_idx.reserve(acc.size());
    values.reserve(acc.size());

    Idx current_row = 0;
    for (const auto& [rc, v] : acc) {
        // Drop numerical zeros: a structural nonzero worth 0 costs bandwidth
        // in a memory-bound SpMV and buys nothing.
        if (std::abs(v) <= tol::numeric_zero) continue;
        while (current_row < rc.first) {
            row_ptr[static_cast<std::size_t>(current_row) + 1] =
                static_cast<Idx>(values.size());
            ++current_row;
        }
        col_idx.push_back(rc.second);
        values.push_back(v);
    }
    while (current_row < rows) {
        row_ptr[static_cast<std::size_t>(current_row) + 1] = static_cast<Idx>(values.size());
        ++current_row;
    }

    return CsrMatrix(rows, cols, std::move(row_ptr), std::move(col_idx), std::move(values));
}

CsrMatrix CsrMatrix::from_dense(Idx rows, Idx cols, std::span<const Real> row_major,
                                Real drop_tol) {
    if (static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) != row_major.size())
        throw Error("from_dense: data size does not match dimensions");

    std::vector<Idx> row_ptr(static_cast<std::size_t>(rows) + 1, 0);
    std::vector<Idx> col_idx;
    std::vector<Real> values;

    for (Idx r = 0; r < rows; ++r) {
        for (Idx c = 0; c < cols; ++c) {
            const Real v = row_major[static_cast<std::size_t>(r) * cols + c];
            if (std::abs(v) > drop_tol) {
                col_idx.push_back(c);
                values.push_back(v);
            }
        }
        row_ptr[static_cast<std::size_t>(r) + 1] = static_cast<Idx>(values.size());
    }
    return CsrMatrix(rows, cols, std::move(row_ptr), std::move(col_idx), std::move(values));
}

Status CsrMatrix::validate(std::string* why) const {
    auto fail = [&](Status s, const char* msg) {
        if (why) *why = msg;
        return s;
    };
    if (static_cast<Idx>(row_ptr_.size()) != rows_ + 1)
        return fail(Status::InvalidArgument, "row_ptr size != rows+1");
    if (col_idx_.size() != values_.size())
        return fail(Status::InvalidArgument, "col_idx/values length mismatch");
    if (!row_ptr_.empty() && row_ptr_.front() != 0)
        return fail(Status::InvalidArgument, "row_ptr does not start at 0");
    if (!row_ptr_.empty() && row_ptr_.back() != static_cast<Idx>(values_.size()))
        return fail(Status::InvalidArgument, "row_ptr does not end at nnz");

    for (Idx r = 0; r < rows_; ++r) {
        const Idx begin = row_ptr_[static_cast<std::size_t>(r)];
        const Idx end = row_ptr_[static_cast<std::size_t>(r) + 1];
        if (begin > end)
            return fail(Status::InvalidArgument, "row_ptr is not monotone");
        for (Idx k = begin; k < end; ++k) {
            const Idx c = col_idx_[static_cast<std::size_t>(k)];
            if (c < 0 || c >= cols_)
                return fail(Status::InvalidArgument, "column index out of range");
            // Ascending order is not cosmetic: the SpMV kernels and the
            // CSR->CSC transpose both assume it, and an unsorted row produces
            // wrong-but-plausible numbers rather than an error.
            if (k > begin && col_idx_[static_cast<std::size_t>(k) - 1] >= c)
                return fail(Status::InvalidArgument, "column indices not strictly ascending");
            if (!finite(values_[static_cast<std::size_t>(k)]))
                return fail(Status::NumericalFailure, "non-finite coefficient");
        }
    }
    if (why) why->clear();
    return Status::Ok;
}

void CsrMatrix::multiply(std::span<const Real> x, std::span<Real> y) const {
    if (static_cast<Idx>(x.size()) != cols_ || static_cast<Idx>(y.size()) != rows_)
        throw Error(Status::DimensionMismatch, "CsrMatrix::multiply");
    for (Idx r = 0; r < rows_; ++r) {
        Real sum = 0.0;
        const Idx end = row_ptr_[static_cast<std::size_t>(r) + 1];
        for (Idx k = row_ptr_[static_cast<std::size_t>(r)]; k < end; ++k)
            sum += values_[static_cast<std::size_t>(k)]
                 * x[static_cast<std::size_t>(col_idx_[static_cast<std::size_t>(k)])];
        y[static_cast<std::size_t>(r)] = sum;
    }
}

void CsrMatrix::multiply_transpose(std::span<const Real> x, std::span<Real> y) const {
    if (static_cast<Idx>(x.size()) != rows_ || static_cast<Idx>(y.size()) != cols_)
        throw Error(Status::DimensionMismatch, "CsrMatrix::multiply_transpose");
    std::fill(y.begin(), y.end(), Real{0});
    for (Idx r = 0; r < rows_; ++r) {
        const Real xr = x[static_cast<std::size_t>(r)];
        const Idx end = row_ptr_[static_cast<std::size_t>(r) + 1];
        for (Idx k = row_ptr_[static_cast<std::size_t>(r)]; k < end; ++k)
            y[static_cast<std::size_t>(col_idx_[static_cast<std::size_t>(k)])]
                += values_[static_cast<std::size_t>(k)] * xr;
    }
}

CscMatrix CsrMatrix::to_csc() const {
    // Counting sort by column: O(nnz + cols), and it emits row indices in
    // ascending order within each column for free because we sweep rows in
    // order.
    std::vector<Idx> col_ptr(static_cast<std::size_t>(cols_) + 1, 0);
    for (Idx k = 0; k < nnz(); ++k)
        ++col_ptr[static_cast<std::size_t>(col_idx_[static_cast<std::size_t>(k)]) + 1];
    for (Idx c = 0; c < cols_; ++c)
        col_ptr[static_cast<std::size_t>(c) + 1] += col_ptr[static_cast<std::size_t>(c)];

    std::vector<Idx> row_idx(static_cast<std::size_t>(nnz()));
    std::vector<Real> vals(static_cast<std::size_t>(nnz()));
    std::vector<Idx> cursor(col_ptr.begin(), col_ptr.end() - 1);

    for (Idx r = 0; r < rows_; ++r) {
        const Idx end = row_ptr_[static_cast<std::size_t>(r) + 1];
        for (Idx k = row_ptr_[static_cast<std::size_t>(r)]; k < end; ++k) {
            const Idx c = col_idx_[static_cast<std::size_t>(k)];
            const Idx dest = cursor[static_cast<std::size_t>(c)]++;
            row_idx[static_cast<std::size_t>(dest)] = r;
            vals[static_cast<std::size_t>(dest)] = values_[static_cast<std::size_t>(k)];
        }
    }
    return CscMatrix(rows_, cols_, std::move(col_ptr), std::move(row_idx), std::move(vals));
}

CsrMatrix::Extremes CsrMatrix::coefficient_extremes() const noexcept {
    Extremes e;
    bool first = true;
    for (Real v : values_) {
        const Real a = std::abs(v);
        if (a > e.max_abs) e.max_abs = a;
        if (a > tol::numeric_zero && (first || a < e.min_abs_nonzero)) {
            e.min_abs_nonzero = a;
            first = false;
        }
    }
    return e;
}

// --------------------------------------------------------------------------
// CscMatrix
// --------------------------------------------------------------------------

CscMatrix::CscMatrix(Idx rows, Idx cols, std::vector<Idx> col_ptr,
                     std::vector<Idx> row_idx, std::vector<Real> values)
    : rows_(rows), cols_(cols), col_ptr_(std::move(col_ptr)),
      row_idx_(std::move(row_idx)), values_(std::move(values)) {
    if (static_cast<Idx>(col_ptr_.size()) != cols_ + 1)
        throw Error("CscMatrix: col_ptr must have cols+1 entries");
    if (row_idx_.size() != values_.size())
        throw Error("CscMatrix: row_idx and values length mismatch");
}

Status CscMatrix::validate(std::string* why) const {
    auto fail = [&](Status s, const char* msg) {
        if (why) *why = msg;
        return s;
    };
    if (static_cast<Idx>(col_ptr_.size()) != cols_ + 1)
        return fail(Status::InvalidArgument, "col_ptr size != cols+1");
    if (row_idx_.size() != values_.size())
        return fail(Status::InvalidArgument, "row_idx/values length mismatch");
    if (!col_ptr_.empty() && col_ptr_.back() != static_cast<Idx>(values_.size()))
        return fail(Status::InvalidArgument, "col_ptr does not end at nnz");

    for (Idx c = 0; c < cols_; ++c) {
        const Idx begin = col_ptr_[static_cast<std::size_t>(c)];
        const Idx end = col_ptr_[static_cast<std::size_t>(c) + 1];
        if (begin > end) return fail(Status::InvalidArgument, "col_ptr is not monotone");
        for (Idx k = begin; k < end; ++k) {
            const Idx r = row_idx_[static_cast<std::size_t>(k)];
            if (r < 0 || r >= rows_)
                return fail(Status::InvalidArgument, "row index out of range");
            if (k > begin && row_idx_[static_cast<std::size_t>(k) - 1] >= r)
                return fail(Status::InvalidArgument, "row indices not strictly ascending");
            if (!finite(values_[static_cast<std::size_t>(k)]))
                return fail(Status::NumericalFailure, "non-finite coefficient");
        }
    }
    if (why) why->clear();
    return Status::Ok;
}

void CscMatrix::multiply(std::span<const Real> x, std::span<Real> y) const {
    if (static_cast<Idx>(x.size()) != cols_ || static_cast<Idx>(y.size()) != rows_)
        throw Error(Status::DimensionMismatch, "CscMatrix::multiply");
    std::fill(y.begin(), y.end(), Real{0});
    for (Idx c = 0; c < cols_; ++c) {
        const Real xc = x[static_cast<std::size_t>(c)];
        const Idx end = col_ptr_[static_cast<std::size_t>(c) + 1];
        for (Idx k = col_ptr_[static_cast<std::size_t>(c)]; k < end; ++k)
            y[static_cast<std::size_t>(row_idx_[static_cast<std::size_t>(k)])]
                += values_[static_cast<std::size_t>(k)] * xc;
    }
}

CsrMatrix CscMatrix::to_csr() const {
    std::vector<Idx> row_ptr(static_cast<std::size_t>(rows_) + 1, 0);
    for (Idx k = 0; k < nnz(); ++k)
        ++row_ptr[static_cast<std::size_t>(row_idx_[static_cast<std::size_t>(k)]) + 1];
    for (Idx r = 0; r < rows_; ++r)
        row_ptr[static_cast<std::size_t>(r) + 1] += row_ptr[static_cast<std::size_t>(r)];

    std::vector<Idx> col_idx(static_cast<std::size_t>(nnz()));
    std::vector<Real> vals(static_cast<std::size_t>(nnz()));
    std::vector<Idx> cursor(row_ptr.begin(), row_ptr.end() - 1);

    for (Idx c = 0; c < cols_; ++c) {
        const Idx end = col_ptr_[static_cast<std::size_t>(c) + 1];
        for (Idx k = col_ptr_[static_cast<std::size_t>(c)]; k < end; ++k) {
            const Idx r = row_idx_[static_cast<std::size_t>(k)];
            const Idx dest = cursor[static_cast<std::size_t>(r)]++;
            col_idx[static_cast<std::size_t>(dest)] = c;
            vals[static_cast<std::size_t>(dest)] = values_[static_cast<std::size_t>(k)];
        }
    }
    return CsrMatrix(rows_, cols_, std::move(row_ptr), std::move(col_idx), std::move(vals));
}

}  // namespace sov
