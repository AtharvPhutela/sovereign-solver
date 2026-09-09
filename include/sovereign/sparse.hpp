// L0 sparse containers -- Build Map ticket #3, gate M0. Bible S4.1.
//
// Host-side storage for the constraint matrix in the two layouts the whole
// solver is built around: CSR (row access -- Ax, the primal residual) and CSC
// (column access -- A^T y, the dual residual). PDHG's inner loop needs both
// every iteration (Bible S4.2 Engine A), so both are first-class rather than
// one being derived on demand.
//
// PRECISION MODEL: the host matrix is the authoritative copy and is always
// fp64. Narrower precisions belong to the *device* copy, where the
// mixed-precision iterative-refinement path of Bible S6.3 will ask for an f32
// or f16 image of an fp64 truth. Keeping the host side single-precision-free
// means there is exactly one place the exact coefficients live.
#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/status.hpp"

namespace sov {

/// One (row, column, value) entry, the assembly-time form.
struct Triplet {
    Idx row = 0;
    Idx col = 0;
    Real value = 0.0;
};

// --------------------------------------------------------------------------
// Compressed Sparse Row
// --------------------------------------------------------------------------

class CscMatrix;

class CsrMatrix {
public:
    CsrMatrix() = default;

    /// Take ownership of already-compressed arrays. `row_ptr` has rows+1
    /// entries; column indices within a row must be ascending.
    CsrMatrix(Idx rows, Idx cols, std::vector<Idx> row_ptr,
              std::vector<Idx> col_idx, std::vector<Real> values);

    /// Assemble from an unordered triplet list.
    ///
    /// Duplicates are summed, which is what MPS actually requires: a COLUMNS
    /// section may name the same (row, column) pair twice and the values add.
    /// Explicit zeros are dropped -- a structural nonzero that is numerically
    /// zero only costs bandwidth in a memory-bound SpMV.
    static CsrMatrix from_triplets(Idx rows, Idx cols, std::span<const Triplet> entries);

    /// Dense-major construction, for tests and small hand-built systems.
    static CsrMatrix from_dense(Idx rows, Idx cols, std::span<const Real> row_major,
                                Real drop_tol = tol::numeric_zero);

    Idx rows() const noexcept { return rows_; }
    Idx cols() const noexcept { return cols_; }
    Idx nnz() const noexcept { return static_cast<Idx>(values_.size()); }

    std::span<const Idx> row_ptr() const noexcept { return row_ptr_; }
    std::span<const Idx> col_idx() const noexcept { return col_idx_; }
    std::span<const Real> values() const noexcept { return values_; }

    /// Bytes the three arrays occupy -- what an upload actually costs.
    std::size_t byte_size() const noexcept {
        return row_ptr_.size() * sizeof(Idx) + col_idx_.size() * sizeof(Idx)
             + values_.size() * sizeof(Real);
    }

    /// Structural self-check: monotone row pointers, in-range and ascending
    /// column indices, finite values. Returns a description on failure.
    ///
    /// This is not paranoia. A matrix that is subtly malformed -- unsorted
    /// indices, a row pointer off by one -- still produces numbers from an
    /// SpMV. They are just the wrong numbers, and the error surfaces three
    /// layers up as "the solver disagrees with the oracle".
    Status validate(std::string* why = nullptr) const;

    /// y := A x  (reference implementation; also the Host backend's kernel)
    void multiply(std::span<const Real> x, std::span<Real> y) const;

    /// y := A^T x  -- the other half of PDHG's iteration.
    void multiply_transpose(std::span<const Real> x, std::span<Real> y) const;

    CscMatrix to_csc() const;

    /// Largest |a_ij|, and the smallest nonzero magnitude. The ratio is a
    /// first, cheap read on how badly the instance needs the Ruiz/
    /// Pock-Chambolle scaling of ticket #6 (Bible S6.2).
    struct Extremes { Real max_abs = 0.0; Real min_abs_nonzero = 0.0; };
    Extremes coefficient_extremes() const noexcept;

private:
    Idx rows_ = 0;
    Idx cols_ = 0;
    std::vector<Idx> row_ptr_{0};
    std::vector<Idx> col_idx_;
    std::vector<Real> values_;
};

// --------------------------------------------------------------------------
// Compressed Sparse Column
// --------------------------------------------------------------------------

class CscMatrix {
public:
    CscMatrix() = default;
    CscMatrix(Idx rows, Idx cols, std::vector<Idx> col_ptr,
              std::vector<Idx> row_idx, std::vector<Real> values);

    Idx rows() const noexcept { return rows_; }
    Idx cols() const noexcept { return cols_; }
    Idx nnz() const noexcept { return static_cast<Idx>(values_.size()); }

    std::span<const Idx> col_ptr() const noexcept { return col_ptr_; }
    std::span<const Idx> row_idx() const noexcept { return row_idx_; }
    std::span<const Real> values() const noexcept { return values_; }

    Status validate(std::string* why = nullptr) const;

    /// y := A x, walking columns (scatter rather than gather).
    void multiply(std::span<const Real> x, std::span<Real> y) const;

    CsrMatrix to_csr() const;

private:
    Idx rows_ = 0;
    Idx cols_ = 0;
    std::vector<Idx> col_ptr_{0};
    std::vector<Idx> row_idx_;
    std::vector<Real> values_;
};

// --------------------------------------------------------------------------
// Blocked layouts -- DECLARED, NOT IMPLEMENTED
//
// Bible S4.1 requires these before the decomposition layer can batch
// subproblems without warp divergence. They are named here so the layer that
// needs them is not the layer that invents them, and so the backend interface
// can carry their signatures from the start.
// --------------------------------------------------------------------------

/// Variable Block CSR -- heterogeneous block sizes.
///
/// Needed by Dantzig-Wolfe (ticket #29): pricing subproblems have different
/// shapes per block, and padding them into uniform blocks is what keeps warp
/// execution uniform across a batch (Bible S4.2A.2). Filling this in is part
/// of ticket #29, not #3.
struct VbcsrMatrix {
    Idx rows = 0;
    Idx cols = 0;
    std::vector<Idx> block_row_ptr;    ///< per block-row, into block_col_idx
    std::vector<Idx> block_col_idx;    ///< block column of each block
    std::vector<Idx> row_block_offset; ///< scalar row where each block-row starts
    std::vector<Idx> col_block_offset; ///< scalar col where each block-col starts
    std::vector<Real> values;          ///< blocks, concatenated row-major

    bool empty() const noexcept { return values.empty(); }
};

/// Block Sparse Row -- uniform block size.
///
/// Needed by Progressive Hedging (ticket #32) for scenario bundles, where the
/// bundle is the natural block and uniformity balances load across SMs
/// (Bible S4.2A.3). Filling this in is part of ticket #32.
struct BsrMatrix {
    Idx block_rows = 0;
    Idx block_cols = 0;
    Idx block_dim = 0;                 ///< square blocks of block_dim x block_dim
    std::vector<Idx> block_row_ptr;
    std::vector<Idx> block_col_idx;
    std::vector<Real> values;

    bool empty() const noexcept { return values.empty(); }
};

}  // namespace sov
