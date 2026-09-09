// L0 sparse container tests -- Build Map ticket #3.
//
// The reference matrix used throughout, small enough to verify by hand:
//
//        [  2   0  -1 ]
//   A =  [  0   3   4 ]        4 x 3, nnz = 6
//        [  1   0   0 ]
//        [  0   0   5 ]
//
//   x = (1, 2, 3)      =>  A x   = (2-3, 6+12, 1, 15)      = (-1, 18, 1, 15)
//   y = (1, 2, 3, 4)   =>  A^T y = (2+3, 6, -1+8+20)       = (5, 6, 27)
#include <vector>

#include "sovereign/sparse.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

constexpr Real kTol = 1e-12;

CsrMatrix reference_matrix() {
    const std::vector<Triplet> t = {
        {0, 0, 2.0}, {0, 2, -1.0},
        {1, 1, 3.0}, {1, 2, 4.0},
        {2, 0, 1.0},
        {3, 2, 5.0},
    };
    return CsrMatrix::from_triplets(4, 3, t);
}

}  // namespace

TEST(sparse, from_triplets_shape_and_nnz) {
    const CsrMatrix A = reference_matrix();
    CHECK_EQ(A.rows(), 4);
    CHECK_EQ(A.cols(), 3);
    CHECK_EQ(A.nnz(), 6);
    CHECK(A.validate() == Status::Ok);
}

TEST(sparse, from_triplets_sums_duplicates) {
    // MPS permits a COLUMNS section to name the same (row, col) twice; the
    // values add. Getting this wrong changes the problem being solved.
    const std::vector<Triplet> t = {{0, 0, 1.5}, {0, 0, 2.5}, {1, 1, 1.0}};
    const CsrMatrix A = CsrMatrix::from_triplets(2, 2, t);
    CHECK_EQ(A.nnz(), 2);
    CHECK_NEAR(A.values()[0], 4.0, kTol);
}

TEST(sparse, from_triplets_drops_numerical_zeros) {
    const std::vector<Triplet> t = {{0, 0, 1.0}, {0, 1, 0.0}, {1, 1, 1e-20}};
    const CsrMatrix A = CsrMatrix::from_triplets(2, 2, t);
    CHECK_EQ(A.nnz(), 1);
}

TEST(sparse, from_triplets_emits_ascending_column_indices) {
    // Out-of-order input; the compressed form must still be sorted, because
    // the SpMV kernels and the transpose both rely on it.
    const std::vector<Triplet> t = {{0, 2, 3.0}, {0, 0, 1.0}, {0, 1, 2.0}};
    const CsrMatrix A = CsrMatrix::from_triplets(1, 3, t);
    CHECK_EQ(A.col_idx()[0], 0);
    CHECK_EQ(A.col_idx()[1], 1);
    CHECK_EQ(A.col_idx()[2], 2);
    CHECK(A.validate() == Status::Ok);
}

TEST(sparse, from_triplets_rejects_out_of_range) {
    const std::vector<Triplet> t = {{5, 0, 1.0}};
    CHECK_THROWS(CsrMatrix::from_triplets(2, 2, t));
}

TEST(sparse, empty_rows_are_representable) {
    // Row 1 has no entries. A free row in an MPS file looks exactly like this,
    // and an off-by-one in row_ptr shows up here first.
    const std::vector<Triplet> t = {{0, 0, 1.0}, {2, 1, 2.0}};
    const CsrMatrix A = CsrMatrix::from_triplets(3, 2, t);
    CHECK(A.validate() == Status::Ok);
    CHECK_EQ(A.row_ptr()[1], A.row_ptr()[2]);

    std::vector<Real> x = {1.0, 1.0};
    std::vector<Real> y(3, 99.0);
    A.multiply(x, y);
    CHECK_NEAR(y[1], 0.0, kTol);
}

TEST(sparse, multiply_matches_hand_computed) {
    const CsrMatrix A = reference_matrix();
    const std::vector<Real> x = {1.0, 2.0, 3.0};
    std::vector<Real> y(4, 0.0);
    A.multiply(x, y);
    CHECK_NEAR(y[0], -1.0, kTol);
    CHECK_NEAR(y[1], 18.0, kTol);
    CHECK_NEAR(y[2], 1.0, kTol);
    CHECK_NEAR(y[3], 15.0, kTol);
}

TEST(sparse, multiply_transpose_matches_hand_computed) {
    const CsrMatrix A = reference_matrix();
    const std::vector<Real> y = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> out(3, 0.0);
    A.multiply_transpose(y, out);
    CHECK_NEAR(out[0], 5.0, kTol);
    CHECK_NEAR(out[1], 6.0, kTol);
    CHECK_NEAR(out[2], 27.0, kTol);
}

TEST(sparse, multiply_rejects_dimension_mismatch) {
    const CsrMatrix A = reference_matrix();
    const std::vector<Real> x = {1.0, 2.0};   // should be 3
    std::vector<Real> y(4, 0.0);
    CHECK_THROWS(A.multiply(x, y));
}

TEST(sparse, csr_to_csc_roundtrip_preserves_the_matrix) {
    const CsrMatrix A = reference_matrix();
    const CscMatrix C = A.to_csc();
    CHECK(C.validate() == Status::Ok);
    CHECK_EQ(C.nnz(), A.nnz());

    const CsrMatrix back = C.to_csr();
    CHECK(back.validate() == Status::Ok);
    CHECK_EQ(back.nnz(), A.nnz());
    for (Idx k = 0; k < A.nnz(); ++k) {
        CHECK_EQ(back.col_idx()[k], A.col_idx()[k]);
        CHECK_NEAR(back.values()[k], A.values()[k], kTol);
    }
    for (std::size_t i = 0; i < A.row_ptr().size(); ++i)
        CHECK_EQ(back.row_ptr()[i], A.row_ptr()[i]);
}

TEST(sparse, csc_multiply_agrees_with_csr_multiply) {
    // The two layouts reach the same answer by opposite access patterns
    // (gather vs scatter). Disagreement means one of the two transposes is
    // wrong, which is the classic source of a silently wrong dual residual.
    const CsrMatrix A = reference_matrix();
    const CscMatrix C = A.to_csc();
    const std::vector<Real> x = {1.0, 2.0, 3.0};
    std::vector<Real> y_csr(4, 0.0), y_csc(4, 0.0);
    A.multiply(x, y_csr);
    C.multiply(x, y_csc);
    for (std::size_t i = 0; i < y_csr.size(); ++i)
        CHECK_NEAR(y_csr[i], y_csc[i], kTol);
}

TEST(sparse, from_dense_matches_triplet_construction) {
    const std::vector<Real> dense = {
        2.0, 0.0, -1.0,
        0.0, 3.0,  4.0,
        1.0, 0.0,  0.0,
        0.0, 0.0,  5.0,
    };
    const CsrMatrix A = CsrMatrix::from_dense(4, 3, dense);
    const CsrMatrix B = reference_matrix();
    CHECK_EQ(A.nnz(), B.nnz());
    for (Idx k = 0; k < A.nnz(); ++k) {
        CHECK_EQ(A.col_idx()[k], B.col_idx()[k]);
        CHECK_NEAR(A.values()[k], B.values()[k], kTol);
    }
}

TEST(sparse, validate_catches_unsorted_columns) {
    // Hand-built with descending indices in row 0. This is precisely the
    // malformed input that still produces numbers from an SpMV -- just the
    // wrong ones -- so validate() has to reject it.
    const CsrMatrix bad(1, 3, {0, 2}, {2, 0}, {1.0, 2.0});
    std::string why;
    CHECK(bad.validate(&why) != Status::Ok);
    CHECK(!why.empty());
}

TEST(sparse, validate_catches_out_of_range_column) {
    const CsrMatrix bad(1, 2, {0, 1}, {7}, {1.0});
    CHECK(bad.validate() != Status::Ok);
}

TEST(sparse, validate_catches_non_monotone_row_ptr) {
    const CsrMatrix bad(2, 2, {0, 2, 1}, {0, 1}, {1.0, 2.0});
    CHECK(bad.validate() != Status::Ok);
}

TEST(sparse, constructor_rejects_wrong_row_ptr_length) {
    CHECK_THROWS(CsrMatrix(3, 2, {0, 1}, {0}, {1.0}));
}

TEST(sparse, coefficient_extremes_report_the_scaling_spread) {
    const std::vector<Triplet> t = {{0, 0, 1e-6}, {0, 1, 1e6}, {1, 0, -3.0}};
    const CsrMatrix A = CsrMatrix::from_triplets(2, 2, t);
    const auto e = A.coefficient_extremes();
    CHECK_NEAR(e.max_abs, 1e6, 1e-9);
    CHECK_NEAR(e.min_abs_nonzero, 1e-6, 1e-9);
}

TEST(sparse, blocked_layouts_are_declared_but_empty) {
    // VBCSR belongs to ticket #29 and BSR to #32. Present so the interface
    // does not shift under them; empty so nobody mistakes them for done.
    const VbcsrMatrix vb;
    const BsrMatrix bs;
    CHECK(vb.empty());
    CHECK(bs.empty());
}

TST_MAIN()
