// Preconditioning / equilibration -- Build Map ticket #6, gate M0. Bible S4.1, S6.2.
//
// PURPOSE. The problem statement grades on ill-conditioned constraint matrices,
// and Bible S6.2 is blunt: without this front-end the first-order engine
// (PDHG, #8) "stalls for thousands of iterations on industrial matrices". So
// scaling is mandatory, not optional polish.
//
// TWO SCHEMES, applied in sequence:
//
//   Ruiz equilibration -- iteratively rescale rows and columns so that the
//   infinity-norm of every row and every column tends to 1. Kills the gross
//   spread between a 1e-6 coefficient and a 1e+6 one.
//
//   Pock-Chambolle diagonal preconditioner -- the step-size scaling the PDHG
//   iteration itself needs: row weights ~ 1/sum_j|A_ij|^(2-a), column weights
//   ~ 1/sum_i|A_ij|^a. Computed on the Ruiz-equilibrated matrix.
//
// SUBSTRATE. The arithmetic here is row/column reductions and an elementwise
// divide -- exactly the L0 backend primitives. This implementation runs on the
// host CsrMatrix (the host IS a backend); the device version is the same
// formulas through the backend's SpMV/reduction calls, and is wired in when a
// GPU build exists. Nothing about the method is CPU-specific.
//
// WHAT SCALING MUST PRESERVE. A scaled solve returns a scaled answer. The
// Scaling object carries the row and column factors so a caller can map a
// solution of the scaled problem back to the original and vice versa; the
// objective value is invariant under the mapping (see unscale_primal).
#pragma once

#include <string>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/problem.hpp"
#include "sovereign/sparse.hpp"

namespace sov {

struct ScalingOptions {
    /// Ruiz sweeps. Convergence is geometric; 5-10 is the usual range and the
    /// marginal gain past ~8 is negligible on real matrices.
    int ruiz_iterations = 8;

    /// Stop early once every row and column infinity-norm is within this of 1.
    Real ruiz_tolerance = 1e-2;

    /// Pock-Chambolle exponent. 1 is the standard choice and splits the work
    /// evenly between the primal and dual step sizes.
    Real pock_chambolle_alpha = 1.0;

    bool apply_ruiz = true;
    bool apply_pock_chambolle = true;
};

/// Diagonal row and column scaling: the scaled matrix is
/// A_scaled = diag(row_scale) * A * diag(col_scale).
class Scaling {
public:
    Scaling() = default;

    /// Compute scaling factors for `matrix`. Does not modify anything; call
    /// apply() to get the scaled matrix.
    static Scaling equilibrate(const CsrMatrix& matrix, const ScalingOptions& options = {});

    /// Compute scaling for a whole problem. Column scaling also rescales the
    /// objective coefficients and the column bounds; row scaling rescales the
    /// row bounds. The returned Problem is mathematically equivalent, better
    /// conditioned, and solved in place of the original.
    static std::pair<Problem, Scaling> equilibrate(const Problem& problem,
                                                   const ScalingOptions& options = {});

    Idx num_rows() const noexcept { return static_cast<Idx>(row_scale_.size()); }
    Idx num_cols() const noexcept { return static_cast<Idx>(col_scale_.size()); }

    std::span<const Real> row_scale() const noexcept { return row_scale_; }
    std::span<const Real> col_scale() const noexcept { return col_scale_; }

    /// A_scaled = diag(row_scale) * A * diag(col_scale).
    CsrMatrix apply(const CsrMatrix& matrix) const;

    /// x_original[j] = x_scaled[j] * col_scale[j]. The scaled problem's
    /// variables are y = x / col_scale, so recovering x multiplies back.
    void unscale_primal(std::span<Real> x_in_scaled_space) const;

    /// y_original[i] = y_scaled[i] * row_scale[i] for the row duals.
    void unscale_dual(std::span<Real> y_in_scaled_space) const;

    // --- conditioning read-outs, for the ticket's "measurably improves" test -
    struct Conditioning {
        Real max_abs = 0.0;           ///< largest |a_ij|
        Real min_abs_nonzero = 0.0;   ///< smallest nonzero |a_ij|
        Real coeff_spread = 0.0;      ///< max_abs / min_abs_nonzero
        Real row_norm_spread = 0.0;   ///< max row inf-norm / min row inf-norm
        Real col_norm_spread = 0.0;   ///< likewise for columns
    };
    static Conditioning measure(const CsrMatrix& matrix);

    std::string summary() const;

private:
    std::vector<Real> row_scale_;
    std::vector<Real> col_scale_;
};

}  // namespace sov
