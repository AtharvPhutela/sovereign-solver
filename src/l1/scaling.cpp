// Preconditioning / equilibration implementation -- Build Map ticket #6.
#include "sovereign/scaling.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace sov {
namespace {

Real safe_sqrt(Real v) { return v > 0.0 ? std::sqrt(v) : 1.0; }

}  // namespace

// --------------------------------------------------------------------------
// Scaling::equilibrate (matrix)
// --------------------------------------------------------------------------

Scaling Scaling::equilibrate(const CsrMatrix& matrix, const ScalingOptions& options) {
    const auto rows = static_cast<std::size_t>(matrix.rows());
    const auto cols = static_cast<std::size_t>(matrix.cols());

    Scaling s;
    s.row_scale_.assign(rows, 1.0);
    s.col_scale_.assign(cols, 1.0);

    const auto row_ptr = matrix.row_ptr();
    const auto col_idx = matrix.col_idx();
    const auto values = matrix.values();

    // Working copy of the current scaled magnitudes is avoided; instead each
    // sweep recomputes row/column extremes from the original values times the
    // running scale factors. O(nnz) per sweep, a handful of sweeps.

    // -- Ruiz equilibration ---------------------------------------------------
    if (options.apply_ruiz) {
        std::vector<Real> row_max(rows, 0.0);
        std::vector<Real> col_max(cols, 0.0);

        for (int sweep = 0; sweep < options.ruiz_iterations; ++sweep) {
            std::fill(row_max.begin(), row_max.end(), Real{0});
            std::fill(col_max.begin(), col_max.end(), Real{0});

            for (Idx r = 0; r < matrix.rows(); ++r) {
                const Idx end = row_ptr[static_cast<std::size_t>(r) + 1];
                for (Idx k = row_ptr[static_cast<std::size_t>(r)]; k < end; ++k) {
                    const auto c = static_cast<std::size_t>(col_idx[static_cast<std::size_t>(k)]);
                    const Real scaled = std::abs(values[static_cast<std::size_t>(k)])
                                      * s.row_scale_[static_cast<std::size_t>(r)]
                                      * s.col_scale_[c];
                    row_max[static_cast<std::size_t>(r)] =
                        std::max(row_max[static_cast<std::size_t>(r)], scaled);
                    col_max[c] = std::max(col_max[c], scaled);
                }
            }

            // Ruiz's update: divide each scale by sqrt of the current inf-norm
            // of its row/column. An all-zero row or column keeps its factor.
            Real worst = 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                if (row_max[r] > 0.0) {
                    s.row_scale_[r] /= safe_sqrt(row_max[r]);
                    worst = std::max(worst, std::abs(row_max[r] - 1.0));
                }
            }
            for (std::size_t c = 0; c < cols; ++c) {
                if (col_max[c] > 0.0) {
                    s.col_scale_[c] /= safe_sqrt(col_max[c]);
                    worst = std::max(worst, std::abs(col_max[c] - 1.0));
                }
            }
            if (worst < options.ruiz_tolerance) break;
        }
    }

    // -- Pock-Chambolle diagonal preconditioner -----------------------------
    // Applied multiplicatively on top of the Ruiz factors, on the already-
    // equilibrated magnitudes. row weight  ~ 1 / sum_j |A_ij|^(2-alpha)
    //                          col weight  ~ 1 / sum_i |A_ij|^alpha
    if (options.apply_pock_chambolle) {
        const Real a = options.pock_chambolle_alpha;
        std::vector<Real> row_sum(rows, 0.0);
        std::vector<Real> col_sum(cols, 0.0);

        for (Idx r = 0; r < matrix.rows(); ++r) {
            const Idx end = row_ptr[static_cast<std::size_t>(r) + 1];
            for (Idx k = row_ptr[static_cast<std::size_t>(r)]; k < end; ++k) {
                const auto c = static_cast<std::size_t>(col_idx[static_cast<std::size_t>(k)]);
                const Real m = std::abs(values[static_cast<std::size_t>(k)])
                             * s.row_scale_[static_cast<std::size_t>(r)]
                             * s.col_scale_[c];
                if (m <= 0.0) continue;
                row_sum[static_cast<std::size_t>(r)] += std::pow(m, 2.0 - a);
                col_sum[c] += std::pow(m, a);
            }
        }
        for (std::size_t r = 0; r < rows; ++r)
            if (row_sum[r] > 0.0) s.row_scale_[r] /= safe_sqrt(row_sum[r]);
        for (std::size_t c = 0; c < cols; ++c)
            if (col_sum[c] > 0.0) s.col_scale_[c] /= safe_sqrt(col_sum[c]);
    }

    return s;
}

// --------------------------------------------------------------------------
// Scaling::apply
// --------------------------------------------------------------------------

CsrMatrix Scaling::apply(const CsrMatrix& matrix) const {
    if (matrix.rows() != num_rows() || matrix.cols() != num_cols())
        throw Error(Status::DimensionMismatch, "Scaling::apply");

    const auto row_ptr = matrix.row_ptr();
    const auto col_idx = matrix.col_idx();
    const auto values = matrix.values();

    std::vector<Idx> new_row_ptr(row_ptr.begin(), row_ptr.end());
    std::vector<Idx> new_col_idx(col_idx.begin(), col_idx.end());
    std::vector<Real> new_values(static_cast<std::size_t>(matrix.nnz()));

    for (Idx r = 0; r < matrix.rows(); ++r) {
        const Real rs = row_scale_[static_cast<std::size_t>(r)];
        const Idx end = row_ptr[static_cast<std::size_t>(r) + 1];
        for (Idx k = row_ptr[static_cast<std::size_t>(r)]; k < end; ++k) {
            const auto c = static_cast<std::size_t>(col_idx[static_cast<std::size_t>(k)]);
            new_values[static_cast<std::size_t>(k)] =
                values[static_cast<std::size_t>(k)] * rs * col_scale_[c];
        }
    }
    return CsrMatrix(matrix.rows(), matrix.cols(), std::move(new_row_ptr),
                     std::move(new_col_idx), std::move(new_values));
}

// --------------------------------------------------------------------------
// solution mapping
// --------------------------------------------------------------------------

void Scaling::unscale_primal(std::span<Real> x) const {
    if (static_cast<Idx>(x.size()) != num_cols())
        throw Error(Status::DimensionMismatch, "Scaling::unscale_primal");
    for (std::size_t j = 0; j < x.size(); ++j) x[j] *= col_scale_[j];
}

void Scaling::unscale_dual(std::span<Real> y) const {
    if (static_cast<Idx>(y.size()) != num_rows())
        throw Error(Status::DimensionMismatch, "Scaling::unscale_dual");
    for (std::size_t i = 0; i < y.size(); ++i) y[i] *= row_scale_[i];
}

// --------------------------------------------------------------------------
// Scaling::equilibrate (problem)
// --------------------------------------------------------------------------

std::pair<Problem, Scaling> Scaling::equilibrate(const Problem& problem,
                                                 const ScalingOptions& options) {
    const Scaling s = equilibrate(problem.matrix(), options);

    Problem::Builder b;
    b.set_name(problem.name());
    b.set_sense(problem.sense());
    b.set_objective_name(problem.objective_name());
    b.set_objective_constant(problem.objective_constant());

    // Substitute y_j = x_j / col_scale_j. Then:
    //   objective term  c_j x_j  =  (c_j * col_scale_j) y_j
    //   column bound    lo <= x_j <= hi  ->  lo/col_scale_j <= y_j <= hi/col_scale_j
    //   row activity    sum_j a_ij x_j  is scaled by row_scale_i, so the row
    //                   bounds scale by row_scale_i as well.
    // col_scale_j > 0 always (it is a product of positive sqrt factors), so the
    // bound divisions never flip an inequality.
    for (Idx j = 0; j < problem.num_cols(); ++j) {
        const auto k = static_cast<std::size_t>(j);
        const Idx col = b.add_column(std::string(problem.col_names()[k]));
        const Real cs = s.col_scale()[k];
        b.set_objective_coefficient(col, problem.objective()[k] * cs);
        const Real lo = problem.col_lower()[k];
        const Real hi = problem.col_upper()[k];
        b.set_column_bounds(col,
                            is_finite_bound(lo) ? lo / cs : lo,
                            is_finite_bound(hi) ? hi / cs : hi);
        b.set_column_kind(col, problem.col_kind()[k]);
    }
    for (Idx i = 0; i < problem.num_rows(); ++i) {
        const auto k = static_cast<std::size_t>(i);
        const Real rs = s.row_scale()[k];
        const Real lo = problem.row_lower()[k];
        const Real hi = problem.row_upper()[k];
        b.add_row(std::string(problem.row_names()[k]),
                  is_finite_bound(lo) ? lo * rs : lo,
                  is_finite_bound(hi) ? hi * rs : hi);
    }

    const CsrMatrix scaled = s.apply(problem.matrix());
    const auto rp = scaled.row_ptr();
    const auto ci = scaled.col_idx();
    const auto va = scaled.values();
    for (Idx r = 0; r < scaled.rows(); ++r) {
        const Idx end = rp[static_cast<std::size_t>(r) + 1];
        for (Idx t = rp[static_cast<std::size_t>(r)]; t < end; ++t)
            b.add_coefficient(r, ci[static_cast<std::size_t>(t)], va[static_cast<std::size_t>(t)]);
    }

    return {b.finish(), s};
}

// --------------------------------------------------------------------------
// conditioning read-outs
// --------------------------------------------------------------------------

Scaling::Conditioning Scaling::measure(const CsrMatrix& matrix) {
    Conditioning c;
    const auto ext = matrix.coefficient_extremes();
    c.max_abs = ext.max_abs;
    c.min_abs_nonzero = ext.min_abs_nonzero;
    c.coeff_spread = ext.min_abs_nonzero > 0.0 ? ext.max_abs / ext.min_abs_nonzero : 0.0;

    const auto row_ptr = matrix.row_ptr();
    const auto col_idx = matrix.col_idx();
    const auto values = matrix.values();

    std::vector<Real> row_norm(static_cast<std::size_t>(matrix.rows()), 0.0);
    std::vector<Real> col_norm(static_cast<std::size_t>(matrix.cols()), 0.0);
    for (Idx r = 0; r < matrix.rows(); ++r) {
        const Idx end = row_ptr[static_cast<std::size_t>(r) + 1];
        for (Idx k = row_ptr[static_cast<std::size_t>(r)]; k < end; ++k) {
            const Real m = std::abs(values[static_cast<std::size_t>(k)]);
            row_norm[static_cast<std::size_t>(r)] = std::max(row_norm[static_cast<std::size_t>(r)], m);
            const auto cc = static_cast<std::size_t>(col_idx[static_cast<std::size_t>(k)]);
            col_norm[cc] = std::max(col_norm[cc], m);
        }
    }
    auto spread = [](const std::vector<Real>& v) -> Real {
        Real lo = kInfinity, hi = 0.0;
        for (Real x : v) {
            if (x <= 0.0) continue;
            lo = std::min(lo, x);
            hi = std::max(hi, x);
        }
        return (hi > 0.0 && std::isfinite(lo)) ? hi / lo : 0.0;
    };
    c.row_norm_spread = spread(row_norm);
    c.col_norm_spread = spread(col_norm);
    return c;
}

std::string Scaling::summary() const {
    Real rlo = kInfinity, rhi = 0.0, clo = kInfinity, chi = 0.0;
    for (Real v : row_scale_) { rlo = std::min(rlo, v); rhi = std::max(rhi, v); }
    for (Real v : col_scale_) { clo = std::min(clo, v); chi = std::max(chi, v); }
    std::ostringstream os;
    os << "scaling: row factors in [" << rlo << ", " << rhi << "], "
       << "column factors in [" << clo << ", " << chi << "]";
    return os.str();
}

}  // namespace sov
