// Host reference backend -- Build Map ticket #3.
//
// Portable C++ implementations of every L0 primitive. Two jobs:
//
//   1. It is the correctness reference. When the CUDA backend lands, its
//      kernels are diffed against these, exactly as the from-scratch CPU
//      simplex (#4) is the oracle for the GPU solvers. "It ran on the GPU" is
//      not evidence; "it ran on the GPU and matched the host to 1e-12" is.
//   2. It keeps the abstraction honest. An interface with one implementation
//      is a fiction shaped around that implementation. With Host always
//      compiled, every call site is exercised by at least two backends.
//
// It is NOT the performance path -- Bible Part II is explicit that chasing the
// saturated CPU curve is the trap. No threading here on purpose; the parallel
// host work belongs to the scheduler of ticket #46.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include "sovereign/backend.hpp"

namespace sov {
namespace {

class HostBackend final : public Backend {
public:
    std::string_view name() const noexcept override { return "host"; }
    BackendKind kind() const noexcept override { return BackendKind::Host; }
    bool has_discrete_memory() const noexcept override { return false; }
    std::string_view device_description() const noexcept override {
        return "host CPU (reference implementation, single-threaded)";
    }

    // -- memory ----------------------------------------------------------
    // "Device" memory is host memory here, but it still goes through
    // allocate/copy so the ownership and residency machinery above is
    // exercised identically on every backend.

    Status allocate(std::size_t bytes, void** out) override {
        if (out == nullptr) return Status::InvalidArgument;
        if (bytes == 0) { *out = nullptr; return Status::Ok; }
        // Over-aligned to keep the door open for vectorized host kernels.
        void* p = std::aligned_alloc(64, ((bytes + 63) / 64) * 64);
        if (p == nullptr) return Status::OutOfMemory;
        *out = p;
        return Status::Ok;
    }

    void deallocate(void* ptr) noexcept override { std::free(ptr); }

    Status copy_to_device(const void* src, void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        if (src == nullptr || dst == nullptr) return Status::InvalidArgument;
        std::memcpy(dst, src, bytes);
        transfers_.h2d_calls += 1;
        transfers_.h2d_bytes += bytes;
        return Status::Ok;
    }

    Status copy_to_host(const void* src, void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        if (src == nullptr || dst == nullptr) return Status::InvalidArgument;
        std::memcpy(dst, src, bytes);
        transfers_.d2h_calls += 1;
        transfers_.d2h_bytes += bytes;
        return Status::Ok;
    }

    Status copy_device_to_device(const void* src, void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        if (src == nullptr || dst == nullptr) return Status::InvalidArgument;
        std::memmove(dst, src, bytes);
        return Status::Ok;
    }

    Status fill_zero(void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        if (dst == nullptr) return Status::InvalidArgument;
        std::memset(dst, 0, bytes);
        return Status::Ok;
    }

    Status synchronize() override { return Status::Ok; }

    // -- level 1 ---------------------------------------------------------

    Status axpy(Idx n, Real alpha, const Real* x, Real* y) override {
        if (n < 0) return Status::InvalidArgument;
        if (n > 0 && (x == nullptr || y == nullptr)) return Status::InvalidArgument;
        for (Idx i = 0; i < n; ++i) y[i] += alpha * x[i];
        return Status::Ok;
    }

    Status scale(Idx n, Real alpha, Real* x) override {
        if (n < 0) return Status::InvalidArgument;
        if (n > 0 && x == nullptr) return Status::InvalidArgument;
        for (Idx i = 0; i < n; ++i) x[i] *= alpha;
        return Status::Ok;
    }

    Status dot(Idx n, const Real* x, const Real* y, Real* result) override {
        if (n < 0 || result == nullptr) return Status::InvalidArgument;
        if (n > 0 && (x == nullptr || y == nullptr)) return Status::InvalidArgument;
        // Pairwise summation. A naive running sum loses accuracy linearly in n,
        // and this dot product feeds convergence tests where that error is
        // indistinguishable from a stalled iteration.
        *result = pairwise_dot(x, y, static_cast<std::size_t>(n));
        return Status::Ok;
    }

    Status norm2(Idx n, const Real* x, Real* result) override {
        Real d = 0;
        const Status s = dot(n, x, x, &d);
        if (s != Status::Ok) return s;
        *result = std::sqrt(d);
        return Status::Ok;
    }

    Status max_abs(Idx n, const Real* x, Real* result) override {
        if (n < 0 || result == nullptr) return Status::InvalidArgument;
        if (n > 0 && x == nullptr) return Status::InvalidArgument;
        Real m = 0;
        for (Idx i = 0; i < n; ++i) m = std::max(m, std::abs(x[i]));
        *result = m;
        return Status::Ok;
    }

    // -- sparse level 2 --------------------------------------------------

    Status spmv(const CsrView& A, Real alpha, const Real* x, Real beta, Real* y) override {
        if (A.rows < 0 || A.cols < 0) return Status::InvalidArgument;
        if (A.rows > 0 && (y == nullptr || A.row_ptr == nullptr)) return Status::InvalidArgument;
        if (A.nnz > 0 && (x == nullptr || A.col_idx == nullptr || A.values == nullptr))
            return Status::InvalidArgument;

        for (Idx r = 0; r < A.rows; ++r) {
            Real sum = 0.0;
            const Idx end = A.row_ptr[r + 1];
            for (Idx k = A.row_ptr[r]; k < end; ++k)
                sum += A.values[k] * x[A.col_idx[k]];
            // beta == 0 must *overwrite*, not multiply: y may hold uninitialized
            // memory or a NaN from a previous failed solve, and 0 * NaN is NaN.
            y[r] = (beta == Real{0}) ? alpha * sum : alpha * sum + beta * y[r];
        }
        return Status::Ok;
    }

    Status spmv_transpose(const CsrView& A, Real alpha, const Real* x,
                          Real beta, Real* y) override {
        if (A.rows < 0 || A.cols < 0) return Status::InvalidArgument;
        if (A.cols > 0 && y == nullptr) return Status::InvalidArgument;
        if (A.nnz > 0 && (x == nullptr || A.col_idx == nullptr || A.values == nullptr))
            return Status::InvalidArgument;

        if (beta == Real{0}) {
            for (Idx c = 0; c < A.cols; ++c) y[c] = 0.0;
        } else {
            for (Idx c = 0; c < A.cols; ++c) y[c] *= beta;
        }
        for (Idx r = 0; r < A.rows; ++r) {
            const Real ax = alpha * x[r];
            const Idx end = A.row_ptr[r + 1];
            for (Idx k = A.row_ptr[r]; k < end; ++k)
                y[A.col_idx[k]] += A.values[k] * ax;
        }
        return Status::Ok;
    }

    // -- elementwise -----------------------------------------------------

    Status project_box(Idx n, const Real* lo, const Real* hi, Real* x) override {
        if (n < 0) return Status::InvalidArgument;
        if (n > 0 && x == nullptr) return Status::InvalidArgument;
        for (Idx i = 0; i < n; ++i) {
            if (lo != nullptr && x[i] < lo[i]) x[i] = lo[i];
            if (hi != nullptr && x[i] > hi[i]) x[i] = hi[i];
        }
        return Status::Ok;
    }

    Status sort_by_key(Idx n, Idx* keys, Real* values) override {
        if (n < 0) return Status::InvalidArgument;
        if (n > 0 && (keys == nullptr || values == nullptr)) return Status::InvalidArgument;
        std::vector<std::size_t> order(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
        std::vector<Idx> k2(order.size());
        std::vector<Real> v2(order.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            k2[i] = keys[order[i]];
            v2[i] = values[order[i]];
        }
        std::copy(k2.begin(), k2.end(), keys);
        std::copy(v2.begin(), v2.end(), values);
        return Status::Ok;
    }

private:
    /// Divide-and-conquer summation: error grows like log(n) rather than n.
    static Real pairwise_dot(const Real* x, const Real* y, std::size_t n) {
        constexpr std::size_t kBlock = 128;
        if (n == 0) return 0.0;
        if (n <= kBlock) {
            Real s = 0.0;
            for (std::size_t i = 0; i < n; ++i) s += x[i] * y[i];
            return s;
        }
        const std::size_t half = n / 2;
        return pairwise_dot(x, y, half) + pairwise_dot(x + half, y + half, n - half);
    }
};

}  // namespace

std::unique_ptr<Backend> make_host_backend() {
    return std::unique_ptr<Backend>(new HostBackend());
}

}  // namespace sov
