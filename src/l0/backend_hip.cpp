// ROCm / HIP backend -- Build Map ticket #3.
//
// Compiled only when ROCm is found at configure time (SOVEREIGN_WITH_HIP).
//
// This is the SKELETON the ticket asks for: "the ROCm path compiles (even if
// stubbed)". Memory management and level-1 are wired to hipMalloc/rocBLAS
// because they are mechanical; the sparse kernels return Unsupported with a
// pointer to the rocSPARSE call that fills them in.
//
// WHY IT EXISTS NOW RATHER THAN LATER. The ticket's own warning: hardcode CUDA
// at the call sites and hardware sovereignty is gone, because retrofitting the
// seam is a rewrite. The way that failure actually happens is not that someone
// decides against portability -- it is that with only one backend, nothing
// stops a CUDA-shaped assumption leaking into the interface, and nobody
// notices until the second backend is attempted. A compiling second GPU
// backend, even a stubbed one, is what makes the seam falsifiable.
//
// Bible Part X.6 is candid that true hardware sovereignty depends on ROCm
// reaching parity, and until it does the fast path is NVIDIA-first with a
// CPU/ROCm correctness fallback. This file is the fallback's foundation.
//
// STATUS ON THIS MACHINE: unverified. No ROCm installation here, so this
// translation unit has never been compiled.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <string>

#include "sovereign/backend.hpp"

namespace sov {
namespace {

inline Status from_hip(hipError_t e) {
    if (e == hipSuccess) return Status::Ok;
    if (e == hipErrorOutOfMemory) return Status::OutOfMemory;
    return Status::BackendError;
}

inline Status from_rocblas(rocblas_status s) {
    if (s == rocblas_status_success) return Status::Ok;
    if (s == rocblas_status_memory_error) return Status::OutOfMemory;
    return Status::BackendError;
}

__global__ void project_box_kernel(long long n, const Real* lo, const Real* hi, Real* x) {
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    Real v = x[i];
    if (lo != nullptr) { const Real l = lo[i]; if (v < l) v = l; }
    if (hi != nullptr) { const Real h = hi[i]; if (v > h) v = h; }
    x[i] = v;
}

class HipBackend final : public Backend {
public:
    HipBackend() {
        if (rocblas_create_handle(&blas_) != rocblas_status_success)
            throw Error("rocBLAS handle creation failed");
        rocblas_set_pointer_mode(blas_, rocblas_pointer_mode_host);

        int device = 0;
        hipGetDevice(&device);
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) == hipSuccess) {
            description_ = std::string(prop.name) + " (ROCm, "
                         + std::to_string(prop.totalGlobalMem >> 20) + " MiB)";
        } else {
            description_ = "HIP device (properties unavailable)";
        }
    }

    ~HipBackend() override { rocblas_destroy_handle(blas_); }

    std::string_view name() const noexcept override { return "hip"; }
    BackendKind kind() const noexcept override { return BackendKind::Hip; }
    bool has_discrete_memory() const noexcept override { return true; }
    std::string_view device_description() const noexcept override { return description_; }

    // -- memory: complete -------------------------------------------------

    Status allocate(std::size_t bytes, void** out) override {
        if (out == nullptr) return Status::InvalidArgument;
        if (bytes == 0) { *out = nullptr; return Status::Ok; }
        return from_hip(hipMalloc(out, bytes));
    }

    void deallocate(void* ptr) noexcept override {
        if (ptr != nullptr) hipFree(ptr);
    }

    Status copy_to_device(const void* src, void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        const Status s = from_hip(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
        if (s == Status::Ok) {
            transfers_.h2d_calls += 1;
            transfers_.h2d_bytes += bytes;
        }
        return s;
    }

    Status copy_to_host(const void* src, void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        const Status s = from_hip(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
        if (s == Status::Ok) {
            transfers_.d2h_calls += 1;
            transfers_.d2h_bytes += bytes;
        }
        return s;
    }

    Status copy_device_to_device(const void* src, void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        return from_hip(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice));
    }

    Status fill_zero(void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        return from_hip(hipMemset(dst, 0, bytes));
    }

    Status synchronize() override { return from_hip(hipDeviceSynchronize()); }

    // -- level 1: complete ------------------------------------------------

    Status axpy(Idx n, Real alpha, const Real* x, Real* y) override {
        if (n == 0) return Status::Ok;
        return from_rocblas(rocblas_daxpy(blas_, static_cast<int>(n), &alpha, x, 1, y, 1));
    }

    Status scale(Idx n, Real alpha, Real* x) override {
        if (n == 0) return Status::Ok;
        return from_rocblas(rocblas_dscal(blas_, static_cast<int>(n), &alpha, x, 1));
    }

    Status dot(Idx n, const Real* x, const Real* y, Real* result) override {
        if (result == nullptr) return Status::InvalidArgument;
        if (n == 0) { *result = 0; return Status::Ok; }
        return from_rocblas(rocblas_ddot(blas_, static_cast<int>(n), x, 1, y, 1, result));
    }

    Status norm2(Idx n, const Real* x, Real* result) override {
        if (result == nullptr) return Status::InvalidArgument;
        if (n == 0) { *result = 0; return Status::Ok; }
        return from_rocblas(rocblas_dnrm2(blas_, static_cast<int>(n), x, 1, result));
    }

    // -- elementwise: complete --------------------------------------------

    Status project_box(Idx n, const Real* lo, const Real* hi, Real* x) override {
        if (n == 0) return Status::Ok;
        if (x == nullptr) return Status::InvalidArgument;
        constexpr int kThreads = 256;
        const long long blocks = (static_cast<long long>(n) + kThreads - 1) / kThreads;
        hipLaunchKernelGGL(project_box_kernel, dim3(static_cast<int>(blocks)),
                           dim3(kThreads), 0, 0,
                           static_cast<long long>(n), lo, hi, x);
        return from_hip(hipGetLastError());
    }

    // -- TO BE FILLED IN ---------------------------------------------------
    // Each returns Unsupported rather than a wrong answer. The concurrent
    // engine design (Bible S4.2) already expects a backend to decline work, so
    // an honest Unsupported is a first-class outcome; a plausible-looking wrong
    // number is not.

    /// TODO: rocsparse_spmv with rocsparse_operation_none, CSR descriptor,
    /// mirroring backend_cuda.cu's spmv_impl including the retained workspace.
    Status max_abs(Idx, const Real*, Real*) override { return Status::Unsupported; }
    Status spmv(const CsrView&, Real, const Real*, Real, Real*) override {
        return Status::Unsupported;
    }
    Status spmv_transpose(const CsrView&, Real, const Real*, Real, Real*) override {
        return Status::Unsupported;
    }

private:
    rocblas_handle blas_ = nullptr;
    std::string description_;
};

}  // namespace

std::unique_ptr<Backend> make_hip_backend() {
    return std::unique_ptr<Backend>(new HipBackend());
}

}  // namespace sov
