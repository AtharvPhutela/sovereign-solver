// CUDA backend -- Build Map ticket #3. Bible S4.1.
//
// Compiled only when a CUDA toolkit is found at configure time
// (SOVEREIGN_WITH_CUDA). Uses cuSPARSE for SpMV and cuBLAS for level-1, both
// classified PERMITTED in sovereignty.toml: they are numerical primitives.
// cuSPARSE multiplies a matrix by a vector; the primal-dual iteration wrapped
// around it, the restarts, the preconditioning and the convergence theory are
// ours (docs/dependency-ledger.md S5.2).
//
// STATUS ON THIS MACHINE: unverified. The development host has no CUDA toolkit
// and no driver stack, so this translation unit has never been compiled or
// run. It is written against the documented cuSPARSE/cuBLAS APIs and must be
// treated as unproven until it is built and diffed against the Host backend
// (see tests/test_l0_backend.cpp, which runs the identical suite over every
// available backend for exactly that purpose).
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <string>

#include "sovereign/backend.hpp"

namespace sov {
namespace {

// cuSPARSE's generic API wants an index type constant matching our Idx.
constexpr cusparseIndexType_t kCusparseIndexType =
    kIndex64 ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;

inline Status from_cuda(cudaError_t e) {
    if (e == cudaSuccess) return Status::Ok;
    if (e == cudaErrorMemoryAllocation) return Status::OutOfMemory;
    return Status::BackendError;
}

inline Status from_cublas(cublasStatus_t s) {
    if (s == CUBLAS_STATUS_SUCCESS) return Status::Ok;
    if (s == CUBLAS_STATUS_ALLOC_FAILED) return Status::OutOfMemory;
    return Status::BackendError;
}

inline Status from_cusparse(cusparseStatus_t s) {
    if (s == CUSPARSE_STATUS_SUCCESS) return Status::Ok;
    if (s == CUSPARSE_STATUS_ALLOC_FAILED) return Status::OutOfMemory;
    return Status::BackendError;
}

/// x := clamp(x, lo, hi). A null bound pointer means unbounded on that side.
/// This is the projection half of PDHG's inner loop; keeping it as one kernel
/// (rather than two passes) matters because the loop is bandwidth-bound.
__global__ void project_box_kernel(long long n, const Real* lo, const Real* hi, Real* x) {
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    Real v = x[i];
    if (lo != nullptr) { const Real l = lo[i]; if (v < l) v = l; }
    if (hi != nullptr) { const Real h = hi[i]; if (v > h) v = h; }
    x[i] = v;
}

__global__ void abs_max_kernel(long long n, const Real* x, Real* partial) {
    extern __shared__ Real shared[];
    const unsigned tid = threadIdx.x;
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + tid;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;

    Real local = 0;
    for (; i < n; i += stride) {
        const Real a = fabs(x[i]);
        if (a > local) local = a;
    }
    shared[tid] = local;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared[tid + s] > shared[tid]) shared[tid] = shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = shared[0];
}

class CudaBackend final : public Backend {
public:
    CudaBackend() {
        if (cublasCreate(&blas_) != CUBLAS_STATUS_SUCCESS)
            throw Error("cuBLAS handle creation failed");
        if (cusparseCreate(&sparse_) != CUSPARSE_STATUS_SUCCESS) {
            cublasDestroy(blas_);
            throw Error("cuSPARSE handle creation failed");
        }
        // Scalars (alpha/beta) are passed from host memory.
        cublasSetPointerMode(blas_, CUBLAS_POINTER_MODE_HOST);
        cusparseSetPointerMode(sparse_, CUSPARSE_POINTER_MODE_HOST);

        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            description_ = std::string(prop.name) + " (sm_"
                         + std::to_string(prop.major) + std::to_string(prop.minor)
                         + ", " + std::to_string(prop.totalGlobalMem >> 20) + " MiB)";
        } else {
            description_ = "CUDA device (properties unavailable)";
        }
    }

    ~CudaBackend() override {
        if (spmv_workspace_ != nullptr) cudaFree(spmv_workspace_);
        cusparseDestroy(sparse_);
        cublasDestroy(blas_);
    }

    std::string_view name() const noexcept override { return "cuda"; }
    BackendKind kind() const noexcept override { return BackendKind::Cuda; }
    bool has_discrete_memory() const noexcept override { return true; }
    std::string_view device_description() const noexcept override { return description_; }

    // -- memory ----------------------------------------------------------

    Status allocate(std::size_t bytes, void** out) override {
        if (out == nullptr) return Status::InvalidArgument;
        if (bytes == 0) { *out = nullptr; return Status::Ok; }
        return from_cuda(cudaMalloc(out, bytes));
    }

    void deallocate(void* ptr) noexcept override {
        if (ptr != nullptr) cudaFree(ptr);
    }

    Status copy_to_device(const void* src, void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        const Status s = from_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
        if (s == Status::Ok) {
            transfers_.h2d_calls += 1;
            transfers_.h2d_bytes += bytes;
        }
        return s;
    }

    Status copy_to_host(const void* src, void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        const Status s = from_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
        if (s == Status::Ok) {
            transfers_.d2h_calls += 1;
            transfers_.d2h_bytes += bytes;
        }
        return s;
    }

    Status copy_device_to_device(const void* src, void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        return from_cuda(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    }

    Status fill_zero(void* dst, std::size_t bytes) override {
        if (bytes == 0) return Status::Ok;
        return from_cuda(cudaMemset(dst, 0, bytes));
    }

    Status synchronize() override { return from_cuda(cudaDeviceSynchronize()); }

    // -- level 1 ---------------------------------------------------------

    Status axpy(Idx n, Real alpha, const Real* x, Real* y) override {
        if (n == 0) return Status::Ok;
        return from_cublas(cublasDaxpy(blas_, static_cast<int>(n), &alpha, x, 1, y, 1));
    }

    Status scale(Idx n, Real alpha, Real* x) override {
        if (n == 0) return Status::Ok;
        return from_cublas(cublasDscal(blas_, static_cast<int>(n), &alpha, x, 1));
    }

    Status dot(Idx n, const Real* x, const Real* y, Real* result) override {
        if (result == nullptr) return Status::InvalidArgument;
        if (n == 0) { *result = 0; return Status::Ok; }
        return from_cublas(cublasDdot(blas_, static_cast<int>(n), x, 1, y, 1, result));
    }

    Status norm2(Idx n, const Real* x, Real* result) override {
        if (result == nullptr) return Status::InvalidArgument;
        if (n == 0) { *result = 0; return Status::Ok; }
        return from_cublas(cublasDnrm2(blas_, static_cast<int>(n), x, 1, result));
    }

    Status max_abs(Idx n, const Real* x, Real* result) override {
        // Deliberately not cublasIdamax: that returns an *index*, and reading
        // the element back would be a second device-to-host round trip for one
        // scalar. A custom reduction keeps it to one.
        if (result == nullptr) return Status::InvalidArgument;
        if (n == 0) { *result = 0; return Status::Ok; }

        constexpr int kThreads = 256;
        const int blocks = static_cast<int>(
            std::min<long long>(1024, (static_cast<long long>(n) + kThreads - 1) / kThreads));

        Real* partial = nullptr;
        const Status alloc = allocate(static_cast<std::size_t>(blocks) * sizeof(Real),
                                      reinterpret_cast<void**>(&partial));
        if (alloc != Status::Ok) return alloc;

        abs_max_kernel<<<blocks, kThreads, kThreads * sizeof(Real)>>>(
            static_cast<long long>(n), x, partial);
        Status s = from_cuda(cudaGetLastError());
        if (s == Status::Ok) {
            std::vector<Real> host(static_cast<std::size_t>(blocks));
            s = copy_to_host(partial, host.data(), host.size() * sizeof(Real));
            if (s == Status::Ok) {
                Real m = 0;
                for (Real v : host) if (v > m) m = v;
                *result = m;
            }
        }
        deallocate(partial);
        return s;
    }

    // -- sparse level 2 --------------------------------------------------

    Status spmv(const CsrView& A, Real alpha, const Real* x, Real beta, Real* y) override {
        return spmv_impl(A, alpha, x, beta, y, CUSPARSE_OPERATION_NON_TRANSPOSE);
    }

    Status spmv_transpose(const CsrView& A, Real alpha, const Real* x,
                          Real beta, Real* y) override {
        return spmv_impl(A, alpha, x, beta, y, CUSPARSE_OPERATION_TRANSPOSE);
    }

    // -- elementwise -----------------------------------------------------

    Status project_box(Idx n, const Real* lo, const Real* hi, Real* x) override {
        if (n == 0) return Status::Ok;
        if (x == nullptr) return Status::InvalidArgument;
        constexpr int kThreads = 256;
        const long long blocks = (static_cast<long long>(n) + kThreads - 1) / kThreads;
        project_box_kernel<<<static_cast<int>(blocks), kThreads>>>(
            static_cast<long long>(n), lo, hi, x);
        return from_cuda(cudaGetLastError());
    }

private:
    Status spmv_impl(const CsrView& A, Real alpha, const Real* x, Real beta, Real* y,
                     cusparseOperation_t op) {
        if (A.rows == 0 || A.cols == 0) return Status::Ok;
        if (A.nnz == 0) {
            // Still has to honour beta, or a caller relying on y := beta*y
            // silently keeps stale values.
            const Idx m = (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? A.rows : A.cols;
            if (beta == Real{0}) return fill_zero(y, static_cast<std::size_t>(m) * sizeof(Real));
            return scale(m, beta, y);
        }

        cusparseSpMatDescr_t mat = nullptr;
        cusparseStatus_t cs = cusparseCreateCsr(
            &mat, A.rows, A.cols, A.nnz,
            const_cast<Idx*>(A.row_ptr), const_cast<Idx*>(A.col_idx),
            const_cast<Real*>(A.values),
            kCusparseIndexType, kCusparseIndexType,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        if (cs != CUSPARSE_STATUS_SUCCESS) return from_cusparse(cs);

        const Idx x_len = (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? A.cols : A.rows;
        const Idx y_len = (op == CUSPARSE_OPERATION_NON_TRANSPOSE) ? A.rows : A.cols;

        cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
        cusparseCreateDnVec(&vx, x_len, const_cast<Real*>(x), CUDA_R_64F);
        cusparseCreateDnVec(&vy, y_len, y, CUDA_R_64F);

        std::size_t needed = 0;
        cs = cusparseSpMV_bufferSize(sparse_, op, &alpha, mat, vx, &beta, vy,
                                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &needed);
        Status st = from_cusparse(cs);
        if (st == Status::Ok) st = ensure_workspace(needed);
        if (st == Status::Ok) {
            cs = cusparseSpMV(sparse_, op, &alpha, mat, vx, &beta, vy,
                              CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_workspace_);
            st = from_cusparse(cs);
        }

        cusparseDestroyDnVec(vx);
        cusparseDestroyDnVec(vy);
        cusparseDestroySpMat(mat);
        return st;
    }

    /// The SpMV scratch buffer is grown and kept, never reallocated per call.
    /// PDHG runs this millions of times; a cudaMalloc in the inner loop would
    /// serialize on the allocator and dominate the kernel it serves.
    Status ensure_workspace(std::size_t bytes) {
        if (bytes <= spmv_workspace_bytes_) return Status::Ok;
        if (spmv_workspace_ != nullptr) cudaFree(spmv_workspace_);
        spmv_workspace_ = nullptr;
        spmv_workspace_bytes_ = 0;
        const Status s = from_cuda(cudaMalloc(&spmv_workspace_, bytes));
        if (s == Status::Ok) spmv_workspace_bytes_ = bytes;
        return s;
    }

    cublasHandle_t blas_ = nullptr;
    cusparseHandle_t sparse_ = nullptr;
    void* spmv_workspace_ = nullptr;
    std::size_t spmv_workspace_bytes_ = 0;
    std::string description_;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend() {
    return std::unique_ptr<Backend>(new CudaBackend());
}

}  // namespace sov
