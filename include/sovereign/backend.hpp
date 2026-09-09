// L0 backend abstraction -- Build Map ticket #3, gate M0. Bible S4.1, Part V.
//
// Every primitive the solver will ever run on a device goes through this one
// interface. The ticket is blunt about why it has to exist on day one: hardcode
// CUDA calls at the call sites and hardware sovereignty is gone, and putting
// the seam back in later is a rewrite rather than a refactor.
//
// The cost is one virtual call per primitive. Bible S4.1 accepts that trade
// explicitly -- "costs a little indirection, buys hardware sovereignty" -- and
// the granularity keeps it honest: a call here is a whole SpMV or a whole
// reduction over millions of entries, never a per-element operation.
//
// THREE implementations, not two:
//
//   Host  -- portable C++. Always available. It is not a placeholder: it is
//            the reference the GPU backends are diffed against, the same role
//            the from-scratch CPU simplex (#4) plays for the GPU solvers. It
//            also means the abstraction is exercised by more than one
//            implementation from the first commit, which is the only way to
//            know the seam is real and not a fiction shaped around CUDA.
//   Cuda  -- cuSPARSE / cuBLAS / Thrust. Compiled when a CUDA toolkit is found.
//   Hip   -- rocSPARSE / rocBLAS. Compiled when ROCm is found; currently a
//            declared skeleton, per the ticket ("the ROCm path compiles, even
//            if stubbed").
//
// VRAM RESIDENCY is the other non-negotiable (Bible Part III: "PCIe transfers
// are the enemy"). It is enforced structurally here, not by convention:
// DeviceCsr and DeviceVector are move-only, so an accidental copy does not
// compile, and every backend counts its host<->device transfers so a test can
// assert that a matrix crossed the bus exactly once.
#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/sparse.hpp"
#include "sovereign/status.hpp"

namespace sov {

enum class BackendKind { Host, Cuda, Hip };

constexpr const char* to_string(BackendKind k) noexcept {
    switch (k) {
        case BackendKind::Host: return "host";
        case BackendKind::Cuda: return "cuda";
        case BackendKind::Hip:  return "hip";
    }
    return "?";
}

/// Device-resident CSR, as the backend sees it. Raw pointers because their
/// meaning is backend-specific: device addresses for Cuda/Hip, host addresses
/// for Host. Ownership lives in DeviceCsr below.
struct CsrView {
    Idx rows = 0;
    Idx cols = 0;
    Idx nnz = 0;
    const Idx* row_ptr = nullptr;
    const Idx* col_idx = nullptr;
    const Real* values = nullptr;
};

/// Counts of what actually crossed the bus. The point of tracking this is that
/// "the matrix stays resident" is otherwise an unfalsifiable claim -- with a
/// counter, a test can assert it (see tests/test_l0_backend.cpp).
struct TransferStats {
    std::size_t h2d_calls = 0;
    std::size_t d2h_calls = 0;
    std::size_t h2d_bytes = 0;
    std::size_t d2h_bytes = 0;
};

// --------------------------------------------------------------------------
// The interface
// --------------------------------------------------------------------------

class Backend {
public:
    virtual ~Backend() = default;

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    virtual std::string_view name() const noexcept = 0;
    virtual BackendKind kind() const noexcept = 0;

    /// True when the backend has a memory space distinct from the host, i.e.
    /// when copy_h2d is a real transfer rather than a memcpy.
    virtual bool has_discrete_memory() const noexcept = 0;

    /// Human-readable device description, for benchmark provenance.
    virtual std::string_view device_description() const noexcept = 0;

    // -- memory -----------------------------------------------------------
    virtual Status allocate(std::size_t bytes, void** out) = 0;
    virtual void deallocate(void* ptr) noexcept = 0;
    virtual Status copy_to_device(const void* host_src, void* dev_dst, std::size_t bytes) = 0;
    virtual Status copy_to_host(const void* dev_src, void* host_dst, std::size_t bytes) = 0;
    virtual Status copy_device_to_device(const void* src, void* dst, std::size_t bytes) = 0;
    virtual Status fill_zero(void* dev_dst, std::size_t bytes) = 0;

    /// Block until queued work has completed. A no-op on Host.
    virtual Status synchronize() = 0;

    // -- level-1 ----------------------------------------------------------
    virtual Status axpy(Idx n, Real alpha, const Real* x, Real* y) = 0;
    virtual Status scale(Idx n, Real alpha, Real* x) = 0;
    virtual Status dot(Idx n, const Real* x, const Real* y, Real* result) = 0;
    virtual Status norm2(Idx n, const Real* x, Real* result) = 0;
    virtual Status max_abs(Idx n, const Real* x, Real* result) = 0;

    // -- sparse level-2 ---------------------------------------------------
    // Both orientations are primitives rather than one being expressed via the
    // other: PDHG needs Ax and A^T y every iteration, and a transpose-by-
    // materialization would double the resident footprint of the largest
    // object in the solver.

    /// y := alpha * A * x + beta * y
    virtual Status spmv(const CsrView& A, Real alpha, const Real* x,
                        Real beta, Real* y) = 0;

    /// y := alpha * A^T * x + beta * y
    virtual Status spmv_transpose(const CsrView& A, Real alpha, const Real* x,
                                  Real beta, Real* y) = 0;

    // -- elementwise ------------------------------------------------------
    /// x := clamp(x, lo, hi), elementwise. The projection half of PDHG's
    /// iteration (Bible S4.2 Engine A): SpMV plus this is the entire inner
    /// loop. A null bound pointer means unbounded on that side.
    virtual Status project_box(Idx n, const Real* lo, const Real* hi, Real* x) = 0;

    // -- declared for later tickets --------------------------------------
    // Present so the interface does not have to change when the ticket that
    // needs them arrives. Default to NotImplemented rather than being absent,
    // so a caller gets a clear status instead of a compile error against a
    // backend that has not caught up.

    /// Sort (key, value) pairs by key. Needed for CSR<->CSC on device and for
    /// cut-pool ordering in L4.
    virtual Status sort_by_key(Idx n, Idx* keys, Real* values) {
        (void)n; (void)keys; (void)values;
        return Status::NotImplemented;
    }

    /// Sparse LDL^T of a symmetric quasi-definite system. Ticket #9's
    /// pivoting-free IPM (Bible S4.2 Engine B) is the caller; on Cuda this
    /// becomes cuDSS.
    virtual Status factorize_ldlt(const CsrView& A, void** factor_out) {
        (void)A; (void)factor_out;
        return Status::NotImplemented;
    }

    // -- instrumentation --------------------------------------------------
    const TransferStats& transfers() const noexcept { return transfers_; }
    void reset_transfer_stats() noexcept { transfers_ = {}; }

protected:
    Backend() = default;
    TransferStats transfers_{};
};

// --------------------------------------------------------------------------
// Factory
// --------------------------------------------------------------------------

/// Backends compiled into this build. Host is always present.
std::vector<BackendKind> available_backends();

bool is_available(BackendKind kind);

/// Preference order Cuda > Hip > Host: the fastest substrate actually built.
BackendKind default_backend_kind();

/// Throws Error if the kind was not compiled in; check is_available() first
/// when a fallback is wanted rather than a failure.
std::unique_ptr<Backend> make_backend(BackendKind kind);

/// A short line naming what this build can target, for logs and reports.
std::string backend_report();

// --------------------------------------------------------------------------
// Owning device handles
//
// Move-only by design. Bible Part III makes PCIe traffic the enemy, so a
// silent deep copy of the constraint matrix is exactly the bug this type
// system should refuse to compile.
// --------------------------------------------------------------------------

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(Backend& backend, std::size_t count);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    /// Allocate and upload in one step.
    static DeviceBuffer upload(Backend& backend, std::span<const Real> host_data);

    Status download(std::span<Real> host_dst) const;
    std::vector<Real> to_host() const;

    Real* data() noexcept { return ptr_; }
    const Real* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }
    Backend* backend() const noexcept { return backend_; }

private:
    void release() noexcept;

    Backend* backend_ = nullptr;
    Real* ptr_ = nullptr;
    std::size_t count_ = 0;
};

/// A constraint matrix that lives on the device for its whole lifetime.
///
/// Constructed by one upload and never copied. Every subsequent operation
/// takes a view() -- so the residency property the architecture depends on is
/// a property of the type, not of the discipline of whoever writes the solver.
class DeviceCsr {
public:
    DeviceCsr() = default;
    ~DeviceCsr();

    DeviceCsr(const DeviceCsr&) = delete;
    DeviceCsr& operator=(const DeviceCsr&) = delete;
    DeviceCsr(DeviceCsr&& other) noexcept;
    DeviceCsr& operator=(DeviceCsr&& other) noexcept;

    /// The single point at which the matrix crosses the bus.
    static DeviceCsr upload(Backend& backend, const CsrMatrix& host_matrix);

    CsrView view() const noexcept {
        return CsrView{rows_, cols_, nnz_, row_ptr_, col_idx_, values_};
    }

    Idx rows() const noexcept { return rows_; }
    Idx cols() const noexcept { return cols_; }
    Idx nnz() const noexcept { return nnz_; }
    std::size_t device_bytes() const noexcept { return bytes_; }
    bool empty() const noexcept { return values_ == nullptr; }
    Backend* backend() const noexcept { return backend_; }

private:
    void release() noexcept;

    Backend* backend_ = nullptr;
    Idx rows_ = 0, cols_ = 0, nnz_ = 0;
    Idx* row_ptr_ = nullptr;
    Idx* col_idx_ = nullptr;
    Real* values_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace sov
