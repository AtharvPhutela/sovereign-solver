// L0 backend factory and owning device handles -- Build Map ticket #3.
#include "sovereign/backend.hpp"

#include <string>
#include <utility>

namespace sov {

// Provided by the per-backend translation units. The CUDA and HIP ones are
// only compiled when their toolkit was found; the preprocessor guards below
// are the single place that knowledge lives.
std::unique_ptr<Backend> make_host_backend();
#if defined(SOVEREIGN_WITH_CUDA)
std::unique_ptr<Backend> make_cuda_backend();
#endif
#if defined(SOVEREIGN_WITH_HIP)
std::unique_ptr<Backend> make_hip_backend();
#endif

// --------------------------------------------------------------------------
// Factory
// --------------------------------------------------------------------------

std::vector<BackendKind> available_backends() {
    std::vector<BackendKind> kinds;
    kinds.push_back(BackendKind::Host);   // always
#if defined(SOVEREIGN_WITH_CUDA)
    kinds.push_back(BackendKind::Cuda);
#endif
#if defined(SOVEREIGN_WITH_HIP)
    kinds.push_back(BackendKind::Hip);
#endif
    return kinds;
}

bool is_available(BackendKind kind) {
    for (BackendKind k : available_backends())
        if (k == kind) return true;
    return false;
}

BackendKind default_backend_kind() {
    // Fastest substrate actually compiled in. Bible Part II: the GPU is the
    // primary compute substrate, so it wins whenever it exists.
    if (is_available(BackendKind::Cuda)) return BackendKind::Cuda;
    if (is_available(BackendKind::Hip)) return BackendKind::Hip;
    return BackendKind::Host;
}

std::unique_ptr<Backend> make_backend(BackendKind kind) {
    switch (kind) {
        case BackendKind::Host:
            return make_host_backend();
        case BackendKind::Cuda:
#if defined(SOVEREIGN_WITH_CUDA)
            return make_cuda_backend();
#else
            throw Error("backend 'cuda' was not compiled into this build "
                        "(no CUDA toolkit found at configure time)");
#endif
        case BackendKind::Hip:
#if defined(SOVEREIGN_WITH_HIP)
            return make_hip_backend();
#else
            throw Error("backend 'hip' was not compiled into this build "
                        "(no ROCm installation found at configure time)");
#endif
    }
    throw Error("unknown backend kind");
}

std::string backend_report() {
    std::string s = "backends:";
    for (BackendKind k : available_backends()) {
        s += ' ';
        s += to_string(k);
        if (k == default_backend_kind()) s += "(default)";
    }
    s += "  index=";
    s += kIndex64 ? "int64" : "int32";
    s += "  real=f64";
    return s;
}

// --------------------------------------------------------------------------
// DeviceBuffer
// --------------------------------------------------------------------------

DeviceBuffer::DeviceBuffer(Backend& backend, std::size_t count)
    : backend_(&backend), count_(count) {
    if (count == 0) return;
    void* raw = nullptr;
    check(backend.allocate(count * sizeof(Real), &raw), "DeviceBuffer::allocate");
    ptr_ = static_cast<Real*>(raw);
    check(backend.fill_zero(ptr_, count * sizeof(Real)), "DeviceBuffer::fill_zero");
}

DeviceBuffer DeviceBuffer::upload(Backend& backend, std::span<const Real> host_data) {
    DeviceBuffer buf(backend, host_data.size());
    if (!host_data.empty())
        check(backend.copy_to_device(host_data.data(), buf.ptr_,
                                     host_data.size() * sizeof(Real)),
              "DeviceBuffer::upload");
    return buf;
}

DeviceBuffer::~DeviceBuffer() { release(); }

void DeviceBuffer::release() noexcept {
    if (backend_ != nullptr && ptr_ != nullptr) backend_->deallocate(ptr_);
    ptr_ = nullptr;
    count_ = 0;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : backend_(other.backend_), ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        release();
        backend_ = other.backend_;
        ptr_ = other.ptr_;
        count_ = other.count_;
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    return *this;
}

Status DeviceBuffer::download(std::span<Real> host_dst) const {
    if (host_dst.size() != count_) return Status::DimensionMismatch;
    if (count_ == 0) return Status::Ok;
    return backend_->copy_to_host(ptr_, host_dst.data(), count_ * sizeof(Real));
}

std::vector<Real> DeviceBuffer::to_host() const {
    std::vector<Real> out(count_);
    check(download(out), "DeviceBuffer::to_host");
    return out;
}

// --------------------------------------------------------------------------
// DeviceCsr
//
// The upload is the one and only bus crossing for the constraint matrix. Note
// there is no download and no non-const accessor: nothing in the design ever
// needs to read the matrix back, and not providing the door is cheaper than
// policing it.
// --------------------------------------------------------------------------

DeviceCsr DeviceCsr::upload(Backend& backend, const CsrMatrix& host_matrix) {
    std::string why;
    const Status v = host_matrix.validate(&why);
    if (v != Status::Ok)
        throw Error(v, "DeviceCsr::upload: source matrix is malformed (" + why + ")");

    DeviceCsr d;
    d.backend_ = &backend;
    d.rows_ = host_matrix.rows();
    d.cols_ = host_matrix.cols();
    d.nnz_ = host_matrix.nnz();

    const std::size_t ptr_bytes = host_matrix.row_ptr().size() * sizeof(Idx);
    const std::size_t idx_bytes = host_matrix.col_idx().size() * sizeof(Idx);
    const std::size_t val_bytes = host_matrix.values().size() * sizeof(Real);
    d.bytes_ = ptr_bytes + idx_bytes + val_bytes;

    void* raw = nullptr;
    check(backend.allocate(ptr_bytes, &raw), "DeviceCsr: row_ptr allocate");
    d.row_ptr_ = static_cast<Idx*>(raw);
    check(backend.allocate(idx_bytes, &raw), "DeviceCsr: col_idx allocate");
    d.col_idx_ = static_cast<Idx*>(raw);
    check(backend.allocate(val_bytes, &raw), "DeviceCsr: values allocate");
    d.values_ = static_cast<Real*>(raw);

    check(backend.copy_to_device(host_matrix.row_ptr().data(), d.row_ptr_, ptr_bytes),
          "DeviceCsr: row_ptr upload");
    if (idx_bytes > 0) {
        check(backend.copy_to_device(host_matrix.col_idx().data(), d.col_idx_, idx_bytes),
              "DeviceCsr: col_idx upload");
        check(backend.copy_to_device(host_matrix.values().data(), d.values_, val_bytes),
              "DeviceCsr: values upload");
    }
    return d;
}

DeviceCsr::~DeviceCsr() { release(); }

void DeviceCsr::release() noexcept {
    if (backend_ != nullptr) {
        backend_->deallocate(row_ptr_);
        backend_->deallocate(col_idx_);
        backend_->deallocate(values_);
    }
    row_ptr_ = nullptr;
    col_idx_ = nullptr;
    values_ = nullptr;
    rows_ = cols_ = nnz_ = 0;
    bytes_ = 0;
}

DeviceCsr::DeviceCsr(DeviceCsr&& other) noexcept
    : backend_(other.backend_), rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_),
      row_ptr_(other.row_ptr_), col_idx_(other.col_idx_), values_(other.values_),
      bytes_(other.bytes_) {
    other.row_ptr_ = nullptr;
    other.col_idx_ = nullptr;
    other.values_ = nullptr;
    other.rows_ = other.cols_ = other.nnz_ = 0;
    other.bytes_ = 0;
}

DeviceCsr& DeviceCsr::operator=(DeviceCsr&& other) noexcept {
    if (this != &other) {
        release();
        backend_ = other.backend_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        nnz_ = other.nnz_;
        row_ptr_ = other.row_ptr_;
        col_idx_ = other.col_idx_;
        values_ = other.values_;
        bytes_ = other.bytes_;
        other.row_ptr_ = nullptr;
        other.col_idx_ = nullptr;
        other.values_ = nullptr;
        other.rows_ = other.cols_ = other.nnz_ = 0;
        other.bytes_ = 0;
    }
    return *this;
}

}  // namespace sov
