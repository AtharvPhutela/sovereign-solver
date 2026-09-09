// L0 backend abstraction tests -- Build Map ticket #3, gate M0.
//
// This is the ticket's stated test: "allocate a CSR matrix on device, run one
// SpMV through the abstraction layer against a hand-checked dense result;
// confirm the abstraction dispatches correctly and the numbers match."
//
// Every check below runs over EVERY backend compiled into the build. That is
// the design point rather than a convenience: the same hand-computed answers
// are demanded of Host, CUDA and HIP alike, so adding a GPU backend cannot
// quietly change what "correct" means. On a machine with a CUDA toolkit these
// same cases become the host-vs-device differential test with no edit.
//
// Reference matrix (identical to test_l0_sparse.cpp):
//
//        [  2   0  -1 ]
//   A =  [  0   3   4 ]        4 x 3, nnz = 6
//        [  1   0   0 ]
//        [  0   0   5 ]
//
//   x = (1, 2, 3)      =>  A x   = (-1, 18, 1, 15)
//   y = (1, 2, 3, 4)   =>  A^T y = (5, 6, 27)
#include <cstdio>
#include <vector>

#include "sovereign/backend.hpp"
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

/// Run `body` once per backend compiled into this build, labelling any failure
/// with the backend that produced it.
template <class F>
void for_each_backend(F&& body) {
    for (const BackendKind kind : available_backends()) {
        std::unique_ptr<Backend> be = make_backend(kind);
        try {
            body(*be);
        } catch (const tst::Failure& f) {
            throw tst::Failure{std::string("[backend=") + to_string(kind) + "] " + f.message};
        }
    }
}

}  // namespace

// --------------------------------------------------------------------------
// Factory and reporting
// --------------------------------------------------------------------------

TEST(backend, host_is_always_available) {
    CHECK(is_available(BackendKind::Host));
    CHECK(!available_backends().empty());
}

TEST(backend, default_prefers_a_device_when_one_was_compiled_in) {
    const BackendKind d = default_backend_kind();
    if (is_available(BackendKind::Cuda)) {
        CHECK(d == BackendKind::Cuda);
    } else if (is_available(BackendKind::Hip)) {
        CHECK(d == BackendKind::Hip);
    } else {
        CHECK(d == BackendKind::Host);
    }
}

TEST(backend, requesting_an_uncompiled_backend_fails_loudly) {
    // Silently falling back to Host would make a benchmark claim "GPU" while
    // running on the CPU. Refusing is the only safe behaviour.
    if (!is_available(BackendKind::Cuda))
        CHECK_THROWS(make_backend(BackendKind::Cuda));
    if (!is_available(BackendKind::Hip))
        CHECK_THROWS(make_backend(BackendKind::Hip));
}

TEST(backend, every_backend_identifies_itself) {
    for_each_backend([](Backend& be) {
        CHECK(!be.name().empty());
        CHECK(!be.device_description().empty());
        CHECK(be.kind() == BackendKind::Host ? !be.has_discrete_memory()
                                             : be.has_discrete_memory());
    });
}

// --------------------------------------------------------------------------
// Memory and the residency contract
// --------------------------------------------------------------------------

TEST(backend, buffer_upload_download_roundtrip) {
    for_each_backend([](Backend& be) {
        const std::vector<Real> host = {1.0, -2.5, 3.25, 0.0, 1e-7};
        DeviceBuffer buf = DeviceBuffer::upload(be, host);
        CHECK_EQ(buf.size(), host.size());
        const std::vector<Real> back = buf.to_host();
        for (std::size_t i = 0; i < host.size(); ++i) CHECK_NEAR(back[i], host[i], kTol);
    });
}

TEST(backend, new_buffer_is_zeroed) {
    // Not cosmetic: spmv with beta=0 writes over y, but every other consumer
    // reads it, and an uninitialized NaN propagates silently through a solve.
    for_each_backend([](Backend& be) {
        DeviceBuffer buf(be, 16);
        for (Real v : buf.to_host()) CHECK_NEAR(v, 0.0, kTol);
    });
}

TEST(backend, matrix_crosses_the_bus_exactly_once) {
    // The VRAM-residency contract (Bible Part III: "PCIe transfers are the
    // enemy"), asserted rather than asserted-about. Three uploads for the
    // three CSR arrays, and then no further host-to-device traffic no matter
    // how many times the matrix is used.
    for_each_backend([](Backend& be) {
        const CsrMatrix host_A = reference_matrix();
        be.reset_transfer_stats();

        DeviceCsr A = DeviceCsr::upload(be, host_A);
        CHECK_EQ(be.transfers().h2d_calls, std::size_t{3});
        const std::size_t after_upload = be.transfers().h2d_bytes;
        CHECK_EQ(after_upload, host_A.byte_size());

        DeviceBuffer x = DeviceBuffer::upload(be, std::vector<Real>{1.0, 2.0, 3.0});
        DeviceBuffer y(be, 4);
        const std::size_t baseline = be.transfers().h2d_calls;

        for (int i = 0; i < 32; ++i)
            CHECK(be.spmv(A.view(), 1.0, x.data(), 0.0, y.data()) == Status::Ok);

        CHECK_MSG(be.transfers().h2d_calls == baseline,
                  "an SpMV re-uploaded data; the matrix is not resident");
    });
}

TEST(backend, device_handles_are_move_only) {
    // Compile-time property, restated here so the intent is discoverable:
    // a deep copy of the constraint matrix is the single most expensive
    // accident available, so it must not compile.
    static_assert(!std::is_copy_constructible_v<DeviceCsr>);
    static_assert(!std::is_copy_assignable_v<DeviceCsr>);
    static_assert(std::is_move_constructible_v<DeviceCsr>);
    static_assert(!std::is_copy_constructible_v<DeviceBuffer>);
    static_assert(std::is_move_constructible_v<DeviceBuffer>);

    for_each_backend([](Backend& be) {
        DeviceCsr a = DeviceCsr::upload(be, reference_matrix());
        const Idx nnz = a.nnz();
        DeviceCsr b = std::move(a);            // ownership transfers...
        CHECK_EQ(b.nnz(), nnz);
        CHECK(a.empty());                      // ...and the source is emptied
    });
}

TEST(backend, upload_rejects_a_malformed_matrix) {
    for_each_backend([](Backend& be) {
        const CsrMatrix bad(1, 3, {0, 2}, {2, 0}, {1.0, 2.0});  // unsorted columns
        CHECK_THROWS(DeviceCsr::upload(be, bad));
    });
}

// --------------------------------------------------------------------------
// Level 1
// --------------------------------------------------------------------------

TEST(backend, axpy_scale_dot_norm) {
    for_each_backend([](Backend& be) {
        DeviceBuffer x = DeviceBuffer::upload(be, std::vector<Real>{1.0, 2.0, 3.0, 4.0});
        DeviceBuffer y = DeviceBuffer::upload(be, std::vector<Real>{10.0, 20.0, 30.0, 40.0});

        CHECK(be.axpy(4, 2.0, x.data(), y.data()) == Status::Ok);
        const std::vector<Real> after = y.to_host();     // y += 2x
        CHECK_NEAR(after[0], 12.0, kTol);
        CHECK_NEAR(after[3], 48.0, kTol);

        CHECK(be.scale(4, 0.5, y.data()) == Status::Ok);
        CHECK_NEAR(y.to_host()[0], 6.0, kTol);

        Real d = 0;
        CHECK(be.dot(4, x.data(), x.data(), &d) == Status::Ok);
        CHECK_NEAR(d, 30.0, kTol);                       // 1+4+9+16

        Real n = 0;
        CHECK(be.norm2(4, x.data(), &n) == Status::Ok);
        CHECK_NEAR(n, std::sqrt(30.0), 1e-12);
    });
}

TEST(backend, max_abs) {
    for_each_backend([](Backend& be) {
        DeviceBuffer x = DeviceBuffer::upload(be, std::vector<Real>{1.0, -7.5, 3.0, 2.0});
        Real m = 0;
        const Status s = be.max_abs(4, x.data(), &m);
        if (s == Status::Unsupported) return;            // HIP skeleton, honestly declined
        CHECK(s == Status::Ok);
        CHECK_NEAR(m, 7.5, kTol);
    });
}

TEST(backend, level1_on_empty_vectors_is_a_no_op) {
    for_each_backend([](Backend& be) {
        Real d = 123.0;
        CHECK(be.dot(0, nullptr, nullptr, &d) == Status::Ok);
        CHECK_NEAR(d, 0.0, kTol);
        CHECK(be.axpy(0, 1.0, nullptr, nullptr) == Status::Ok);
    });
}

// --------------------------------------------------------------------------
// SpMV -- the ticket's headline test
// --------------------------------------------------------------------------

TEST(backend, spmv_matches_the_hand_computed_result) {
    for_each_backend([](Backend& be) {
        DeviceCsr A = DeviceCsr::upload(be, reference_matrix());
        DeviceBuffer x = DeviceBuffer::upload(be, std::vector<Real>{1.0, 2.0, 3.0});
        DeviceBuffer y(be, 4);

        const Status s = be.spmv(A.view(), 1.0, x.data(), 0.0, y.data());
        if (s == Status::Unsupported) return;            // HIP skeleton
        CHECK(s == Status::Ok);
        CHECK(be.synchronize() == Status::Ok);

        const std::vector<Real> out = y.to_host();
        CHECK_NEAR(out[0], -1.0, kTol);
        CHECK_NEAR(out[1], 18.0, kTol);
        CHECK_NEAR(out[2], 1.0, kTol);
        CHECK_NEAR(out[3], 15.0, kTol);
    });
}

TEST(backend, spmv_honours_alpha_and_beta) {
    // y := 2*A*x + 3*y  with y0 = (1,1,1,1)
    //    = 2*(-1,18,1,15) + (3,3,3,3) = (1, 39, 5, 33)
    for_each_backend([](Backend& be) {
        DeviceCsr A = DeviceCsr::upload(be, reference_matrix());
        DeviceBuffer x = DeviceBuffer::upload(be, std::vector<Real>{1.0, 2.0, 3.0});
        DeviceBuffer y = DeviceBuffer::upload(be, std::vector<Real>{1.0, 1.0, 1.0, 1.0});

        const Status s = be.spmv(A.view(), 2.0, x.data(), 3.0, y.data());
        if (s == Status::Unsupported) return;
        CHECK(s == Status::Ok);
        CHECK(be.synchronize() == Status::Ok);

        const std::vector<Real> out = y.to_host();
        CHECK_NEAR(out[0], 1.0, kTol);
        CHECK_NEAR(out[1], 39.0, kTol);
        CHECK_NEAR(out[2], 5.0, kTol);
        CHECK_NEAR(out[3], 33.0, kTol);
    });
}

TEST(backend, spmv_transpose_matches_the_hand_computed_result) {
    // A^T y is not a convenience: PDHG needs it every iteration, and
    // materializing a transpose would double the resident footprint of the
    // largest object in the solver.
    for_each_backend([](Backend& be) {
        DeviceCsr A = DeviceCsr::upload(be, reference_matrix());
        DeviceBuffer y = DeviceBuffer::upload(be, std::vector<Real>{1.0, 2.0, 3.0, 4.0});
        DeviceBuffer out(be, 3);

        const Status s = be.spmv_transpose(A.view(), 1.0, y.data(), 0.0, out.data());
        if (s == Status::Unsupported) return;
        CHECK(s == Status::Ok);
        CHECK(be.synchronize() == Status::Ok);

        const std::vector<Real> o = out.to_host();
        CHECK_NEAR(o[0], 5.0, kTol);
        CHECK_NEAR(o[1], 6.0, kTol);
        CHECK_NEAR(o[2], 27.0, kTol);
    });
}

TEST(backend, spmv_with_beta_zero_overwrites_rather_than_multiplies) {
    // 0 * NaN is NaN, so beta == 0 must overwrite y, not scale it. A y buffer
    // holding a NaN from a previous failed solve would otherwise poison every
    // subsequent iteration, and the failure would surface far from its cause.
    for_each_backend([](Backend& be) {
        DeviceCsr A = DeviceCsr::upload(be, reference_matrix());
        DeviceBuffer x = DeviceBuffer::upload(be, std::vector<Real>{1.0, 2.0, 3.0});
        const Real nan = std::numeric_limits<Real>::quiet_NaN();
        DeviceBuffer y = DeviceBuffer::upload(be, std::vector<Real>{nan, nan, nan, nan});

        const Status s = be.spmv(A.view(), 1.0, x.data(), 0.0, y.data());
        if (s == Status::Unsupported) return;
        CHECK(s == Status::Ok);
        CHECK(be.synchronize() == Status::Ok);

        const std::vector<Real> out = y.to_host();
        for (Real v : out) CHECK_MSG(!std::isnan(v), "beta=0 failed to overwrite y");
        CHECK_NEAR(out[1], 18.0, kTol);
    });
}

TEST(backend, spmv_agrees_with_the_host_reference_implementation) {
    // The differential check that gives the whole abstraction its value: every
    // backend must reproduce CsrMatrix::multiply exactly. On a CUDA-equipped
    // machine this is the host-vs-device test, unchanged.
    const CsrMatrix host_A = reference_matrix();
    const std::vector<Real> x = {0.5, -1.25, 2.75};
    std::vector<Real> expected(4, 0.0);
    host_A.multiply(x, expected);

    for_each_backend([&](Backend& be) {
        DeviceCsr A = DeviceCsr::upload(be, host_A);
        DeviceBuffer dx = DeviceBuffer::upload(be, x);
        DeviceBuffer dy(be, 4);
        const Status s = be.spmv(A.view(), 1.0, dx.data(), 0.0, dy.data());
        if (s == Status::Unsupported) return;
        CHECK(s == Status::Ok);
        CHECK(be.synchronize() == Status::Ok);
        const std::vector<Real> got = dy.to_host();
        for (std::size_t i = 0; i < expected.size(); ++i)
            CHECK_NEAR(got[i], expected[i], 1e-14);
    });
}

// --------------------------------------------------------------------------
// Elementwise projection -- PDHG's other half
// --------------------------------------------------------------------------

TEST(backend, project_box_clamps_both_sides) {
    for_each_backend([](Backend& be) {
        DeviceBuffer lo = DeviceBuffer::upload(be, std::vector<Real>{0.0, 0.0, -1.0, -5.0});
        DeviceBuffer hi = DeviceBuffer::upload(be, std::vector<Real>{1.0, 10.0, 1.0, 5.0});
        DeviceBuffer x = DeviceBuffer::upload(be, std::vector<Real>{-3.0, 4.0, 7.0, 0.5});

        CHECK(be.project_box(4, lo.data(), hi.data(), x.data()) == Status::Ok);
        CHECK(be.synchronize() == Status::Ok);

        const std::vector<Real> out = x.to_host();
        CHECK_NEAR(out[0], 0.0, kTol);    // clamped up to lo
        CHECK_NEAR(out[1], 4.0, kTol);    // untouched
        CHECK_NEAR(out[2], 1.0, kTol);    // clamped down to hi
        CHECK_NEAR(out[3], 0.5, kTol);    // untouched
    });
}

TEST(backend, project_box_treats_a_null_bound_as_unbounded) {
    for_each_backend([](Backend& be) {
        DeviceBuffer lo = DeviceBuffer::upload(be, std::vector<Real>{0.0, 0.0});
        DeviceBuffer x = DeviceBuffer::upload(be, std::vector<Real>{-3.0, 1e9});
        CHECK(be.project_box(2, lo.data(), nullptr, x.data()) == Status::Ok);
        CHECK(be.synchronize() == Status::Ok);
        const std::vector<Real> out = x.to_host();
        CHECK_NEAR(out[0], 0.0, kTol);
        CHECK_NEAR(out[1], 1e9, 1e-6);    // no upper bound applied
    });
}

// --------------------------------------------------------------------------
// Primitives reserved for later tickets
// --------------------------------------------------------------------------

TEST(backend, unimplemented_primitives_say_so_rather_than_guessing) {
    for_each_backend([](Backend& be) {
        void* factor = nullptr;
        const CsrView empty{};
        // Ticket #9 (pivoting-free IPM) fills this in via cuDSS.
        CHECK(be.factorize_ldlt(empty, &factor) == Status::NotImplemented);
    });
}

TEST(backend, host_sort_by_key_orders_pairs) {
    std::unique_ptr<Backend> be = make_backend(BackendKind::Host);
    std::vector<Idx> keys = {3, 1, 2};
    std::vector<Real> vals = {30.0, 10.0, 20.0};
    CHECK(be->sort_by_key(3, keys.data(), vals.data()) == Status::Ok);
    CHECK_EQ(keys[0], 1);
    CHECK_EQ(keys[2], 3);
    CHECK_NEAR(vals[0], 10.0, kTol);
    CHECK_NEAR(vals[2], 30.0, kTol);
}

int main(int argc, char** argv) {
    std::printf("%s\n", backend_report().c_str());
    for (const BackendKind k : available_backends()) {
        std::unique_ptr<Backend> be = make_backend(k);
        std::printf("  %-5s : %s\n", to_string(k),
                    std::string(be->device_description()).c_str());
    }
    std::printf("\n");
    return tst::run_all(argc > 1 ? argv[1] : nullptr);
}
