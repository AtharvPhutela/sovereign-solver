// L0 error vocabulary -- Build Map ticket #3, gate M0.
//
// Two channels, deliberately:
//
//   Status  -- for conditions a caller can legitimately meet and handle: a
//              backend that does not implement a primitive, a device that is
//              out of memory, a factorization that broke down. These are
//              returned, never thrown, because the concurrent-engine design
//              (Bible S4.2) expects engines to fail and be raced past.
//
//   throw   -- for programmer error: a malformed matrix, mismatched dimensions
//              wired up at assembly time. These are bugs, not outcomes.
//
// The distinction matters more here than in ordinary code. A numerical routine
// that returns a plausible-looking wrong answer is worse than one that
// crashes (Build Map, "Test step is load-bearing"), so anything that would
// silently produce a wrong number must be loud.
#pragma once

#include <stdexcept>
#include <string>

namespace sov {

enum class Status {
    Ok = 0,
    Unsupported,        ///< this backend has no implementation of the primitive
    NotImplemented,     ///< declared for a later ticket; deliberately absent
    InvalidArgument,
    DimensionMismatch,
    OutOfMemory,
    BackendError,       ///< the underlying runtime (CUDA/HIP) reported a failure
    NumericalFailure,   ///< breakdown: singular pivot, non-finite iterate
};

constexpr const char* to_string(Status s) noexcept {
    switch (s) {
        case Status::Ok:               return "ok";
        case Status::Unsupported:      return "unsupported by this backend";
        case Status::NotImplemented:   return "not implemented yet";
        case Status::InvalidArgument:  return "invalid argument";
        case Status::DimensionMismatch:return "dimension mismatch";
        case Status::OutOfMemory:      return "out of memory";
        case Status::BackendError:     return "backend error";
        case Status::NumericalFailure: return "numerical failure";
    }
    return "unknown status";
}

constexpr bool ok(Status s) noexcept { return s == Status::Ok; }

/// Thrown by `check()`; also the type used for structural/programmer errors.
class Error : public std::runtime_error {
public:
    explicit Error(std::string what) : std::runtime_error(std::move(what)) {}
    Error(Status s, const std::string& context)
        : std::runtime_error(context + ": " + to_string(s)), status_(s) {}

    Status status() const noexcept { return status_; }

private:
    Status status_ = Status::BackendError;
};

/// Promote a Status to an exception at a boundary where failure is not an
/// expected outcome (test rigs, one-shot setup paths).
inline void check(Status s, const char* context) {
    if (s != Status::Ok) throw Error(s, context);
}

}  // namespace sov
