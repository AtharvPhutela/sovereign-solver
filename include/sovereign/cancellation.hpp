// Cooperative cancellation -- Build Map ticket #10. Bible S4.2, "concurrency
// by default."
//
// PURPOSE. The concurrent engine race (race.hpp) launches every continuous
// engine on the same model at once and wants the losers to actually STOP,
// not keep burning CPU (or, later, VRAM and GPU cycles) after a winner is
// already in hand. "The cancellation must be clean" (Build Map #10's own
// "Watch out") means every engine checks a shared flag on its own hot loop
// and returns promptly -- not that the race harness kills threads or leaks
// half-finished state. A `CancellationToken` is exactly that flag, shared by
// pointer so the harness can set it from outside without the engine ever
// needing to know who is racing it.
//
// Checking an atomic load once per iteration is a fixed, negligible cost
// next to an SpMV or a dense factorization -- this is deliberately not
// polled more often than that.
#pragma once

#include <atomic>

namespace sov {

class CancellationToken {
public:
    void cancel() noexcept { flag_.store(true, std::memory_order_relaxed); }
    bool is_cancelled() const noexcept { return flag_.load(std::memory_order_relaxed); }

private:
    std::atomic<bool> flag_{false};
};

}  // namespace sov
