// Minimal test harness -- Build Map ticket #3.
//
// GoogleTest and Catch2 are both PERMITTED (sovereignty.toml), but pulling one
// in means a FetchContent download into the build tree and a network
// dependency in CI, to replace roughly sixty lines. Not a good trade at this
// stage. Swap to GoogleTest whenever the suite outgrows this; nothing depends
// on the harness beyond these macros.
#pragma once

#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace tst {

struct Case {
    const char* suite;
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

struct Registrar {
    Registrar(const char* suite, const char* name, std::function<void()> fn) {
        registry().push_back(Case{suite, name, std::move(fn)});
    }
};

/// Thrown by a failing check; carries the location so the report is actionable.
struct Failure {
    std::string message;
};

inline void fail(const char* file, int line, const std::string& what) {
    throw Failure{std::string(file) + ":" + std::to_string(line) + "  " + what};
}

inline bool near(double a, double b, double tol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    const double diff = std::abs(a - b);
    if (diff <= tol) return true;
    const double scale = std::max(std::abs(a), std::abs(b));
    return diff <= tol * scale;
}

inline int run_all(const char* filter = nullptr) {
    int passed = 0;
    std::vector<std::string> failures;
    for (const Case& c : registry()) {
        if (filter != nullptr && std::string(c.suite).find(filter) == std::string::npos)
            continue;
        try {
            c.fn();
            std::printf("  [ ok ] %s.%s\n", c.suite, c.name);
            ++passed;
        } catch (const Failure& f) {
            std::printf("  [FAIL] %s.%s\n         %s\n", c.suite, c.name, f.message.c_str());
            failures.push_back(std::string(c.suite) + "." + c.name);
        } catch (const std::exception& e) {
            std::printf("  [FAIL] %s.%s\n         unexpected exception: %s\n",
                        c.suite, c.name, e.what());
            failures.push_back(std::string(c.suite) + "." + c.name);
        } catch (...) {
            std::printf("  [FAIL] %s.%s\n         unexpected non-standard exception\n",
                        c.suite, c.name);
            failures.push_back(std::string(c.suite) + "." + c.name);
        }
    }
    std::printf("\n%d passed, %zu failed\n", passed, failures.size());
    for (const std::string& f : failures) std::printf("  failed: %s\n", f.c_str());
    return failures.empty() ? 0 : 1;
}

}  // namespace tst

#define TST_CONCAT_(a, b) a##b
#define TST_CONCAT(a, b) TST_CONCAT_(a, b)

#define TEST(suite, name)                                                       \
    static void TST_CONCAT(tst_fn_, __LINE__)();                                \
    static ::tst::Registrar TST_CONCAT(tst_reg_, __LINE__)(                     \
        #suite, #name, TST_CONCAT(tst_fn_, __LINE__));                          \
    static void TST_CONCAT(tst_fn_, __LINE__)()

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) ::tst::fail(__FILE__, __LINE__, "CHECK failed: " #cond);   \
    } while (0)

#define CHECK_MSG(cond, msg)                                                    \
    do {                                                                        \
        if (!(cond))                                                            \
            ::tst::fail(__FILE__, __LINE__,                                     \
                        std::string("CHECK failed: " #cond " -- ") + (msg));    \
    } while (0)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                        \
        const auto tst_a_ = (a);                                                \
        const auto tst_b_ = (b);                                                \
        if (!(tst_a_ == tst_b_))                                                \
            ::tst::fail(__FILE__, __LINE__,                                     \
                        std::string("CHECK_EQ failed: " #a " == " #b            \
                                    "\n           lhs = ")                      \
                            + std::to_string(tst_a_)                            \
                            + "\n           rhs = " + std::to_string(tst_b_));  \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                   \
    do {                                                                        \
        const double tst_a_ = static_cast<double>(a);                           \
        const double tst_b_ = static_cast<double>(b);                           \
        if (!::tst::near(tst_a_, tst_b_, (tol)))                                \
            ::tst::fail(__FILE__, __LINE__,                                     \
                        std::string("CHECK_NEAR failed: " #a " ~= " #b          \
                                    "\n           lhs = ")                      \
                            + std::to_string(tst_a_)                            \
                            + "\n           rhs = " + std::to_string(tst_b_)    \
                            + "\n           tol = " + std::to_string(tol));     \
    } while (0)

#define CHECK_THROWS(expr)                                                      \
    do {                                                                        \
        bool tst_threw_ = false;                                                \
        try { (void)(expr); } catch (...) { tst_threw_ = true; }                \
        if (!tst_threw_)                                                        \
            ::tst::fail(__FILE__, __LINE__,                                     \
                        "CHECK_THROWS failed, no exception: " #expr);           \
    } while (0)

#define TST_MAIN()                                                              \
    int main(int argc, char** argv) {                                           \
        return ::tst::run_all(argc > 1 ? argv[1] : nullptr);                    \
    }
