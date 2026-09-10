// Model file readers -- Build Map ticket #5. Bible S4.7 (L6 interface layer).
//
// The front door for every benchmark instance, and later for the Python
// bindings. MPS is the format the corpora actually use; LP is supported as the
// human-writable alternative.
#pragma once

#include <string>
#include <vector>

#include "sovereign/problem.hpp"

namespace sov {

/// Non-fatal observations made while reading: an ignored section, a bound type
/// applied under a convention that is genuinely ambiguous across
/// implementations, a duplicate coefficient that was summed.
///
/// These exist because the ticket's warning is that a parser which "silently
/// mishandles one" dialect quirk produces a *different problem* than the
/// benchmark intends. Silence is the danger, so anything ambiguous is recorded
/// rather than swallowed.
struct ReaderWarning {
    int line = 0;
    std::string section;
    std::string message;
};

struct ReadResult {
    Problem problem;
    std::vector<ReaderWarning> warnings;
    std::string format;   ///< "mps" or "lp"
};

/// How to resolve the one genuinely ambiguous MPS rule (see mps_reader.cpp).
enum class NegativeUpperBoundRule {
    /// UP with a negative value on a column still at its default lower bound
    /// of zero also drives the lower bound to -infinity. What CPLEX, Gurobi
    /// and HiGHS do; the default here for that reason.
    LowerBecomesMinusInfinity,
    /// Leave the lower bound at zero, yielding a crossed, infeasible bound
    /// pair. The literal reading of the standard.
    KeepLowerAtZero,
};

struct MpsOptions {
    NegativeUpperBoundRule negative_upper_bound = NegativeUpperBoundRule::LowerBecomesMinusInfinity;
    /// Keep non-objective N rows as free rows so row counts match the file.
    /// Dropping them changes the reported shape, which is what ticket #5 is
    /// tested on.
    bool keep_free_rows = true;
};

/// Read an MPS file. Throws Error with a line number on a malformed file --
/// a model that cannot be read is never a model that gets solved anyway.
/// Handles gzip-compressed files transparently when the name ends in .gz.
ReadResult read_mps(const std::string& path, const MpsOptions& options = {});
ReadResult read_mps_string(const std::string& text, const MpsOptions& options = {},
                           const std::string& origin = "<string>");

/// Read a CPLEX LP file. See lp_reader.cpp for the supported subset.
ReadResult read_lp(const std::string& path);
ReadResult read_lp_string(const std::string& text, const std::string& origin = "<string>");

/// Dispatch on file extension (.mps, .mps.gz, .lp, .lp.gz), falling back to
/// sniffing the contents.
ReadResult read_model(const std::string& path);

}  // namespace sov
