// Hypergraph structure detection -- Build Map ticket #14, gate M3. Bible
// S4.4, S4.2A ("the L1.5 dispatch decision").
//
// SCOPE. The Build Map asks for multi-level hypergraph partitioning
// (columns as nodes, rows as hyperedges, minimize the cut corresponding to
// coupling constraints) -- genuinely NP-hard, and the ticket itself is
// tagged Research because a GPU-parallel version is an open question in the
// literature. What is built here is a much simpler, single-level, EXACT
// (not minimized-cut, not heuristically refined) classifier that proves the
// idea on the clean case real block-angular / scenario-decomposable models
// actually present: most rows are LOCAL to one block of columns, and a
// small number of LINKING rows couple columns across blocks (a stochastic
// program's first-stage variables, a capacity constraint touching every
// scenario, etc.).
//
// THE METHOD (from scratch, no partitioning library -- see sovereignty.toml,
// hypergraph partitioners are FORBIDDEN precisely because this is ticket
// #14's own job). Rows are split into two sets by degree (active nonzero
// count) relative to the MEDIAN row degree -- median rather than mean
// because a handful of high-degree linking rows should not drag the
// reference degree up and hide themselves; a few outliers barely move a
// median the way they move a mean:
//   - LOCAL CANDIDATES (degree <= threshold): every column such a row
//     touches is unioned into the same disjoint-set component (Bible's
//     column-graph idea, built directly rather than via an explicit graph).
//   - LINKING CANDIDATES (degree > threshold): excluded from the union step
//     entirely, on the working assumption that a row touching unusually
//     many columns is more likely a coupling constraint than a local one.
// After every local-candidate row has been unioned, each column has a
// final block id. EVERY row (both candidate sets) is then reclassified by
// how many DISTINCT blocks its columns actually belong to: exactly one ->
// LOCAL to that block; more than one -> LINKING, regardless of which
// candidate set it started in (a "local candidate" that still happens to
// bridge two blocks -- e.g. a lone row connecting two otherwise-disjoint
// pieces -- is correctly caught here, not hidden by the initial guess).
//
// This is deliberately NOT the general hypergraph min-cut problem: it finds
// the TRUE connected-component structure exactly (a correct, checkable
// claim) rather than an approximately-optimal partition of a structure that
// doesn't cleanly decompose. On an instance with no real block structure,
// it is expected to -- and does, see the "Watch out" in the .cpp -- report
// Monolithic rather than force a partition that isn't really there.
#pragma once

#include <string>
#include <vector>

#include "sovereign/numeric.hpp"
#include "sovereign/problem.hpp"

namespace sov {

enum class StructureKind { Monolithic, BlockAngular };

const char* to_string(StructureKind k) noexcept;

struct StructureOptions {
    /// A row is treated as a LINKING CANDIDATE (excluded from the union-find
    /// pass) when its active nonzero count exceeds this factor times the
    /// MEDIAN row degree. See the file comment for why median, not mean.
    double degree_threshold_factor = 2.0;

    /// Absolute floor on the degree threshold, so a degree-1-median instance
    /// (extremely sparse, e.g. mostly singleton rows) doesn't end up
    /// excluding every row with 2+ nonzeros as "linking".
    Idx min_degree_threshold = 4;

    /// After reclassification, BlockAngular is only reported when the
    /// number of blocks found is at least this many...
    Idx min_blocks = 2;

    /// ...AND linking rows are no more than this fraction of all rows.
    /// Above it, the "structure" isn't actually exploitable -- decomposing
    /// would mean shipping most of the model across the coupling boundary
    /// anyway, which is the case #25's own dispatcher (ticket #25) should
    /// route to the monolithic solve regardless of what this classifier
    /// technically found.
    double max_linking_row_fraction = 0.5;
};

struct StructureResult {
    StructureKind kind = StructureKind::Monolithic;

    Idx num_blocks = 0;
    std::vector<Idx> col_block;     ///< length num_cols; block id, or -1 if the
                                    ///< column touches no row at all
    std::vector<Idx> row_block;     ///< length num_rows; block id for a LOCAL
                                    ///< row, -1 for a linking row or an empty one
    std::vector<Idx> linking_rows;  ///< rows touching more than one block

    std::string message;   ///< explains a Monolithic verdict (too few blocks,
                           ///< too many linking rows, ...)
};

/// Classifies `problem`'s block structure. See the file comment for the
/// method and its scope.
StructureResult detect_structure(const Problem& problem, const StructureOptions& options = {});

}  // namespace sov
