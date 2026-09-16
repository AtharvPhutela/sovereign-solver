// Hypergraph structure detection -- Build Map ticket #14. See structure.hpp
// for the method and its scope.
//
// WATCH OUT, HONESTLY NAMED (matching the ticket's own framing -- Research,
// literature thin). Two failure modes are structural, not bugs, and both
// are exercised by tests rather than left to be discovered in the field:
//   1. A single row that happens to touch columns from what would otherwise
//      be two separate blocks merges them into one, even if that row was a
//      LOCAL CANDIDATE by degree. This is CORRECT behavior (that row really
//      does make the model non-separable there), not a detection failure --
//      see test_structure.cpp's "one_bridging_row_correctly_merges_blocks".
//   2. The degree-threshold split is a heuristic, not a proof. An instance
//      whose true linking rows do NOT stand out by degree (e.g. a coupling
//      constraint with only 2 terms, no denser than the local rows around
//      it) will not be excluded from the union step and the blocks it
//      couples will simply merge -- reported as Monolithic, which is the
//      honest, safe answer (never claims decomposability that isn't
//      checkable) rather than a wrong BlockAngular claim.
#include "sovereign/structure.hpp"

#include <algorithm>
#include <numeric>

#include "sovereign/sparse.hpp"

namespace sov {

const char* to_string(StructureKind k) noexcept {
    switch (k) {
        case StructureKind::Monolithic:  return "monolithic";
        case StructureKind::BlockAngular: return "block_angular";
    }
    return "unknown";
}

namespace {

/// Disjoint-set (union-find) with path compression and union-by-size --
/// textbook, no partitioning library involved (sovereignty.toml forbids
/// exactly that for this job).
class DisjointSet {
public:
    explicit DisjointSet(std::size_t n) : parent_(n), size_(n, 1) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    std::size_t find(std::size_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];   // path halving
            x = parent_[x];
        }
        return x;
    }

    void unite(std::size_t a, std::size_t b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (size_[a] < size_[b]) std::swap(a, b);
        parent_[b] = a;
        size_[a] += size_[b];
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> size_;
};

}  // namespace

StructureResult detect_structure(const Problem& problem, const StructureOptions& options) {
    StructureResult result;
    const Idx n = problem.num_cols();
    const Idx m = problem.num_rows();
    const CsrMatrix& csr = problem.matrix();

    result.col_block.assign(static_cast<std::size_t>(n), Idx{-1});
    result.row_block.assign(static_cast<std::size_t>(m), Idx{-1});

    if (m == 0 || n == 0) {
        result.message = "empty problem (no rows or columns)";
        return result;
    }

    // -- row degrees, and the median-based linking-candidate threshold -----
    std::vector<Idx> degree(static_cast<std::size_t>(m));
    for (Idx i = 0; i < m; ++i) {
        const auto ri = static_cast<std::size_t>(i);
        degree[ri] = csr.row_ptr()[ri + 1] - csr.row_ptr()[ri];
    }
    std::vector<Idx> sorted_degree = degree;
    std::sort(sorted_degree.begin(), sorted_degree.end());
    const Idx median_degree = sorted_degree[sorted_degree.size() / 2];
    const Idx threshold = std::max(options.min_degree_threshold,
                                   static_cast<Idx>(options.degree_threshold_factor * median_degree));

    // -- union-find over columns, using only LOCAL-CANDIDATE rows -----------
    DisjointSet dsu(static_cast<std::size_t>(n));
    for (Idx i = 0; i < m; ++i) {
        if (degree[static_cast<std::size_t>(i)] > threshold) continue;   // linking candidate: skip
        const auto begin = csr.row_ptr()[static_cast<std::size_t>(i)];
        const auto end = csr.row_ptr()[static_cast<std::size_t>(i) + 1];
        if (end <= begin) continue;
        const Idx first_col = csr.col_idx()[static_cast<std::size_t>(begin)];
        for (Idx k = begin + 1; k < end; ++k)
            dsu.unite(static_cast<std::size_t>(first_col),
                     static_cast<std::size_t>(csr.col_idx()[static_cast<std::size_t>(k)]));
    }

    // -- assign compact block ids to every column that appears in a row ----
    // A column with no coefficients anywhere is not part of any block and
    // is left at -1 (presolve's EmptyColumnMove is the real answer for
    // those -- this classifier runs on whatever it is handed). The CSC view
    // is what makes "does this column have any coefficient" a direct check.
    std::vector<Idx> root_to_block(static_cast<std::size_t>(n), -1);
    Idx next_block = 0;
    const CscMatrix csc = csr.to_csc();
    for (Idx j = 0; j < n; ++j) {
        const auto jc = static_cast<std::size_t>(j);
        if (csc.col_ptr()[jc + 1] <= csc.col_ptr()[jc]) continue;   // empty column
        const std::size_t root = dsu.find(jc);
        if (root_to_block[root] < 0) root_to_block[root] = next_block++;
        result.col_block[jc] = root_to_block[root];
    }
    result.num_blocks = next_block;

    // -- reclassify every row by how many distinct blocks it touches -------
    std::vector<Idx> touched_blocks;   // scratch, reused per row
    for (Idx i = 0; i < m; ++i) {
        const auto ri = static_cast<std::size_t>(i);
        const auto begin = csr.row_ptr()[ri];
        const auto end = csr.row_ptr()[ri + 1];
        touched_blocks.clear();
        for (Idx k = begin; k < end; ++k) {
            const Idx j = csr.col_idx()[static_cast<std::size_t>(k)];
            const Idx b = result.col_block[static_cast<std::size_t>(j)];
            if (b >= 0) touched_blocks.push_back(b);
        }
        std::sort(touched_blocks.begin(), touched_blocks.end());
        touched_blocks.erase(std::unique(touched_blocks.begin(), touched_blocks.end()),
                             touched_blocks.end());

        if (touched_blocks.size() == 1) {
            result.row_block[ri] = touched_blocks.front();
        } else if (touched_blocks.size() > 1) {
            result.linking_rows.push_back(i);
        }
        // touched_blocks.empty(): an empty row, or one touching only
        // already-empty columns -- neither local nor linking; row_block
        // stays -1, and it is not counted toward the linking fraction below.
    }

    // -- decide the verdict --------------------------------------------------
    const double linking_fraction =
        static_cast<double>(result.linking_rows.size()) / static_cast<double>(m);

    if (result.num_blocks < options.min_blocks) {
        result.kind = StructureKind::Monolithic;
        result.message = "only " + std::to_string(result.num_blocks)
                        + " block(s) found (need >= " + std::to_string(options.min_blocks) + ")";
    } else if (linking_fraction > options.max_linking_row_fraction) {
        result.kind = StructureKind::Monolithic;
        result.message = std::to_string(result.linking_rows.size()) + "/" + std::to_string(m)
                        + " rows are linking rows (" + std::to_string(linking_fraction * 100)
                        + "%), above the " + std::to_string(options.max_linking_row_fraction * 100)
                        + "% threshold for exploitable structure";
    } else {
        result.kind = StructureKind::BlockAngular;
        result.message = std::to_string(result.num_blocks) + " blocks, "
                        + std::to_string(result.linking_rows.size()) + "/" + std::to_string(m)
                        + " linking rows";
    }
    return result;
}

}  // namespace sov
