// Hypergraph structure detection tests -- Build Map ticket #14, gate M3.
//
// The Build Map's own Test step: "on a known block-angular instance (e.g. a
// stochastic program with obvious scenario blocks), confirm the partitioner
// recovers the true block structure and emits a clean routing decision."
// The synthetic instances here are constructed with a KNOWN true block
// structure so "recovers" is checkable by construction, not by inspection.
#include "sovereign/io.hpp"
#include "sovereign/structure.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

/// Builds a synthetic scenario-decomposable model: `blocks` independent
/// groups of `cols_per_block` columns, each with `rows_per_block` LOCAL rows
/// (degree 2, coupling two columns within the same block only), plus
/// `linking_rows` rows each touching one column from EVERY block (degree ==
/// blocks) -- the classic stochastic-program shape: local recourse
/// constraints per scenario, a handful of first-stage linking constraints.
Problem build_block_angular(int blocks, int cols_per_block, int rows_per_block, int linking_rows) {
    Problem::Builder b;
    std::vector<std::vector<Idx>> block_cols(static_cast<std::size_t>(blocks));
    for (int blk = 0; blk < blocks; ++blk)
        for (int c = 0; c < cols_per_block; ++c) {
            const Idx col = b.add_column("x" + std::to_string(blk) + "_" + std::to_string(c));
            b.set_column_bounds(col, 0, 10);
            b.set_objective_coefficient(col, 1);
            block_cols[static_cast<std::size_t>(blk)].push_back(col);
        }

    for (int blk = 0; blk < blocks; ++blk) {
        const auto& cols = block_cols[static_cast<std::size_t>(blk)];
        for (int r = 0; r < rows_per_block; ++r) {
            const Idx row = b.add_row("local_" + std::to_string(blk) + "_" + std::to_string(r), 0, 5);
            // Degree 2: couple two columns within this block only.
            b.add_coefficient(row, cols[static_cast<std::size_t>(r % cols.size())], 1.0);
            b.add_coefficient(row, cols[static_cast<std::size_t>((r + 1) % cols.size())], 1.0);
        }
    }

    // Linking rows touch EVERY column of EVERY block -- degree
    // blocks*cols_per_block, deliberately far above local rows' degree 2 so
    // the median-based threshold cleanly separates the two candidate sets
    // regardless of the exact block/row counts a given test picks.
    for (int l = 0; l < linking_rows; ++l) {
        const Idx row = b.add_row("link_" + std::to_string(l), 0, 100);
        for (int blk = 0; blk < blocks; ++blk)
            for (Idx col : block_cols[static_cast<std::size_t>(blk)])
                b.add_coefficient(row, col, 1.0);
    }

    return b.finish();
}

}  // namespace

TEST(structure, recovers_the_true_blocks_on_a_synthetic_scenario_model) {
    const Problem p = build_block_angular(/*blocks=*/4, /*cols_per_block=*/5,
                                          /*rows_per_block=*/4, /*linking_rows=*/2);
    const StructureResult r = detect_structure(p);

    CHECK(r.kind == StructureKind::BlockAngular);
    CHECK_EQ(r.num_blocks, 4);
    CHECK_EQ(r.linking_rows.size(), std::size_t{2});

    // Every column within one synthetic block must land in the SAME
    // detected block, and columns from DIFFERENT synthetic blocks must land
    // in DIFFERENT detected blocks -- the actual "recovers the true
    // structure" claim, checked by construction rather than eyeballed.
    for (Idx blk = 0; blk < 4; ++blk) {
        const Idx first_col = blk * 5;
        const Idx expected_block = r.col_block[static_cast<std::size_t>(first_col)];
        CHECK(expected_block >= 0);
        for (Idx c = 0; c < 5; ++c)
            CHECK_EQ(r.col_block[static_cast<std::size_t>(first_col + c)], expected_block);
        for (Idx other = 0; other < 4; ++other) {
            if (other == blk) continue;
            CHECK(r.col_block[static_cast<std::size_t>(other * 5)] != expected_block);
        }
    }
}

TEST(structure, two_blocks_no_linking_rows_still_recovered) {
    const Problem p = build_block_angular(2, 3, 3, /*linking_rows=*/0);
    const StructureResult r = detect_structure(p);
    CHECK(r.kind == StructureKind::BlockAngular);
    CHECK_EQ(r.num_blocks, 2);
    CHECK(r.linking_rows.empty());
}

TEST(structure, a_single_bridging_row_correctly_merges_two_blocks) {
    // A LOCAL-CANDIDATE row (low degree) that happens to touch columns from
    // what would otherwise be two separate blocks must merge them -- this
    // is genuinely correct (the model really isn't separable there), not a
    // detection failure. Build 2 blocks with no linking rows, then add one
    // more degree-2 row bridging a column from each block.
    Problem::Builder b;
    std::vector<Idx> a_cols, c_cols;
    for (int i = 0; i < 3; ++i) {
        const Idx col = b.add_column("a" + std::to_string(i));
        b.set_column_bounds(col, 0, 10);
        a_cols.push_back(col);
    }
    for (int i = 0; i < 3; ++i) {
        const Idx col = b.add_column("c" + std::to_string(i));
        b.set_column_bounds(col, 0, 10);
        c_cols.push_back(col);
    }
    const Idx r1 = b.add_row("r1", 0, 5); b.add_coefficient(r1, a_cols[0], 1); b.add_coefficient(r1, a_cols[1], 1);
    const Idx r2 = b.add_row("r2", 0, 5); b.add_coefficient(r2, c_cols[0], 1); b.add_coefficient(r2, c_cols[1], 1);
    const Idx bridge = b.add_row("bridge", 0, 5);
    b.add_coefficient(bridge, a_cols[0], 1);
    b.add_coefficient(bridge, c_cols[0], 1);
    (void)r1; (void)r2;
    const Problem p = b.finish();

    const StructureResult r = detect_structure(p);
    // Everything ends up in ONE block -- the bridge row genuinely connects
    // both groups, so this is correctly reported as not decomposable here
    // (too few blocks), not a wrong partition.
    CHECK(r.kind == StructureKind::Monolithic);
    CHECK_EQ(r.col_block[static_cast<std::size_t>(a_cols[0])],
            r.col_block[static_cast<std::size_t>(c_cols[0])]);
}

TEST(structure, an_ordinary_dense_lp_reports_monolithic) {
    // Every column appears in every row -- no exploitable structure exists,
    // and the classifier must say so rather than force a partition.
    const char* text = R"(Minimize
 obj: x + y + z
Subject To
 c1: x + y + z <= 10
 c2: x + y + z >= 1
 c3: x - y + z <= 5
End
)";
    const ReadResult rr = read_lp_string(text);
    const StructureResult r = detect_structure(rr.problem);
    CHECK(r.kind == StructureKind::Monolithic);
}

TEST(structure, empty_problem_is_handled_without_crashing) {
    Problem::Builder b;
    const Problem p = b.finish();
    const StructureResult r = detect_structure(p);
    CHECK(r.kind == StructureKind::Monolithic);
    CHECK(!r.message.empty());
}

TEST(structure, a_column_touching_no_row_is_left_unassigned) {
    Problem::Builder b;
    const Idx x = b.add_column("x"); b.set_column_bounds(x, 0, 10);
    const Idx orphan = b.add_column("orphan"); b.set_column_bounds(orphan, 0, 10);
    const Idx c1 = b.add_row("c1", 0, 5);
    b.add_coefficient(c1, x, 1.0);
    const Problem p = b.finish();

    const StructureResult r = detect_structure(p);
    CHECK(r.col_block[static_cast<std::size_t>(orphan)] == -1);
    CHECK(r.col_block[static_cast<std::size_t>(x)] >= 0);
}

TEST(structure, too_many_linking_rows_reports_monolithic_even_with_multiple_blocks) {
    // 4 blocks are genuinely detected (linking rows stay a clear MINORITY
    // by count, so the median-based threshold still excludes them from the
    // union step correctly) -- but a tight max_linking_row_fraction still
    // says the coupling isn't worth exploiting.
    const Problem p = build_block_angular(/*blocks=*/4, /*cols_per_block=*/3,
                                          /*rows_per_block=*/3, /*linking_rows=*/6);
    StructureOptions opt;
    opt.max_linking_row_fraction = 0.2;   // 6/18 = 33% > 20%
    const StructureResult r = detect_structure(p, opt);
    CHECK(r.kind == StructureKind::Monolithic);
    CHECK(r.num_blocks == 4);
    CHECK(!r.linking_rows.empty());
}

TST_MAIN()
