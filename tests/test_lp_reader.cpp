// CPLEX LP reader tests -- Build Map ticket #5.
#include <string>

#include "sovereign/io.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {
constexpr Real kTol = 1e-12;
}

TEST(lp, minimal_model) {
    const char* text = R"(\ a comment
Minimize
 obj: 3 x + 2 y
Subject To
 c1: x + y <= 4
 c2: x - y >= 1
Bounds
 0 <= x <= 10
 y >= 0
End
)";
    const ReadResult r = read_lp_string(text);
    const Problem& p = r.problem;
    CHECK_EQ(p.num_rows(), 2);
    CHECK_EQ(p.num_cols(), 2);
    CHECK(p.sense() == ObjSense::Minimize);
    CHECK_NEAR(p.objective()[0], 3.0, kTol);
    CHECK_NEAR(p.objective()[1], 2.0, kTol);
    CHECK_NEAR(p.col_upper()[0], 10.0, kTol);
    CHECK(p.validate() == Status::Ok);
}

TEST(lp, maximize_sets_the_sense) {
    const char* text = "Maximize\n obj: x\nSubject To\n c1: x <= 5\nEnd\n";
    const ReadResult r = read_lp_string(text);
    CHECK(r.problem.sense() == ObjSense::Maximize);
}

TEST(lp, subject_to_spellings) {
    for (const char* keyword : {"Subject To", "such that", "st", "s.t."}) {
        const std::string text =
            std::string("Minimize\n obj: x\n") + keyword + "\n c1: x >= 2\nEnd\n";
        const ReadResult r = read_lp_string(text);
        CHECK_MSG(r.problem.num_rows() == 1, keyword);
    }
}

TEST(lp, term_forms) {
    // "3x", "3 x", "3 * x", "+x", "-x" must all parse, and an omitted
    // coefficient means 1.
    const char* text = R"(Minimize
 obj: 3x + 4 y + 5 * z - w + v
Subject To
 c1: x + y + z + w + v >= 1
End
)";
    const ReadResult r = read_lp_string(text);
    const Problem& p = r.problem;
    CHECK_EQ(p.num_cols(), 5);
    CHECK_NEAR(p.objective()[0], 3.0, kTol);
    CHECK_NEAR(p.objective()[1], 4.0, kTol);
    CHECK_NEAR(p.objective()[2], 5.0, kTol);
    CHECK_NEAR(p.objective()[3], -1.0, kTol);
    CHECK_NEAR(p.objective()[4], 1.0, kTol);
}

TEST(lp, objective_constant) {
    const char* text = "Minimize\n obj: 2 x + 7\nSubject To\n c1: x >= 1\nEnd\n";
    const ReadResult r = read_lp_string(text);
    CHECK_NEAR(r.problem.objective_constant(), 7.0, kTol);
}

TEST(lp, two_sided_constraint) {
    const char* text = "Minimize\n obj: x\nSubject To\n c1: 3 <= x + y <= 8\nEnd\n";
    const ReadResult r = read_lp_string(text);
    CHECK_NEAR(r.problem.row_lower()[0], 3.0, kTol);
    CHECK_NEAR(r.problem.row_upper()[0], 8.0, kTol);
    CHECK(r.problem.row_type(0) == RowType::Range);
}

TEST(lp, equality_constraint) {
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x + y = 5\nEnd\n";
    const ReadResult r = read_lp_string(text);
    CHECK(r.problem.row_type(0) == RowType::Equality);
    CHECK_NEAR(r.problem.row_lower()[0], 5.0, kTol);
}

TEST(lp, unnamed_constraints_get_generated_names) {
    const char* text = "Minimize\n obj: x\nSubject To\n x >= 1\n x <= 9\nEnd\n";
    const ReadResult r = read_lp_string(text);
    CHECK_EQ(r.problem.num_rows(), 2);
    CHECK(r.problem.row_names()[0] != r.problem.row_names()[1]);
}

TEST(lp, multi_line_constraint) {
    // LP lets a single constraint span lines; splitting on newlines alone
    // would turn this into two malformed statements.
    const char* text = R"(Minimize
 obj: x + y + z
Subject To
 c1: x + y
     + z >= 6
End
)";
    const ReadResult r = read_lp_string(text);
    CHECK_EQ(r.problem.num_rows(), 1);
    CHECK_EQ(r.problem.num_nonzeros(), 3);
}

TEST(lp, bound_forms) {
    const char* text = R"(Minimize
 obj: a + b + c + d
Subject To
 c1: a + b + c + d >= 1
Bounds
 a >= 2
 b <= 7
 -3 <= c <= 3
 d free
End
)";
    const ReadResult r = read_lp_string(text);
    const Problem& p = r.problem;
    CHECK_NEAR(p.col_lower()[0], 2.0, kTol);
    CHECK_NEAR(p.col_upper()[1], 7.0, kTol);
    CHECK_NEAR(p.col_lower()[2], -3.0, kTol);
    CHECK_NEAR(p.col_upper()[2], 3.0, kTol);
    CHECK(!is_finite_bound(p.col_lower()[3]));
    CHECK(!is_finite_bound(p.col_upper()[3]));
}

TEST(lp, reversed_bound_form) {
    const char* text = "Minimize\n obj: x\nSubject To\n c1: x >= 0\nBounds\n 5 >= x >= 1\nEnd\n";
    const ReadResult r = read_lp_string(text);
    CHECK_NEAR(r.problem.col_lower()[0], 1.0, kTol);
    CHECK_NEAR(r.problem.col_upper()[0], 5.0, kTol);
}

TEST(lp, integer_and_binary_sections) {
    const char* text = R"(Minimize
 obj: x + y + z
Subject To
 c1: x + y + z >= 1
General
 x
Binary
 z
End
)";
    const ReadResult r = read_lp_string(text);
    const Problem& p = r.problem;
    CHECK(p.col_kind()[0] == VarKind::Integer);
    CHECK(p.col_kind()[1] == VarKind::Continuous);
    CHECK(p.col_kind()[2] == VarKind::Integer);
    CHECK_NEAR(p.col_upper()[2], 1.0, kTol);      // binary forces [0,1]
    CHECK_EQ(p.num_integer_columns(), 2);
}

TEST(lp, unsupported_constructs_are_refused_not_guessed) {
    // Reading a quadratic objective as if it were linear would silently solve
    // a different problem -- the same failure mode the MPS reader guards.
    CHECK_THROWS(read_lp_string(
        "Minimize\n obj: x + [ x ^ 2 ] / 2\nSubject To\n c1: x >= 1\nEnd\n"));
    CHECK_THROWS(read_lp_string(
        "Minimize\n obj: x\nSubject To\n c1: x >= 1\nSOS\n s1: x:1\nEnd\n"));
}

TEST(lp, missing_objective_section_is_an_error) {
    CHECK_THROWS(read_lp_string("Subject To\n c1: x >= 1\nEnd\n"));
}

TEST(lp, constraint_without_a_relation_is_an_error) {
    CHECK_THROWS(read_lp_string("Minimize\n obj: x\nSubject To\n c1: x + y\nEnd\n"));
}

TEST(lp, agrees_with_the_equivalent_mps_model) {
    // The two readers must reach the same Problem. If they disagree, one of
    // them is wrong and the tests above cannot say which.
    const char* lp = R"(Minimize
 cost: 1 x + 2 y
Subject To
 r1: x + y <= 4
 r2: x - y >= 1
End
)";
    const char* mps = R"(NAME T
ROWS
 N  cost
 L  r1
 G  r2
COLUMNS
    x         cost         1.0   r1           1.0
    x         r2           1.0
    y         cost         2.0   r1           1.0
    y         r2          -1.0
RHS
    RHS       r1           4.0   r2           1.0
ENDATA
)";
    const Problem& a = read_lp_string(lp).problem;
    const Problem& b = read_mps_string(mps).problem;

    CHECK_EQ(a.num_rows(), b.num_rows());
    CHECK_EQ(a.num_cols(), b.num_cols());
    CHECK_EQ(a.num_nonzeros(), b.num_nonzeros());
    for (Idx j = 0; j < a.num_cols(); ++j)
        CHECK_NEAR(a.objective()[j], b.objective()[j], kTol);
    for (Idx i = 0; i < a.num_rows(); ++i) {
        CHECK_MSG(a.row_type(i) == b.row_type(i), "row type mismatch between readers");
        if (is_finite_bound(a.row_upper()[i]))
            CHECK_NEAR(a.row_upper()[i], b.row_upper()[i], kTol);
        if (is_finite_bound(a.row_lower()[i]))
            CHECK_NEAR(a.row_lower()[i], b.row_lower()[i], kTol);
    }
}

TST_MAIN()
