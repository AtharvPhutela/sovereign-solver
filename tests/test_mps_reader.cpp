// MPS reader tests -- Build Map ticket #5.
//
// Weighted heavily toward the dialect quirks, because those are where a parser
// silently builds a *different problem* than the file describes. A test that
// only checks "it parsed without throwing" would pass on every one of the bugs
// below.
#include <string>

#include "sovereign/io.hpp"
#include "test_support.hpp"

using namespace sov;

namespace {

constexpr Real kTol = 1e-12;

/// min  x + 2y   s.t.  R1: x + y <= 4,  R2: x - y >= 1,  R3: x + 2y = 3
const char* kBasic = R"(NAME          BASIC
ROWS
 N  COST
 L  R1
 G  R2
 E  R3
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0   R3           1.0
    Y         COST         2.0   R1           1.0
    Y         R2          -1.0   R3           2.0
RHS
    RHS       R1           4.0   R2           1.0
    RHS       R3           3.0
ENDATA
)";

const Problem& read(const char* text, ReadResult& holder) {
    holder = read_mps_string(text);
    return holder.problem;
}

}  // namespace

// --------------------------------------------------------------------------
// structure
// --------------------------------------------------------------------------

TEST(mps, basic_shape_and_sense) {
    ReadResult r;
    const Problem& p = read(kBasic, r);
    CHECK_EQ(p.num_rows(), 3);
    CHECK_EQ(p.num_cols(), 2);
    CHECK_EQ(p.num_nonzeros(), 6);
    CHECK(p.sense() == ObjSense::Minimize);
    CHECK(p.name() == "BASIC");
    CHECK(p.objective_name() == "COST");
    CHECK(p.validate() == Status::Ok);
}

TEST(mps, objective_row_is_not_a_constraint) {
    // The N row supplies the objective and must never become a row of A.
    // Counting it as a constraint is a classic off-by-one that makes every
    // downstream dimension wrong.
    ReadResult r;
    const Problem& p = read(kBasic, r);
    CHECK_EQ(p.num_rows(), 3);
    CHECK_NEAR(p.objective()[0], 1.0, kTol);
    CHECK_NEAR(p.objective()[1], 2.0, kTol);
}

TEST(mps, row_senses_become_bounds) {
    ReadResult r;
    const Problem& p = read(kBasic, r);
    CHECK(p.row_type(0) == RowType::LessEqual);
    CHECK(p.row_type(1) == RowType::GreaterEqual);
    CHECK(p.row_type(2) == RowType::Equality);
    CHECK_NEAR(p.row_upper()[0], 4.0, kTol);
    CHECK(!is_finite_bound(p.row_lower()[0]));
    CHECK_NEAR(p.row_lower()[1], 1.0, kTol);
    CHECK(!is_finite_bound(p.row_upper()[1]));
    CHECK_NEAR(p.row_lower()[2], 3.0, kTol);
    CHECK_NEAR(p.row_upper()[2], 3.0, kTol);
}

TEST(mps, default_column_bounds_are_zero_to_infinity) {
    // Most Netlib instances have no BOUNDS section at all, so this default is
    // load-bearing for the majority of the corpus.
    ReadResult r;
    const Problem& p = read(kBasic, r);
    for (Idx j = 0; j < p.num_cols(); ++j) {
        CHECK_NEAR(p.col_lower()[j], 0.0, kTol);
        CHECK(!is_finite_bound(p.col_upper()[j]));
    }
}

TEST(mps, second_n_row_is_a_free_row_not_a_second_objective) {
    const char* text = R"(NAME T
ROWS
 N  COST
 N  FREEROW
 L  R1
COLUMNS
    X         COST         1.0   FREEROW      5.0
    X         R1           1.0
RHS
    RHS       R1           4.0
ENDATA
)";
    ReadResult r;
    const Problem& p = read(text, r);
    CHECK_EQ(p.num_rows(), 2);              // FREEROW is a row, COST is not
    CHECK_EQ(p.num_free_rows(), 1);
    CHECK_NEAR(p.objective()[0], 1.0, kTol);   // only COST feeds the objective
}

TEST(mps, duplicate_coefficients_sum) {
    const char* text = R"(NAME T
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.5
    X         R1           2.5
RHS
    RHS       R1           4.0
ENDATA
)";
    ReadResult r;
    const Problem& p = read(text, r);
    CHECK_EQ(p.num_nonzeros(), 1);
    CHECK_NEAR(p.matrix().values()[0], 4.0, kTol);
}

// --------------------------------------------------------------------------
// the objective constant: an RHS on the objective row is NEGATED
// --------------------------------------------------------------------------

TEST(mps, rhs_on_the_objective_row_is_the_negated_constant) {
    // Getting this sign backwards shifts every reported objective by 2d and
    // looks exactly like a numerical bug rather than a parsing one.
    const char* text = R"(NAME T
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1           4.0   COST        -7.113
ENDATA
)";
    ReadResult r;
    const Problem& p = read(text, r);
    CHECK_NEAR(p.objective_constant(), 7.113, 1e-12);

    const std::vector<Real> x = {2.0};
    CHECK_NEAR(p.evaluate_objective(x), 2.0 + 7.113, 1e-12);
}

// --------------------------------------------------------------------------
// RANGES -- the asymmetric, sign-dependent table
// --------------------------------------------------------------------------

TEST(mps, ranges_on_l_row_use_the_magnitude) {
    // L row, rhs b, range R:  b - |R| <= row <= b
    for (const char* range_value : {"2.0", "-2.0"}) {
        const std::string text = std::string(R"(NAME T
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
RANGES
    RNG       R1           )") + range_value + "\nENDATA\n";
        ReadResult r;
        const Problem& p = read(text.c_str(), r);
        CHECK_NEAR(p.row_lower()[0], 8.0, kTol);
        CHECK_NEAR(p.row_upper()[0], 10.0, kTol);
    }
}

TEST(mps, ranges_on_g_row_use_the_magnitude) {
    // G row, rhs b, range R:  b <= row <= b + |R|
    for (const char* range_value : {"2.0", "-2.0"}) {
        const std::string text = std::string(R"(NAME T
ROWS
 N  COST
 G  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
RANGES
    RNG       R1           )") + range_value + "\nENDATA\n";
        ReadResult r;
        const Problem& p = read(text.c_str(), r);
        CHECK_NEAR(p.row_lower()[0], 10.0, kTol);
        CHECK_NEAR(p.row_upper()[0], 12.0, kTol);
    }
}

TEST(mps, ranges_on_e_row_are_sign_dependent) {
    // THE quirk. E rows use the SIGNED range, unlike L and G which use |R|:
    //   R >= 0  ->  b     .. b + R
    //   R <  0  ->  b + R .. b
    // Treating E like L or G flips which side of the equality is relaxed.
    const char* positive = R"(NAME T
ROWS
 N  COST
 E  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
RANGES
    RNG       R1           3.0
ENDATA
)";
    ReadResult rp;
    const Problem& pp = read(positive, rp);
    CHECK_NEAR(pp.row_lower()[0], 10.0, kTol);
    CHECK_NEAR(pp.row_upper()[0], 13.0, kTol);

    const char* negative = R"(NAME T
ROWS
 N  COST
 E  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
RANGES
    RNG       R1          -3.0
ENDATA
)";
    ReadResult rn;
    const Problem& pn = read(negative, rn);
    CHECK_NEAR(pn.row_lower()[0], 7.0, kTol);
    CHECK_NEAR(pn.row_upper()[0], 10.0, kTol);
}

// --------------------------------------------------------------------------
// BOUNDS
// --------------------------------------------------------------------------

namespace {
std::string with_bounds(const std::string& bounds_section) {
    return std::string(R"(NAME T
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
BOUNDS
)") + bounds_section + "ENDATA\n";
}
}  // namespace

TEST(mps, bound_types) {
    struct Case { const char* card; Real lower; Real upper; bool integer; };
    const Case cases[] = {
        {" UP BND       X            5.0\n", 0.0, 5.0, false},
        {" LO BND       X            2.0\n", 2.0, kInfinity, false},
        {" FX BND       X            3.0\n", 3.0, 3.0, false},
        {" FR BND       X\n", -kInfinity, kInfinity, false},
        {" MI BND       X\n", -kInfinity, kInfinity, false},
        {" BV BND       X\n", 0.0, 1.0, true},
        {" LI BND       X            2.0\n", 2.0, kInfinity, true},
        {" UI BND       X            7.0\n", 0.0, 7.0, true},
    };
    for (const Case& c : cases) {
        ReadResult r;
        const Problem& p = read(with_bounds(c.card).c_str(), r);
        CHECK_MSG(p.col_lower()[0] == c.lower, std::string("lower for card: ") + c.card);
        CHECK_MSG(p.col_upper()[0] == c.upper, std::string("upper for card: ") + c.card);
        CHECK_MSG((p.col_kind()[0] == VarKind::Integer) == c.integer,
                  std::string("integrality for card: ") + c.card);
    }
}

TEST(mps, negative_upper_bound_frees_the_lower_bound_by_default) {
    // The one genuinely ambiguous rule in MPS. The default follows the
    // established readers so instances agree with the oracle -- and the choice
    // is reported as a warning rather than made silently.
    ReadResult r = read_mps_string(with_bounds(" UP BND       X           -5.0\n"));
    CHECK(!is_finite_bound(r.problem.col_lower()[0]));
    CHECK_NEAR(r.problem.col_upper()[0], -5.0, kTol);
    CHECK_MSG(!r.warnings.empty(), "the ambiguous choice must be recorded, not hidden");
}

TEST(mps, negative_upper_bound_literal_reading_is_selectable) {
    MpsOptions opt;
    opt.negative_upper_bound = NegativeUpperBoundRule::KeepLowerAtZero;
    // The literal reading leaves lower=0 > upper=-5, which is a crossed bound
    // pair -- an infeasible model, which validate() rejects. That the strict
    // rule produces a rejected model is exactly why it is not the default.
    CHECK_THROWS(read_mps_string(with_bounds(" UP BND       X           -5.0\n"), opt));
}

TEST(mps, infinite_bound_spellings) {
    for (const char* card : {" UP BND       X            1E30\n",
                             " UP BND       X            INF\n"}) {
        ReadResult r;
        const Problem& p = read(with_bounds(card).c_str(), r);
        CHECK_MSG(!is_finite_bound(p.col_upper()[0]), card);
    }
}

// --------------------------------------------------------------------------
// integrality markers, OBJSENSE, comments
// --------------------------------------------------------------------------

TEST(mps, integer_markers) {
    const char* text = R"(NAME T
ROWS
 N  COST
 L  R1
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    X         COST         1.0   R1           1.0
    MARKER                 'MARKER'                 'INTEND'
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
ENDATA
)";
    ReadResult r;
    const Problem& p = read(text, r);
    CHECK(p.col_kind()[0] == VarKind::Integer);
    CHECK(p.col_kind()[1] == VarKind::Continuous);
    CHECK_EQ(p.num_integer_columns(), 1);
    CHECK(p.is_mip());
    // An INTORG column keeps the literal [0, inf) default. Imposing [0,1] here
    // would cut off the optimum of every general-integer model.
    CHECK(!is_finite_bound(p.col_upper()[0]));
}

TEST(mps, objsense_section_and_inline_form) {
    for (const char* head : {"OBJSENSE\n    MAX\n", "OBJSENSE MAX\n"}) {
        const std::string text = std::string("NAME T\n") + head + R"(ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
ENDATA
)";
        ReadResult r;
        const Problem& p = read(text.c_str(), r);
        CHECK_MSG(p.sense() == ObjSense::Maximize, head);
    }
}

TEST(mps, comment_cards_and_crlf_are_tolerated) {
    const std::string text =
        "* a comment card\r\n"
        "NAME T\r\n"
        "ROWS\r\n"
        " N  COST\r\n"
        " L  R1\r\n"
        "* another comment\r\n"
        "COLUMNS\r\n"
        "    X         COST         1.0   R1           1.0\r\n"
        "RHS\r\n"
        "    RHS       R1          10.0\r\n"
        "ENDATA\r\n";
    ReadResult r;
    const Problem& p = read(text.c_str(), r);
    CHECK_EQ(p.num_rows(), 1);
    CHECK_EQ(p.num_cols(), 1);
}

// --------------------------------------------------------------------------
// fixed-column fallback
// --------------------------------------------------------------------------

namespace {

/// Build a fixed-column MPS card. The format assigns 1-indexed card columns
/// 2-3, 5-12, 15-22, 25-36, 40-47 and 50-61 to the six fields; writing that out
/// by hand is exactly how a "fixed format" fixture ends up not being one, so
/// the padding is computed rather than typed.
std::string fixed_card(const std::string& f1, const std::string& f2 = "",
                       const std::string& f3 = "", const std::string& f4 = "",
                       const std::string& f5 = "", const std::string& f6 = "") {
    static constexpr std::size_t kBegin[] = {1, 4, 14, 24, 39, 49};
    const std::string* fields[] = {&f1, &f2, &f3, &f4, &f5, &f6};
    std::string card;
    for (int i = 0; i < 6; ++i) {
        if (fields[i]->empty()) continue;
        if (card.size() < kBegin[i]) card.resize(kBegin[i], ' ');
        card += *fields[i];
    }
    return card + "\n";
}

}  // namespace

TEST(mps, fixed_column_names_with_embedded_spaces) {
    // Netlib's `forplan` really does this: row names like "DEDO3 1R" that
    // free-form tokenization splits into two, producing duplicate names. The
    // reader must notice the failure and retry under fixed-column rules.
    const std::string text =
        "NAME          T\n"
        "ROWS\n"
        + fixed_card("N", "COST")
        + fixed_card("L", "ROW A")
        + fixed_card("L", "ROW B")
        + "COLUMNS\n"
        + fixed_card("", "X", "COST", "1.0", "ROW A", "1.0")
        + fixed_card("", "X", "ROW B", "2.0")
        + "RHS\n"
        + fixed_card("", "RHS", "ROW A", "10.0", "ROW B", "20.0")
        + "ENDATA\n";

    const ReadResult r = read_mps_string(text);
    CHECK_EQ(r.problem.num_rows(), 2);
    CHECK_EQ(r.problem.num_cols(), 1);
    CHECK_EQ(r.problem.num_nonzeros(), 2);
    CHECK(r.problem.row_names()[0] == "ROW A");
    CHECK(r.problem.row_names()[1] == "ROW B");
    CHECK_NEAR(r.problem.row_upper()[0], 10.0, kTol);
    CHECK_NEAR(r.problem.row_upper()[1], 20.0, kTol);
    CHECK_MSG(!r.warnings.empty(), "the format fallback must be recorded, not silent");
}

TEST(mps, free_form_is_preferred_when_it_works) {
    // The fallback must not fire on ordinary files: a free-form parse that
    // succeeds is the answer, because re-reading a free-form file under
    // fixed-column rules would slice its names at the wrong offsets.
    ReadResult r = read_mps_string(kBasic);
    CHECK(r.warnings.empty());
    CHECK_EQ(r.problem.num_rows(), 3);
}

// --------------------------------------------------------------------------
// malformed input must fail loudly
// --------------------------------------------------------------------------

TEST(mps, malformed_files_are_rejected) {
    CHECK_THROWS(read_mps_string("NAME T\nROWS\n L  R1\nCOLUMNS\nENDATA\n"));   // no N row
    CHECK_THROWS(read_mps_string("NAME T\nROWS\n Q  R1\nENDATA\n"));            // bad row type
    CHECK_THROWS(read_mps_string(
        "NAME T\nROWS\n N  C\n L  R1\nCOLUMNS\n    X   NOSUCHROW   1.0\nENDATA\n"));
    CHECK_THROWS(read_mps_string(
        "NAME T\nROWS\n N  C\n L  R1\nCOLUMNS\n    X   R1   notanumber\nENDATA\n"));
}

TEST(mps, semi_continuous_bounds_are_refused_not_guessed) {
    // SC is not representable in the current model. Reading it as an ordinary
    // bound would silently solve a different problem.
    CHECK_THROWS(read_mps_string(with_bounds(" SC BND       X            5.0\n")));
}

TEST(mps, unsupported_sections_are_reported_rather_than_ignored) {
    const char* text = R"(NAME T
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
QUADOBJ
    X         X            2.0
ENDATA
)";
    ReadResult r = read_mps_string(text);
    bool mentioned = false;
    for (const ReaderWarning& w : r.warnings)
        if (w.section == "QUADOBJ") mentioned = true;
    CHECK_MSG(mentioned, "an ignored section must produce a warning");
}

TST_MAIN()
