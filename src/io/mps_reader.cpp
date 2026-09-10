// MPS reader -- Build Map ticket #5.
//
// The ticket's warning is the whole design brief: "MPS has real dialect quirks
// (RANGES, BOUNDS types, free rows) -- a parser that silently mishandles one
// produces a *different problem* than the benchmark intends, and your 'wrong
// answer vs oracle' will actually be a parsing bug."
//
// So every quirk below is handled explicitly and commented with the rule it
// implements, and anything genuinely ambiguous across implementations is
// recorded as a ReaderWarning rather than resolved in silence.
//
// FORMAT NOTE, and a lesson learned the hard way. Fixed-column MPS assigns
// meaning to card columns 2-3, 5-12, 15-22, 25-36, 40-47 and 50-61, which
// permits embedded spaces in names. Free MPS tokenizes on whitespace. This
// reader tries free form first, because that is what every modern writer
// emits -- but Netlib's `forplan` really does use the fixed layout, with row
// names like "DEDO3 1R" that free-form tokenization splits in half and turns
// into duplicate names. So a parse failure triggers a retry under fixed-column
// rules, and the fallback is recorded as a warning rather than hidden.
//
// This is precisely the failure the ticket warns about: had the split names
// happened to stay unique, the reader would have built a *different problem*
// and the disagreement would have surfaced much later as a numerical mystery.
//
// The one structural rule shared by both layouts is the useful one: a line
// whose first character is non-blank starts a section, and a line beginning
// with whitespace is a data card.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "sovereign/io.hpp"

namespace sov {
namespace {

// --------------------------------------------------------------------------
// lexing helpers
// --------------------------------------------------------------------------

std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (i >= line.size()) break;
        const std::size_t start = i;
        while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        out.push_back(line.substr(start, i - start));
    }
    return out;
}

/// Fixed-column MPS fields, 1-indexed as the format defines them:
///   f1 = 2-3, f2 = 5-12, f3 = 15-22, f4 = 25-36, f5 = 40-47, f6 = 50-61.
/// Empty fields are dropped, so the resulting token list has the same shape the
/// free-form tokenizer produces for a well-behaved card -- which is what lets
/// both layouts share every section handler.
std::vector<std::string> tokenize_fixed(const std::string& line) {
    static constexpr struct { std::size_t begin, length; } kFields[] = {
        {1, 2}, {4, 8}, {14, 8}, {24, 12}, {39, 8}, {49, 12},
    };
    std::vector<std::string> out;
    for (const auto& f : kFields) {
        if (f.begin >= line.size()) break;
        std::string piece = line.substr(f.begin, std::min(f.length, line.size() - f.begin));
        std::size_t b = 0, e = piece.size();
        while (b < e && std::isspace(static_cast<unsigned char>(piece[b]))) ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(piece[e - 1]))) --e;
        piece = piece.substr(b, e - b);
        if (!piece.empty()) out.push_back(std::move(piece));
    }
    return out;
}

bool parse_real(const std::string& token, Real* out) {
    const char* begin = token.c_str();
    char* end = nullptr;
    const double v = std::strtod(begin, &end);
    if (end == begin || *end != '\0') return false;
    *out = v;
    return true;
}

/// MPS has several spellings for an infinite bound.
bool infinite_token(const std::string& t, Real* out) {
    std::string u = upper(t);
    bool negative = false;
    if (!u.empty() && (u[0] == '+' || u[0] == '-')) {
        negative = (u[0] == '-');
        u.erase(0, 1);
    }
    if (u == "INF" || u == "INFINITY") {
        *out = negative ? -kInfinity : kInfinity;
        return true;
    }
    // 1e30 is the conventional stand-in for infinity in MPS files.
    Real v = 0;
    if (parse_real(t, &v) && std::abs(v) >= 1e30) {
        *out = v > 0 ? kInfinity : -kInfinity;
        return true;
    }
    return false;
}

enum class Section { None, Name, ObjSense, Rows, Columns, Rhs, Ranges, Bounds, Ignored };

// --------------------------------------------------------------------------
// reader
// --------------------------------------------------------------------------

class MpsReader {
public:
    MpsReader(const MpsOptions& opt, std::string origin, bool fixed_format)
        : opt_(opt), origin_(std::move(origin)), fixed_(fixed_format) {}

    ReadResult read(std::istream& in);

private:
    [[noreturn]] void error(const std::string& msg) const {
        throw Error(origin_ + ":" + std::to_string(line_no_) + ": " + msg);
    }
    void warn(const std::string& section, const std::string& msg) {
        warnings_.push_back(ReaderWarning{line_no_, section, msg});
    }

    Real require_real(const std::string& token) const {
        Real v = 0;
        if (!parse_real(token, &v)) error("expected a number, got '" + token + "'");
        return v;
    }
    Real bound_value(const std::string& token) const {
        Real v = 0;
        if (infinite_token(token, &v)) return v;
        return require_real(token);
    }

    void apply_objsense(const std::string& token);
    void handle_rows(const std::vector<std::string>& t);
    void handle_columns(const std::vector<std::string>& t);
    void handle_rhs(const std::vector<std::string>& t);
    void handle_ranges(const std::vector<std::string>& t);
    void handle_bounds(const std::vector<std::string>& t);
    void apply_bound(Idx col, const std::string& type, const std::string* value_token);
    void apply_range(Idx row, Real range);
    Idx column_for(const std::string& name);

    MpsOptions opt_;
    std::string origin_;
    bool fixed_ = false;
    int line_no_ = 0;

    Problem::Builder builder_;
    std::vector<ReaderWarning> warnings_;

    // The objective is the FIRST N row; later N rows are free rows. Treating a
    // second N row as another objective would silently change the problem, so
    // the objective is tracked by name and is never a row of the matrix.
    std::string objective_name_;
    bool objective_seen_ = false;

    // Indexed by BUILDER row index, which skips the objective row (and any
    // dropped free rows). Keeping these in lockstep with the builder rather
    // than with file order is what makes RANGES address the right row.
    std::vector<char> row_sense_;
    std::vector<Real> row_rhs_;

    // Names of free rows dropped when keep_free_rows is false, so COLUMNS
    // cards referring to them can be skipped rather than erroring.
    std::vector<std::string> dropped_rows_;

    bool in_integer_block_ = false;
};

void MpsReader::apply_objsense(const std::string& token) {
    const std::string u = upper(token);
    if (u == "MAX" || u == "MAXIMIZE") {
        builder_.set_sense(ObjSense::Maximize);
    } else if (u == "MIN" || u == "MINIMIZE") {
        builder_.set_sense(ObjSense::Minimize);
    } else {
        error("unknown OBJSENSE '" + token + "'");
    }
}

Idx MpsReader::column_for(const std::string& name) {
    Idx c = builder_.find_column(name);
    if (c < 0) c = builder_.add_column(name);
    return c;
}

void MpsReader::handle_rows(const std::vector<std::string>& t) {
    if (t.size() < 2) error("ROWS card needs a type and a name");
    const std::string type = upper(t[0]);
    const std::string& name = t[1];

    auto record = [&](char sense, Real lo, Real hi) {
        builder_.add_row(name, lo, hi);
        row_sense_.push_back(sense);
        row_rhs_.push_back(0.0);
    };

    if (type == "N") {
        if (!objective_seen_) {
            // First N row is the objective. It is not a constraint, so it never
            // becomes a row of A.
            objective_seen_ = true;
            objective_name_ = name;
            builder_.set_objective_name(name);
            return;
        }
        if (!opt_.keep_free_rows) {
            warn("ROWS", "dropping free row '" + name + "' (keep_free_rows = false)");
            dropped_rows_.push_back(name);
            return;
        }
        record('N', -kInfinity, kInfinity);
    } else if (type == "L") {
        record('L', -kInfinity, 0.0);
    } else if (type == "G") {
        record('G', 0.0, kInfinity);
    } else if (type == "E") {
        record('E', 0.0, 0.0);
    } else {
        error("unknown row type '" + t[0] + "'");
    }
}

void MpsReader::handle_columns(const std::vector<std::string>& t) {
    // Integer markers:  <field1>  'MARKER'  'INTORG' | 'INTEND'
    // The quoted tokens can sit in different fields depending on the writer,
    // so detect by content rather than position.
    bool has_marker = false, intorg = false, intend = false;
    for (const std::string& tok : t) {
        std::string u = upper(tok);
        u.erase(std::remove(u.begin(), u.end(), '\''), u.end());
        if (u == "MARKER") has_marker = true;
        else if (u == "INTORG") intorg = true;
        else if (u == "INTEND") intend = true;
    }
    if (has_marker && (intorg || intend)) {
        in_integer_block_ = intorg;
        return;
    }

    if (t.size() < 3) error("COLUMNS card needs a column, a row and a value");
    const Idx col = column_for(t[0]);
    if (in_integer_block_) {
        // A column inside INTORG/INTEND is integral. Note we do NOT impose
        // [0,1] here: some writers mean binary, but the literal default is
        // [0, +inf), and an instance that means binary says so in BOUNDS.
        // Guessing would cut off the optimum on every general-integer model.
        builder_.set_column_kind(col, VarKind::Integer);
    }

    for (std::size_t k = 1; k + 1 < t.size(); k += 2) {
        const std::string& row_name = t[k];
        const Real value = require_real(t[k + 1]);

        if (row_name == objective_name_) {
            builder_.set_objective_coefficient(col, value);
            continue;
        }
        const Idx r = builder_.find_row(row_name);
        if (r < 0) {
            if (std::find(dropped_rows_.begin(), dropped_rows_.end(), row_name)
                != dropped_rows_.end())
                continue;   // coefficient on a free row we chose to drop
            error("unknown row '" + row_name + "' in COLUMNS");
        }
        builder_.add_coefficient(r, col, value);
    }
}

void MpsReader::handle_rhs(const std::vector<std::string>& t) {
    // Cards read  [SETNAME]  ROW  VALUE  [ROW  VALUE].
    // The set name is optional in practice, so parity decides: an odd token
    // count means the leading token is the set name.
    if (t.size() < 2) error("RHS card needs a row and a value");
    std::size_t k = (t.size() % 2 == 1) ? 1 : 0;

    for (; k + 1 < t.size(); k += 2) {
        const std::string& row_name = t[k];
        const Real value = require_real(t[k + 1]);

        if (row_name == objective_name_) {
            // QUIRK: an RHS entry on the objective row gives the NEGATIVE of
            // the objective's constant term. A sign error here shifts every
            // reported objective by 2d and looks exactly like a numerical bug.
            builder_.set_objective_constant(-value);
            continue;
        }

        const Idx r = builder_.find_row(row_name);
        if (r < 0) {
            if (std::find(dropped_rows_.begin(), dropped_rows_.end(), row_name)
                != dropped_rows_.end())
                continue;
            error("unknown row '" + row_name + "' in RHS");
        }

        const auto k_row = static_cast<std::size_t>(r);
        row_rhs_[k_row] = value;
        switch (row_sense_[k_row]) {
            case 'L': builder_.set_row_bounds(r, -kInfinity, value); break;
            case 'G': builder_.set_row_bounds(r, value, kInfinity); break;
            case 'E': builder_.set_row_bounds(r, value, value); break;
            case 'N': warn("RHS", "RHS given for free row '" + row_name + "'; ignored"); break;
            default: break;
        }
    }
}

void MpsReader::apply_range(Idx row, Real range) {
    // QUIRK: RANGES semantics are asymmetric and, for E rows, sign-dependent.
    // This table is the one every MPS reader has to get exactly right:
    //
    //   row type   sign of R     resulting  lo .. hi
    //   --------   ---------     ---------------------
    //     L        any           rhs - |R| .. rhs
    //     G        any           rhs       .. rhs + |R|
    //     E        R >= 0        rhs       .. rhs + R
    //     E        R <  0        rhs + R   .. rhs
    //
    // L and G use |R| while E uses the signed value. Treating E like the others
    // silently flips which side of an equality gets relaxed.
    const auto k = static_cast<std::size_t>(row);
    const Real rhs = row_rhs_[k];
    const Real mag = std::abs(range);

    switch (row_sense_[k]) {
        case 'L': builder_.set_row_bounds(row, rhs - mag, rhs); break;
        case 'G': builder_.set_row_bounds(row, rhs, rhs + mag); break;
        case 'E':
            if (range >= 0) builder_.set_row_bounds(row, rhs, rhs + range);
            else            builder_.set_row_bounds(row, rhs + range, rhs);
            break;
        case 'N': warn("RANGES", "range given for a free row; ignored"); break;
        default: break;
    }
}

void MpsReader::handle_ranges(const std::vector<std::string>& t) {
    if (t.size() < 2) error("RANGES card needs a row and a value");
    std::size_t k = (t.size() % 2 == 1) ? 1 : 0;
    for (; k + 1 < t.size(); k += 2) {
        const Idx r = builder_.find_row(t[k]);
        if (r < 0) {
            if (std::find(dropped_rows_.begin(), dropped_rows_.end(), t[k])
                != dropped_rows_.end())
                continue;
            error("unknown row '" + t[k] + "' in RANGES");
        }
        apply_range(r, require_real(t[k + 1]));
    }
}

void MpsReader::handle_bounds(const std::vector<std::string>& t) {
    // Cards read  TYPE  [SETNAME]  COLUMN  [VALUE].
    // FR/MI/PL/BV take no value and the set name is optional, so the column is
    // located from the end rather than by a fixed field position.
    if (t.size() < 2) error("BOUNDS card needs a type and a column");
    const std::string type = upper(t[0]);
    const bool valueless = (type == "FR" || type == "MI" || type == "PL" || type == "BV");

    std::size_t col_pos = 0;
    if (valueless) {
        col_pos = t.size() - 1;
    } else {
        if (t.size() < 3) error("BOUNDS card of type " + type + " needs a value");
        col_pos = t.size() - 2;
    }

    const std::string& col_name = t[col_pos];
    Idx col = builder_.find_column(col_name);
    if (col < 0) {
        // A bound on a column that never appeared in COLUMNS: legal (the column
        // is structurally all-zero) but nearly always a typo, so it is recorded.
        warn("BOUNDS", "bound on column '" + col_name + "' which has no coefficients");
        col = builder_.add_column(col_name);
    }

    const std::string* value_token = valueless ? nullptr : &t[col_pos + 1];
    apply_bound(col, type, value_token);
}

void MpsReader::apply_bound(Idx col, const std::string& type, const std::string* value_token) {
    const Real lo = builder_.column_lower(col);
    const Real hi = builder_.column_upper(col);
    auto value = [&]() -> Real {
        if (value_token == nullptr) error("bound type " + type + " needs a value");
        return bound_value(*value_token);
    };

    if (type == "UP") {
        const Real v = value();
        builder_.set_column_bounds(col, lo, v);
        // QUIRK, and the only genuinely ambiguous rule in MPS. An upper bound
        // below the default lower bound of zero is read either as (a) an
        // implicit declaration that the variable is free below, or (b) a
        // crossed, infeasible bound pair. The standard does not say; the
        // established commercial and open-source readers all take (a). We
        // follow them by default so instances agree with the oracle, and
        // record the choice rather than hide it. MpsOptions can flip it.
        if (v < 0.0 && lo == 0.0) {
            if (opt_.negative_upper_bound == NegativeUpperBoundRule::LowerBecomesMinusInfinity) {
                builder_.set_column_bounds(col, -kInfinity, v);
                warn("BOUNDS", "negative UP bound on a column with the default lower bound 0; "
                               "lower bound set to -infinity (the established-solver "
                               "convention; see MpsOptions::negative_upper_bound)");
            } else {
                warn("BOUNDS", "negative UP bound leaves crossed bounds (literal reading)");
            }
        }
    } else if (type == "LO") {
        builder_.set_column_bounds(col, value(), hi);
    } else if (type == "FX") {
        const Real v = value();
        builder_.set_column_bounds(col, v, v);
    } else if (type == "FR") {
        builder_.set_column_bounds(col, -kInfinity, kInfinity);
    } else if (type == "MI") {
        builder_.set_column_bounds(col, -kInfinity, hi);
    } else if (type == "PL") {
        builder_.set_column_bounds(col, lo, kInfinity);
    } else if (type == "BV") {
        builder_.set_column_bounds(col, 0.0, 1.0);
        builder_.set_column_kind(col, VarKind::Integer);
    } else if (type == "LI") {
        builder_.set_column_bounds(col, value(), hi);
        builder_.set_column_kind(col, VarKind::Integer);
    } else if (type == "UI") {
        builder_.set_column_bounds(col, lo, value());
        builder_.set_column_kind(col, VarKind::Integer);
    } else if (type == "SC") {
        // Semi-continuous: x = 0 or l <= x <= u. Not representable in the
        // current model; pretending otherwise would change the problem.
        error("semi-continuous bounds (SC) are not supported");
    } else {
        error("unknown bound type '" + type + "'");
    }
}

ReadResult MpsReader::read(std::istream& in) {
    Section section = Section::None;
    std::string line;

    while (std::getline(in, line)) {
        ++line_no_;
        if (!line.empty() && line.back() == '\r') line.pop_back();   // CRLF files
        if (line.empty()) continue;
        if (line[0] == '*') continue;                                // comment card

        const bool is_header = !std::isspace(static_cast<unsigned char>(line[0]));
        // Section headers always start in column 1 and never carry embedded
        // spaces, so they tokenize the same way under either layout.
        const std::vector<std::string> t =
            (fixed_ && !is_header) ? tokenize_fixed(line) : tokenize(line);
        if (t.empty()) continue;

        if (is_header) {
            const std::string head = upper(t[0]);
            if (head == "NAME") {
                builder_.set_name(t.size() > 1 ? t[1] : "");
                section = Section::Name;
            } else if (head == "OBJSENSE" || head == "OBJSENS") {
                // Either "OBJSENSE MAX" on one line, or the keyword alone with
                // the direction on the following data card.
                if (t.size() > 1) {
                    apply_objsense(t[1]);
                    section = Section::None;
                } else {
                    section = Section::ObjSense;
                }
            } else if (head == "ROWS") {
                section = Section::Rows;
            } else if (head == "COLUMNS") {
                section = Section::Columns;
            } else if (head == "RHS") {
                section = Section::Rhs;
            } else if (head == "RANGES") {
                section = Section::Ranges;
            } else if (head == "BOUNDS") {
                section = Section::Bounds;
            } else if (head == "ENDATA") {
                break;
            } else {
                // QSECTION (quadratic terms), SOS sets, and other extensions.
                // Skipping them silently would mean solving a different
                // problem, so the omission is recorded.
                warn(head, "unsupported section '" + head + "' ignored");
                section = Section::Ignored;
            }
            continue;
        }

        switch (section) {
            case Section::ObjSense: apply_objsense(t[0]); section = Section::None; break;
            case Section::Rows:     handle_rows(t); break;
            case Section::Columns:  handle_columns(t); break;
            case Section::Rhs:      handle_rhs(t); break;
            case Section::Ranges:   handle_ranges(t); break;
            case Section::Bounds:   handle_bounds(t); break;
            case Section::Ignored:  break;
            case Section::Name:     break;
            default:                error("data card outside any section");
        }
    }

    if (!objective_seen_)
        throw Error(origin_ + ": no objective row (no N row in the ROWS section)");

    ReadResult result;
    result.format = "mps";
    result.problem = builder_.finish();
    result.warnings = std::move(warnings_);

    std::string why;
    if (result.problem.validate(&why) != Status::Ok)
        throw Error(origin_ + ": model failed validation after parse: " + why);
    return result;
}

}  // namespace

// --------------------------------------------------------------------------
// entry points
// --------------------------------------------------------------------------

ReadResult read_mps_string(const std::string& text, const MpsOptions& options,
                           const std::string& origin) {
    // Free form first: it is what every modern writer emits. A structural
    // failure is the signal that the file uses the fixed column layout, where
    // names may contain spaces -- Netlib's `forplan` is the case in point.
    try {
        std::istringstream in(text);
        MpsReader reader(options, origin, /*fixed_format=*/false);
        return reader.read(in);
    } catch (const Error& free_form_error) {
        try {
            std::istringstream in(text);
            MpsReader reader(options, origin, /*fixed_format=*/true);
            ReadResult result = reader.read(in);
            result.warnings.insert(
                result.warnings.begin(),
                ReaderWarning{0, "FORMAT",
                              std::string("free-form parse failed (") + free_form_error.what()
                              + "); re-read under fixed-column MPS rules"});
            return result;
        } catch (const Error&) {
            // Report the free-form failure: for a file that is not actually
            // fixed-format, that is the message that names the real problem.
            throw free_form_error;
        }
    }
}

ReadResult read_mps(const std::string& path, const MpsOptions& options) {
    std::ifstream in(path);
    if (!in) throw Error("cannot open '" + path + "'");
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return read_mps_string(buffer.str(), options, path);
}

}  // namespace sov
