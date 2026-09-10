// CPLEX LP reader -- Build Map ticket #5.
//
// The human-writable alternative to MPS. The benchmark corpora are all MPS, so
// this exists for hand-written models and for the Python bindings of #51.
//
// SUPPORTED SUBSET (anything outside it is an error, never a silent
// misreading -- the same rule that governs the MPS reader):
//
//   \ comments
//   Maximize | Minimize          objective, optionally named, with a constant
//   Subject To | such that | st  constraints, optionally named, including the
//                                two-sided form  lo <= expr <= hi
//   Bounds                       x >= v | x <= v | lo <= x <= hi | x = v |
//                                x free | -inf <= x
//   General | Integer            integer variables
//   Binary                       binary variables (bounds forced to [0,1])
//   End
//
// NOT supported, and rejected loudly: SOS sets, indicator constraints,
// quadratic terms ([ ... ] / 2), lazy constraint and user cut sections,
// semi-continuous declarations.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "sovereign/io.hpp"

namespace sov {
namespace {

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

/// One signed term of a linear expression.
struct Term {
    Real coefficient = 0;
    std::string variable;   ///< empty for a bare constant
};

enum class Section { None, Objective, Constraints, Bounds, General, Binary, End };

class LpReader {
public:
    explicit LpReader(std::string origin) : origin_(std::move(origin)) {}
    ReadResult read(std::istream& in);

private:
    [[noreturn]] void error(const std::string& msg) const {
        throw Error(origin_ + ":" + std::to_string(line_no_) + ": " + msg);
    }
    void warn(const std::string& section, const std::string& msg) {
        warnings_.push_back(ReaderWarning{line_no_, section, msg});
    }

    Idx column_for(const std::string& name);
    std::vector<Term> parse_expression(const std::string& text) const;
    void finish_objective();
    void handle_constraint(const std::string& statement);
    void handle_bound(const std::string& statement);
    void declare_integer(const std::string& statement, bool binary);

    /// Split a section's text into statements. LP allows a statement to span
    /// several lines, so the split is on structure, not on newlines: a new
    /// statement starts at a `name:` label or after a completed relation.
    static std::vector<std::string> split_statements(const std::string& text);

    std::string origin_;
    int line_no_ = 0;

    Problem::Builder builder_;
    std::vector<ReaderWarning> warnings_;
    std::string objective_text_;
    std::string objective_name_ = "obj";
    int constraint_counter_ = 0;
};

Idx LpReader::column_for(const std::string& name) {
    Idx c = builder_.find_column(name);
    if (c < 0) c = builder_.add_column(name);
    return c;
}

// --------------------------------------------------------------------------
// expression parsing:  [+|-] [coef] [*] name  |  [+|-] coef
// --------------------------------------------------------------------------

std::vector<Term> LpReader::parse_expression(const std::string& text) const {
    std::vector<Term> terms;
    std::size_t i = 0;
    const std::size_t n = text.size();

    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        if (i >= n) break;

        Real sign = 1.0;
        bool saw_sign = false;
        while (i < n && (text[i] == '+' || text[i] == '-')) {
            if (text[i] == '-') sign = -sign;
            saw_sign = true;
            ++i;
            while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        }
        if (i >= n) {
            if (saw_sign) error("expression ends with a dangling sign");
            break;
        }

        // Optional numeric coefficient.
        Real coefficient = 1.0;
        bool saw_number = false;
        if (std::isdigit(static_cast<unsigned char>(text[i])) || text[i] == '.') {
            const char* begin = text.c_str() + i;
            char* end = nullptr;
            coefficient = std::strtod(begin, &end);
            if (end == begin) error("malformed number in expression");
            i += static_cast<std::size_t>(end - begin);
            saw_number = true;
            while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
            if (i < n && text[i] == '*') {          // "3 * x" is legal
                ++i;
                while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
            }
        }

        // Optional variable name. LP identifiers may contain letters, digits
        // and a set of punctuation, but must not start with a digit.
        const std::size_t name_begin = i;
        while (i < n) {
            const char c = text[i];
            const bool ok = std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.'
                         || c == '[' || c == ']' || c == '(' || c == ')' || c == '#'
                         || c == '$' || c == '&' || c == '\'' || c == '"' || c == '!'
                         || c == '~' || c == '/' || c == ',' || c == ';' || c == '?'
                         || c == '@' || c == '`';
            if (!ok) break;
            ++i;
        }
        const std::string name = text.substr(name_begin, i - name_begin);

        if (name.empty()) {
            if (!saw_number) error("expected a term, found '" + std::string(1, text[i]) + "'");
            terms.push_back(Term{sign * coefficient, ""});   // bare constant
        } else {
            terms.push_back(Term{sign * coefficient, name});
        }
    }
    return terms;
}

// --------------------------------------------------------------------------
// statement splitting
// --------------------------------------------------------------------------

std::vector<std::string> LpReader::split_statements(const std::string& text) {
    // A statement ends where the next one begins. The reliable marker is a
    // `label:` at the start of a token run; failing that, a statement ends once
    // a relational operator has been seen and a new sign-led term would start a
    // fresh line. Splitting on newlines alone breaks multi-line constraints,
    // which real LP files use freely.
    std::vector<std::string> out;
    std::string current;
    std::istringstream in(text);
    std::string line;
    bool seen_relation = false;

    auto flush = [&]() {
        const std::string t = trim(current);
        if (!t.empty()) out.push_back(t);
        current.clear();
        seen_relation = false;
    };

    while (std::getline(in, line)) {
        const std::string t = trim(line);
        if (t.empty()) continue;

        // A label starts a new statement.
        const std::size_t colon = t.find(':');
        const bool has_label = colon != std::string::npos
            && colon > 0 && t.find_first_of("<>=", 0) > colon;

        if ((has_label || seen_relation) && !trim(current).empty()) flush();

        if (!current.empty()) current += ' ';
        current += t;
        if (t.find_first_of("<>=") != std::string::npos) seen_relation = true;
    }
    flush();
    return out;
}

// --------------------------------------------------------------------------
// sections
// --------------------------------------------------------------------------

void LpReader::finish_objective() {
    std::string text = trim(objective_text_);
    const std::size_t colon = text.find(':');
    if (colon != std::string::npos && text.find_first_of("<>=") > colon) {
        objective_name_ = trim(text.substr(0, colon));
        text = text.substr(colon + 1);
    }
    builder_.set_objective_name(objective_name_);

    Real constant = 0.0;
    for (const Term& term : parse_expression(text)) {
        if (term.variable.empty()) {
            constant += term.coefficient;
        } else {
            const Idx c = column_for(term.variable);
            builder_.set_objective_coefficient(c, term.coefficient);
        }
    }
    builder_.set_objective_constant(constant);
}

void LpReader::handle_constraint(const std::string& statement) {
    std::string text = statement;
    std::string name;

    const std::size_t colon = text.find(':');
    if (colon != std::string::npos && text.find_first_of("<>=") > colon) {
        name = trim(text.substr(0, colon));
        text = text.substr(colon + 1);
    }
    if (name.empty()) name = "c" + std::to_string(++constraint_counter_);

    // Locate the relational operators. Two of them means the two-sided form
    // lo <= expr <= hi.
    struct Rel { std::size_t pos; std::size_t len; int dir; };   // dir: -1 <=, 0 =, +1 >=
    std::vector<Rel> rels;
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '<') {
            rels.push_back(Rel{i, (i + 1 < text.size() && text[i + 1] == '=') ? 2u : 1u, -1});
            i += rels.back().len - 1;
        } else if (text[i] == '>') {
            rels.push_back(Rel{i, (i + 1 < text.size() && text[i + 1] == '=') ? 2u : 1u, +1});
            i += rels.back().len - 1;
        } else if (text[i] == '=') {
            rels.push_back(Rel{i, 1u, 0});
        }
    }
    if (rels.empty()) error("constraint '" + name + "' has no relational operator");
    if (rels.size() > 2) error("constraint '" + name + "' has more than two relational operators");

    auto constant_of = [&](const std::string& piece) -> Real {
        const std::vector<Term> terms = parse_expression(piece);
        Real v = 0;
        for (const Term& t : terms) {
            if (!t.variable.empty())
                error("expected a constant, found variable '" + t.variable + "'");
            v += t.coefficient;
        }
        return v;
    };

    Real lo = -kInfinity, hi = kInfinity;
    std::string body;

    if (rels.size() == 1) {
        const Rel& r = rels[0];
        body = text.substr(0, r.pos);
        const Real rhs = constant_of(text.substr(r.pos + r.len));
        if (r.dir < 0) hi = rhs;
        else if (r.dir > 0) lo = rhs;
        else { lo = hi = rhs; }
    } else {
        // lo <= expr <= hi   (or the reversed  hi >= expr >= lo)
        const Rel& a = rels[0];
        const Rel& b = rels[1];
        if (a.dir == 0 || b.dir == 0 || a.dir != b.dir)
            error("constraint '" + name + "' has an inconsistent two-sided form");
        body = text.substr(a.pos + a.len, b.pos - (a.pos + a.len));
        const Real first = constant_of(text.substr(0, a.pos));
        const Real second = constant_of(text.substr(b.pos + b.len));
        if (a.dir < 0) { lo = first; hi = second; }
        else           { hi = first; lo = second; }
    }

    const Idx row = builder_.add_row(name, lo, hi);
    for (const Term& term : parse_expression(body)) {
        if (term.variable.empty()) {
            // A constant on the left moves to the right on both sides.
            if (is_finite_bound(lo)) lo -= term.coefficient;
            if (is_finite_bound(hi)) hi -= term.coefficient;
            builder_.set_row_bounds(row, lo, hi);
            continue;
        }
        builder_.add_coefficient(row, column_for(term.variable), term.coefficient);
    }
}

void LpReader::handle_bound(const std::string& statement) {
    const std::string text = trim(statement);
    const std::string low = lower(text);

    // "x free"
    if (low.size() > 4 && low.compare(low.size() - 4, 4, "free") == 0) {
        const std::string var = trim(text.substr(0, text.size() - 4));
        builder_.set_column_bounds(column_for(var), -kInfinity, kInfinity);
        return;
    }
    // "free x"
    if (low.rfind("free ", 0) == 0) {
        builder_.set_column_bounds(column_for(trim(text.substr(5))), -kInfinity, kInfinity);
        return;
    }

    struct Rel { std::size_t pos; std::size_t len; int dir; };
    std::vector<Rel> rels;
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '<') {
            rels.push_back(Rel{i, (i + 1 < text.size() && text[i + 1] == '=') ? 2u : 1u, -1});
            i += rels.back().len - 1;
        } else if (text[i] == '>') {
            rels.push_back(Rel{i, (i + 1 < text.size() && text[i + 1] == '=') ? 2u : 1u, +1});
            i += rels.back().len - 1;
        } else if (text[i] == '=') {
            rels.push_back(Rel{i, 1u, 0});
        }
    }
    if (rels.empty()) error("bound statement '" + text + "' has no relational operator");

    auto is_number = [&](const std::string& piece, Real* out) {
        const std::string p = trim(piece);
        if (p.empty()) return false;
        const std::string pl = lower(p);
        if (pl == "-inf" || pl == "-infinity") { *out = -kInfinity; return true; }
        if (pl == "+inf" || pl == "inf" || pl == "+infinity" || pl == "infinity") {
            *out = kInfinity; return true;
        }
        const char* begin = p.c_str();
        char* end = nullptr;
        const double v = std::strtod(begin, &end);
        if (end == begin || *end != '\0') return false;
        *out = v;
        return true;
    };

    if (rels.size() == 1) {
        const Rel& r = rels[0];
        const std::string left = trim(text.substr(0, r.pos));
        const std::string right = trim(text.substr(r.pos + r.len));
        Real v = 0;

        if (is_number(right, &v)) {                 // x <= v
            const Idx c = column_for(left);
            if (r.dir < 0)      builder_.set_column_bounds(c, builder_.column_lower(c), v);
            else if (r.dir > 0) builder_.set_column_bounds(c, v, builder_.column_upper(c));
            else                builder_.set_column_bounds(c, v, v);
        } else if (is_number(left, &v)) {           // v <= x
            const Idx c = column_for(right);
            if (r.dir < 0)      builder_.set_column_bounds(c, v, builder_.column_upper(c));
            else if (r.dir > 0) builder_.set_column_bounds(c, builder_.column_lower(c), v);
            else                builder_.set_column_bounds(c, v, v);
        } else {
            error("bound statement '" + text + "' names no constant");
        }
        return;
    }

    // lo <= x <= hi
    const Rel& a = rels[0];
    const Rel& b = rels[1];
    if (a.dir == 0 || b.dir == 0 || a.dir != b.dir)
        error("bound statement '" + text + "' has an inconsistent two-sided form");
    const std::string var = trim(text.substr(a.pos + a.len, b.pos - (a.pos + a.len)));
    Real first = 0, second = 0;
    if (!is_number(trim(text.substr(0, a.pos)), &first)
        || !is_number(trim(text.substr(b.pos + b.len)), &second))
        error("bound statement '" + text + "' has a non-constant side");

    const Idx c = column_for(var);
    if (a.dir < 0) builder_.set_column_bounds(c, first, second);
    else           builder_.set_column_bounds(c, second, first);
}

void LpReader::declare_integer(const std::string& statement, bool binary) {
    std::istringstream in(statement);
    std::string name;
    while (in >> name) {
        if (name == ",") continue;
        while (!name.empty() && (name.back() == ',' || name.back() == ';')) name.pop_back();
        if (name.empty()) continue;
        const Idx c = column_for(name);
        builder_.set_column_kind(c, VarKind::Integer);
        if (binary) builder_.set_column_bounds(c, 0.0, 1.0);
    }
}

// --------------------------------------------------------------------------
// driver
// --------------------------------------------------------------------------

ReadResult LpReader::read(std::istream& in) {
    // Pass 1: strip comments and split into sections, keeping the section text
    // whole so multi-line statements survive.
    std::string line;
    Section section = Section::None;
    std::string constraints_text, bounds_text, general_text, binary_text;
    bool objective_seen = false;

    auto section_of = [&](const std::string& t) -> Section {
        const std::string l = lower(t);
        if (l == "maximize" || l == "maximise" || l == "max") {
            builder_.set_sense(ObjSense::Maximize);
            return Section::Objective;
        }
        if (l == "minimize" || l == "minimise" || l == "min") {
            builder_.set_sense(ObjSense::Minimize);
            return Section::Objective;
        }
        if (l == "subject to" || l == "such that" || l == "st" || l == "s.t."
            || l == "st." || l == "subjectto")
            return Section::Constraints;
        if (l == "bounds" || l == "bound") return Section::Bounds;
        if (l == "general" || l == "generals" || l == "gen"
            || l == "integer" || l == "integers" || l == "int")
            return Section::General;
        if (l == "binary" || l == "binaries" || l == "bin") return Section::Binary;
        if (l == "end") return Section::End;
        if (l == "sos" || l == "semi-continuous" || l == "semis"
            || l == "lazy constraints" || l == "user cuts")
            error("LP section '" + t + "' is not supported");
        return Section::None;
    };

    while (std::getline(in, line)) {
        ++line_no_;
        const std::size_t comment = line.find('\\');
        if (comment != std::string::npos) line = line.substr(0, comment);
        std::string t = trim(line);
        if (t.empty()) continue;

        if (t.find('[') != std::string::npos && t.find(']') != std::string::npos)
            error("quadratic terms are not supported");

        const Section header = section_of(t);
        if (header != Section::None) {
            if (header == Section::End) break;
            section = header;
            if (header == Section::Objective) objective_seen = true;
            continue;
        }

        switch (section) {
            case Section::Objective:   objective_text_ += ' ' + t; break;
            case Section::Constraints: constraints_text += t + '\n'; break;
            case Section::Bounds:      bounds_text += t + '\n'; break;
            case Section::General:     general_text += ' ' + t; break;
            case Section::Binary:      binary_text += ' ' + t; break;
            default:
                error("statement outside any section: '" + t + "'");
        }
    }

    if (!objective_seen)
        throw Error(origin_ + ": no Maximize or Minimize section");

    // Pass 2: constraints first, so every variable exists before Bounds and the
    // integrality sections refer to it.
    finish_objective();
    for (const std::string& s : split_statements(constraints_text)) handle_constraint(s);
    for (const std::string& s : split_statements(bounds_text)) handle_bound(s);
    declare_integer(general_text, /*binary=*/false);
    declare_integer(binary_text, /*binary=*/true);

    ReadResult result;
    result.format = "lp";
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

ReadResult read_lp_string(const std::string& text, const std::string& origin) {
    std::istringstream in(text);
    LpReader reader(origin);
    return reader.read(in);
}

ReadResult read_lp(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw Error("cannot open '" + path + "'");
    LpReader reader(path);
    return reader.read(in);
}

ReadResult read_model(const std::string& path) {
    std::string p = path;
    if (p.size() > 3 && p.compare(p.size() - 3, 3, ".gz") == 0) p = p.substr(0, p.size() - 3);
    const std::string low = lower(p);

    if (low.size() > 4 && low.compare(low.size() - 4, 4, ".mps") == 0) return read_mps(path);
    if (low.size() > 3 && low.compare(low.size() - 3, 3, ".lp") == 0) return read_lp(path);

    // Unknown extension: sniff. An MPS file has a ROWS section; an LP file has
    // a Maximize or Minimize keyword.
    std::ifstream in(path);
    if (!in) throw Error("cannot open '" + path + "'");
    std::string line;
    while (std::getline(in, line)) {
        const std::string l = lower(trim(line));
        if (l.rfind("rows", 0) == 0 || l.rfind("name", 0) == 0) return read_mps(path);
        if (l.rfind("max", 0) == 0 || l.rfind("min", 0) == 0) return read_lp(path);
    }
    throw Error("cannot determine the format of '" + path + "'");
}

}  // namespace sov
