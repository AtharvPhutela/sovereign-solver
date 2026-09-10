// Command-line front end -- Build Map tickets #4 and #5 (and the seed of #50).
//
//   sovereign-cli info  <model>   parse and report the model's shape
//   sovereign-cli solve <model>   solve it with the from-scratch simplex
//
// `info` exists because ticket #5's pass condition is that the parser
// "reproduces instance dimensions and structure identically to the oracle" --
// which needs the shape emitted in a form a harness can diff. `--json` does
// exactly that.
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "sovereign/io.hpp"
#include "sovereign/scaling.hpp"
#include "sovereign/simplex.hpp"

namespace {

int usage() {
    std::fprintf(stderr,
        "usage: sovereign-cli <command> [options] <model>\n"
        "\n"
        "commands:\n"
        "  info   <model>   parse and report rows, columns, nonzeros, sense\n"
        "  solve  <model>   solve the LP relaxation with the revised simplex\n"
        "\n"
        "options:\n"
        "  --json                   machine-readable output\n"
        "  --warnings               print reader warnings\n"
        "  --max-iterations N       simplex iteration limit (default 1000000)\n"
        "  --time-limit S           simplex time limit in seconds\n"
        "  --refactor N             refactorize every N pivots (default 100)\n"
        "  --bland                  force Bland's rule from the first iteration\n"
        "  --scale                  Ruiz + Pock-Chambolle preconditioning before solving\n"
        "  --verbose                per-phase progress\n");
    return 2;
}

std::string escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

int command_info(const std::string& path, bool json, bool show_warnings) {
    const sov::ReadResult r = sov::read_model(path);
    const sov::Problem& p = r.problem;

    if (json) {
        std::printf("{\n");
        std::printf("  \"path\": \"%s\",\n", escape(path).c_str());
        std::printf("  \"format\": \"%s\",\n", r.format.c_str());
        std::printf("  \"name\": \"%s\",\n", escape(p.name()).c_str());
        std::printf("  \"rows\": %lld,\n", static_cast<long long>(p.num_rows()));
        std::printf("  \"columns\": %lld,\n", static_cast<long long>(p.num_cols()));
        std::printf("  \"nonzeros\": %lld,\n", static_cast<long long>(p.num_nonzeros()));
        std::printf("  \"sense\": \"%s\",\n", sov::to_string(p.sense()));
        std::printf("  \"objective_constant\": %.17g,\n", p.objective_constant());
        std::printf("  \"integer_columns\": %lld,\n",
                    static_cast<long long>(p.num_integer_columns()));
        std::printf("  \"free_rows\": %lld,\n", static_cast<long long>(p.num_free_rows()));

        long long eq = 0, le = 0, ge = 0, rng = 0;
        for (sov::Idx i = 0; i < p.num_rows(); ++i) {
            switch (p.row_type(i)) {
                case sov::RowType::Equality:     ++eq; break;
                case sov::RowType::LessEqual:    ++le; break;
                case sov::RowType::GreaterEqual: ++ge; break;
                case sov::RowType::Range:        ++rng; break;
                default: break;
            }
        }
        std::printf("  \"equality_rows\": %lld,\n", eq);
        std::printf("  \"less_equal_rows\": %lld,\n", le);
        std::printf("  \"greater_equal_rows\": %lld,\n", ge);
        std::printf("  \"range_rows\": %lld,\n", rng);

        const auto ext = p.matrix().coefficient_extremes();
        std::printf("  \"max_abs_coefficient\": %.17g,\n", ext.max_abs);
        std::printf("  \"min_abs_coefficient\": %.17g,\n", ext.min_abs_nonzero);
        std::printf("  \"warnings\": [");
        for (std::size_t i = 0; i < r.warnings.size(); ++i) {
            std::printf("%s\n    {\"line\": %d, \"section\": \"%s\", \"message\": \"%s\"}",
                        i ? "," : "", r.warnings[i].line,
                        escape(r.warnings[i].section).c_str(),
                        escape(r.warnings[i].message).c_str());
        }
        std::printf("%s]\n}\n", r.warnings.empty() ? "" : "\n  ");
    } else {
        std::printf("%s\n", p.summary().c_str());
        const auto ext = p.matrix().coefficient_extremes();
        std::printf("  coefficients: |a| in [%.6g, %.6g]", ext.min_abs_nonzero, ext.max_abs);
        if (ext.min_abs_nonzero > 0)
            std::printf(", spread %.3g", ext.max_abs / ext.min_abs_nonzero);
        std::printf("\n  warnings: %zu\n", r.warnings.size());
    }

    if (show_warnings && !json) {
        for (const sov::ReaderWarning& w : r.warnings)
            std::printf("  [line %d] %s: %s\n", w.line, w.section.c_str(), w.message.c_str());
    }
    return 0;
}

int command_solve(const std::string& path, bool json, const sov::SimplexOptions& opt,
                  bool scale) {
    const sov::ReadResult r = sov::read_model(path);

    // With --scale the problem is Ruiz + Pock-Chambolle equilibrated (#6), the
    // scaled problem is solved, and the solution is mapped back. The reported
    // objective is invariant under the mapping.
    sov::Problem scaled_storage;
    sov::Scaling scaling;
    const sov::Problem* solve_target = &r.problem;
    if (scale) {
        auto pair = sov::Scaling::equilibrate(r.problem);
        scaled_storage = std::move(pair.first);
        scaling = std::move(pair.second);
        solve_target = &scaled_storage;
    }
    const sov::Problem& p = r.problem;

    sov::Simplex simplex(opt);
    sov::SimplexResult result = simplex.solve(*solve_target);
    if (scale && !result.primal.empty()) {
        scaling.unscale_primal(result.primal);
        if (!result.dual.empty()) scaling.unscale_dual(result.dual);
        result.objective = p.evaluate_objective(result.primal);
    }

    if (json) {
        std::printf("{\n");
        std::printf("  \"path\": \"%s\",\n", escape(path).c_str());
        std::printf("  \"status\": \"%s\",\n", sov::to_string(result.status));
        if (result.status == sov::SolveStatus::Optimal)
            std::printf("  \"objective\": %.17g,\n", result.objective);
        else
            std::printf("  \"objective\": null,\n");
        std::printf("  \"iterations\": %lld,\n", static_cast<long long>(result.iterations));
        std::printf("  \"phase1_iterations\": %lld,\n",
                    static_cast<long long>(result.phase1_iterations));
        std::printf("  \"refactorizations\": %lld,\n",
                    static_cast<long long>(result.refactorizations));
        std::printf("  \"bland_switches\": %lld,\n",
                    static_cast<long long>(result.bland_switches));
        std::printf("  \"seconds\": %.6f,\n", result.seconds);
        std::printf("  \"primal_infeasibility\": %.17g,\n", result.primal_infeasibility);
        std::printf("  \"message\": \"%s\"\n", escape(result.message).c_str());
        std::printf("}\n");
    } else {
        std::printf("%s\n", p.summary().c_str());
        std::printf("status      : %s\n", sov::to_string(result.status));
        if (result.status == sov::SolveStatus::Optimal)
            std::printf("objective   : %.12g\n", result.objective);
        std::printf("iterations  : %lld (phase 1: %lld)\n",
                    static_cast<long long>(result.iterations),
                    static_cast<long long>(result.phase1_iterations));
        std::printf("time        : %.3f s\n", result.seconds);
        if (result.bland_switches > 0)
            std::printf("anti-cycling: Bland's rule engaged %lld time(s)\n",
                        static_cast<long long>(result.bland_switches));
        if (!result.message.empty())
            std::printf("note        : %s\n", result.message.c_str());
    }
    return result.status == sov::SolveStatus::Optimal
        || result.status == sov::SolveStatus::Infeasible
        || result.status == sov::SolveStatus::Unbounded ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();

    const std::string command = argv[1];
    std::string path;
    bool json = false, warnings = false, scale = false;
    sov::SimplexOptions opt;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--json") json = true;
        else if (a == "--warnings") warnings = true;
        else if (a == "--verbose") opt.verbose = true;
        else if (a == "--bland") opt.always_bland = true;
        else if (a == "--scale") scale = true;
        else if (a == "--refactor" && i + 1 < argc) opt.refactor_frequency = std::stoi(argv[++i]);
        else if (a == "--max-iterations" && i + 1 < argc) opt.max_iterations = std::stoll(argv[++i]);
        else if (a == "--time-limit" && i + 1 < argc) opt.time_limit_seconds = std::stod(argv[++i]);
        else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return usage(); }
        else path = a;
    }
    if (path.empty()) return usage();

    try {
        if (command == "info") return command_info(path, json, warnings);
        if (command == "solve") return command_solve(path, json, opt, scale);
    } catch (const std::exception& e) {
        if (json) std::printf("{\"error\": \"%s\"}\n", escape(e.what()).c_str());
        else std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "unknown command '%s'\n", command.c_str());
    return usage();
}
