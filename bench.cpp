// Harness for ig_quantile.hpp: reads reference.csv (from gen_reference.py), reports ns/row and error.
//
//   clang++ -O3 -march=native -std=c++17 bench.cpp -o bench && ./bench
//
// Error is measured two ways, both on unclipped rows unless noted:
//   rel err vs x_ref      |x / x_ref - 1|, x_ref = the Python module's solve: port fidelity, not accuracy
//   survival residual     |S(x) / s - 1|, which needs no reference at all (all rows, clipped included)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ig_quantile.hpp"

struct Rows {
    std::vector<double> s, mu, x_ref;
    std::vector<int> clipped;
};

static Rows load(const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s (run gen_reference.py first)\n", path);
        std::exit(1);
    }
    Rows r;
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string f[4];
        for (auto& v : f) std::getline(ss, v, ',');
        r.s.push_back(std::stod(f[0]));
        r.mu.push_back(std::stod(f[1]));
        r.x_ref.push_back(std::stod(f[2]));
        r.clipped.push_back(std::stoi(f[3]));
    }
    return r;
}

template <class F>
static double best_seconds(F&& fn, int repeats = 200) {
    double best = 1e30;
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

int main(int argc, char** argv) {
    const Rows r = load(argc > 1 ? argv[1] : "reference.csv");
    const size_t n = r.s.size();
    std::vector<double> x(n);
    std::printf("rows %zu\n", n);

    struct Config {
        int n_iter;
        bool halley;
    };
    for (const Config c : {Config{0, false}, Config{1, false}, Config{2, false}, Config{1, true}, Config{2, true}}) {
        const double sec = best_seconds([&] {
            for (size_t i = 0; i < n; ++i) x[i] = ig::quantile_survival(r.s[i], r.mu[i], c.n_iter, c.halley);
        });
        double max_rel = 0, max_resid = 0;
        size_t in_dom = 0, off_table = 0;
        for (size_t i = 0; i < n; ++i) {
            const double resid = std::fabs(ig::survival(x[i], r.mu[i]) / r.s[i] - 1.0);
            max_resid = std::max(max_resid, resid);
            if (!r.clipped[i]) max_rel = std::max(max_rel, std::fabs(x[i] / r.x_ref[i] - 1.0));
            ig::in_table(std::log(r.mu[i]), std::log(r.s[i])) ? ++in_dom : ++off_table;
        }
        std::printf("table seed + %d %s step(s): %8.3f ms  %6.1f ns/row | rel err vs x_ref (port fidelity) %.1e | survival resid %.1e\n",
                    c.n_iter, c.halley ? "Halley" : "Newton", sec * 1e3, sec / n * 1e9, max_rel, max_resid);
        if (c.n_iter == 0) std::printf("  (%zu rows in table domain, %zu on the fallback path)\n", in_dom, off_table);
    }

    // Seed alone, as a lower bound on what any step count costs.
    volatile double sink = 0;
    const double seed_sec = best_seconds([&] {
        double acc = 0;
        for (size_t i = 0; i < n; ++i) {
            const double u = std::log(r.mu[i]), w = std::log(r.s[i]);
            if (ig::in_table(u, w)) acc += ig::table_seed(u, w);
        }
        sink = acc;
    });
    std::printf("log + table_seed only: %8.3f ms  %6.1f ns/row\n", seed_sec * 1e3, seed_sec / n * 1e9);

    // The analytic seed + 6 steps, for the like-for-like comparison with the Python module.
    const double analytic_sec = best_seconds([&] {
        for (size_t i = 0; i < n; ++i) {
            const double log_s = std::log(r.s[i]), e2m = std::exp(2.0 / r.mu[i]);
            double y = std::log(ig::analytic_seed(r.s[i], r.mu[i]));
            for (int k = 0; k < 6; ++k) y = ig::newton_step(y, log_s, r.mu[i], e2m);
            x[i] = std::exp(y);
        }
    });
    double analytic_rel = 0;
    for (size_t i = 0; i < n; ++i)
        if (!r.clipped[i]) analytic_rel = std::max(analytic_rel, std::fabs(x[i] / r.x_ref[i] - 1.0));
    std::printf("analytic seed + 6 steps: %6.3f ms  %6.1f ns/row | rel err vs x_ref %.1e\n", analytic_sec * 1e3,
                analytic_sec / n * 1e9, analytic_rel);

    // Synthetic edge rows (two blocks off-table, one at the clipped s ~ 1e-14 inside it), judged against 50-digit mpmath roots, so the
    // error here is real error, not agreement with the Python port. Reported per block because the
    // large-mu block is dominated by the survival function's own precision floor.
    const Rows edge = load(argc > 2 ? argv[2] : "edge.csv");
    const char* block_name[3] = {"mu < 1", "mu > e^17", "s ~ 1e-14"};
    double block_max[3] = {0, 0, 0};
    size_t block_n[3] = {0, 0, 0}, block_off_table[3] = {0, 0, 0};
    for (size_t i = 0; i < edge.s.size(); ++i) {
        const double u = std::log(edge.mu[i]), w = std::log(edge.s[i]);
        const int blk = u < 0 ? 0 : (u >= kTableU1 ? 1 : 2);
        const double got = ig::quantile_survival(edge.s[i], edge.mu[i]);
        block_max[blk] = std::max(block_max[blk], std::fabs(got / edge.x_ref[i] - 1.0));
        ++block_n[blk];
        block_off_table[blk] += !ig::in_table(u, w);
    }
    for (int k = 0; k < 3; ++k)
        std::printf("edge %-10s %3zu rows (%3zu off-table) | max rel err vs mpmath %.1e\n", block_name[k], block_n[k],
                    block_off_table[k], block_max[k]);
    return 0;
}
