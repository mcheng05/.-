// Regression test for the table lookup at the edges of its domain. Build with sanitizers so an
// out-of-bounds read fails loudly instead of returning a plausible number:
//   clang++ -O1 -g -fsanitize=address,undefined -std=c++17 test_table_edges.cpp -o test_table_edges && ./test_table_edges
#include <cmath>
#include <cstdio>

#include "ig_quantile.hpp"

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

int main() {
    // s one ulp below e^-2: (w - W0) / h rounds up to the last cell boundary.
    const double w_top = std::nextafter(kTableW1, -INFINITY);
    const double s_top = std::exp(w_top);
    check(std::log(s_top) == w_top, "ln(exp(w)) round-trips for the edge input");
    for (const double mu : {10.0, std::exp(std::nextafter(kTableU1, 0.0))}) {
        const double x = ig::quantile_survival(s_top, mu);
        const double resid = std::fabs(ig::survival(x, mu) / s_top - 1.0);
        check(std::isfinite(x) && resid < 1e-6, "s just below the top of the table solves without an out-of-bounds read");
    }
    // Interior points on the first and last cell still use the table.
    check(ig::in_table(kTableU0, kTableW0), "lower corner is in the table");
    check(ig::in_table(std::nextafter(kTableU1 - kTableHu, 0.0), kTableW1 - 2 * kTableHw), "last full cell is in the table");
    check(!ig::in_table(kTableU1, -10.0) && !ig::in_table(10.0, kTableW1) && !ig::in_table(NAN, -10.0) &&
              !ig::in_table(INFINITY, -10.0),
          "upper bounds, NaN and inf are rejected");
    return failures == 0 ? 0 : 1;
}
