// C entry point for ig_quantile.hpp, so Python can call the C++ quantile on whole arrays via ctypes.
// Built into a shared library by ig_quantile_cpp.py. One call solves n rows.
#include <cstddef>

#include "ig_quantile.hpp"

extern "C" void ig_quantile_survival_array(const double* s, const double* mu, double* x, std::size_t n, int n_iter,
                                           int halley) {
    for (std::size_t i = 0; i < n; ++i) x[i] = ig::quantile_survival(s[i], mu[i], n_iter, halley != 0);
}
