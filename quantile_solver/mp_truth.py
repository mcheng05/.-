"""High-precision (50 digit) inverse-Gaussian quantile, the ground truth the double-precision code is judged against.

Used by test_ig_quantile.py and gen_reference.py. Slow (about 0.1 s per row), so it covers samples and edge rows only.
"""

import mpmath as mp

mp.mp.dps = 50


def survival(x, mu):
    a, b = mp.sqrt(x) / mu, 1 / mp.sqrt(x)
    return mp.ncdf(b - a) - mp.exp(2 / mu) * mp.ncdf(-a - b)


def quantile(s, mu, x_start):
    """Root of S_IG(x; mu, 1) = s. x_start (a double-precision solution) only seeds the polish; s and mu are exact."""
    s, mu = mp.mpf(float(s)), mp.mpf(float(mu))
    return mp.findroot(lambda x: mp.log(survival(x, mu)) - mp.log(s), mp.mpf(float(x_start)))
