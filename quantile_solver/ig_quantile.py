"""Inverse-Gaussian quantile for Schadner's explicit implied-volatility formula.

Solves  S_IG(x; mu, 1) = s  for x, where S is the IG(mu, 1) survival function and
s = 1 - p = (eta_k - 1 + c) / eta_k. Everything runs on the *survival* side, in
log space, because this project's real SPX data has p > 0.9 on every row (s is
small, down to the 1e-14 clip) and forming s as 1 - p loses digits (survival_inputs
computes s directly).

Approach (../hand_rolled_seed_halley.ipynb has the earlier attempts and why they failed):
  * seed  x0 = min(x_inf, x_tail)
        x_inf  = 1 / Phi^-1(1/2 + s/2)^2     mu -> inf limit (the notebook's "ATM-limit" seed)
        x_tail = ((z mu + sqrt(z^2 mu^2 + 4 mu)) / 2)^2,  z = Phi^-1(1 - s)
                                              Gaussian-tail limit; ignores the second
                                              survival term so it overshoots, but never
                                              by as much as x_inf when mu is moderate.
    Each limit overshoots in the other's regime, so the minimum is always finite and
    keeps the density away from the underflow that froze the Halley iteration before.
  * Refine in y = ln x on h(y) = ln S(e^y) - ln s, by Newton or Halley.
  * That analytic seed is still a factor ~1.2-3 off, hence ~6 Newton steps. The faster path is a
    table seed (ig_seed_table): ln x on a coarse (ln mu, ln s) grid with local cubic
    interpolation, accurate to ~1e-4..1e-3, which one Halley step turns into ~4e-13
    (Newton needs two).

Accuracy domain. The ~4e-13 figure holds on this project's real data (ln mu <= ~15.7, s >= ~6e-8
on valid quotes). ig_survival subtracts two nearly equal tail probabilities when a = sqrt(x)/mu is
O(1) or larger and b = 1/sqrt(x) is tiny, which is the large-mu, small-s corner. There the root
itself is only defined to ~1e-8 inside the table (ln mu >~ 14, ln s <~ -20) and ~5e-7 beyond it
(mu >~ 3e7, i.e. |log-moneyness| <~ 7e-8). More iterations do not help; that is the evaluation
floor, not convergence. Rows that close to the forward belong on the closed form
sigma = (2/sqrt(T)) Phi^-1((1 + c)/2).

Run as a script to benchmark against scipy.stats.invgauss.ppf on the real SPX data.
"""

import functools
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import erfc, ndtri

SQRT2 = np.sqrt(2.0)


def ig_survival(x, mu):
    """S(x; mu, 1) = Phi(b - a) - exp(2/mu) Phi(-a - b), a = sqrt(x)/mu, b = 1/sqrt(x)."""
    a = np.sqrt(x) / mu
    b = 1.0 / np.sqrt(x)
    return 0.5 * erfc((a - b) / SQRT2) - np.exp(2.0 / mu) * 0.5 * erfc((a + b) / SQRT2)


def ig_pdf(x, mu):
    return np.exp(-((x - mu) ** 2) / (2.0 * mu**2 * x)) / np.sqrt(2.0 * np.pi * x**3)


def ig_seed(s, mu):
    """min(mu -> inf limit, Gaussian-tail limit); see module docstring."""
    x_inf = 1.0 / ndtri(0.5 + 0.5 * s) ** 2
    z = -ndtri(s)
    x_tail = ((z * mu + np.sqrt(z * z * mu * mu + 4.0 * mu)) / 2.0) ** 2
    return np.minimum(x_inf, x_tail)


def _dlogpdf(x, mu):
    """d/dx ln f(x; mu, 1)."""
    return -1.5 / x - (x - mu) * (x + mu) / (2.0 * mu**2 * x**2)


def _refine_log(y, s, mu, n_iter, halley=False):
    """n_iter Newton (or Halley) steps on h(y) = ln S(e^y) - ln s; a non-finite step keeps y."""
    with np.errstate(all="ignore"):
        for _ in range(n_iter):
            x = np.exp(y)
            surv = ig_survival(x, mu)
            pdf = ig_pdf(x, mu)
            h = np.log(surv / s)
            h1 = -x * pdf / surv  # dh/dy
            if halley:
                # h''(x) = -f'/S - (f/S)^2 and d2h/dy2 = x h'(x) + x^2 h''(x), with f' = f (ln f)'.
                h2 = h1 - x * x * pdf * _dlogpdf(x, mu) / surv - (x * pdf / surv) ** 2
                step = 2.0 * h * h1 / (2.0 * h1 * h1 - h * h2)
            else:
                step = h / h1
            y = np.where(np.isfinite(step), y - step, y)
    return y


# ---------------------------------------------------------------------------
# Table seed: ln x on a regular grid in (ln mu, ln s), read back with a 4x4 Lagrange cubic.
# ln x is smooth in these coordinates (it saturates in ln mu once mu >> sqrt(x), and bends
# once between the two limits), so a coarse grid is already accurate to ~1e-3..1e-4, and
# one Halley step turns that into ~1e-13. Nodes are solved once with the analytic-seed
# solver (survival side, so tiny s stays accurate); the table is ~19 KB.
# Domain covers this project's data: mu up to ~7e6 (ln mu ~ 15.7) and s down to the 1e-14 clip.
# gen_reference.py writes the same table to ig_table.h for the C++ port; the constants below
# are the single source for both.
# ---------------------------------------------------------------------------
TABLE_U = (0.0, 17.0)  # ln mu
TABLE_W = (-33.0, -2.0)  # ln s
TABLE_HU = TABLE_HW = 0.5
# Grid cells per axis. The table has cells + 3 nodes per axis: cells + 1 spanning the domain
# plus one padding node each side for the cubic stencil.
TABLE_CELLS_U = round((TABLE_U[1] - TABLE_U[0]) / TABLE_HU)
TABLE_CELLS_W = round((TABLE_W[1] - TABLE_W[0]) / TABLE_HW)


@functools.cache
def _table():
    u = TABLE_U[0] - TABLE_HU + TABLE_HU * np.arange(TABLE_CELLS_U + 3)
    w = TABLE_W[0] - TABLE_HW + TABLE_HW * np.arange(TABLE_CELLS_W + 3)
    grid_u, grid_w = np.meshgrid(u, w, indexing="ij")
    mu, s = np.exp(grid_u), np.exp(grid_w)
    y = _refine_log(np.log(ig_seed(s, mu)), s, mu, 40)
    surv_resid = np.abs(ig_survival(np.exp(y), mu) / s - 1)
    # Where mu >~ 1e5 and s <~ 1e-8, ig_survival subtracts two nearly equal tail probabilities
    # and carries a ~1e-6 floor (see the module docstring). Seeds only need ~1e-4, so tolerate
    # that, but still catch a node that failed to converge at all.
    if not (np.isfinite(y).all() and surv_resid.max() < 1e-4):
        raise RuntimeError(f"table nodes did not converge (max resid {surv_resid.max():.1e})")
    return y


def _lagrange_weights(t):
    return (
        -t * (t - 1) * (t - 2) / 6,
        (t + 1) * (t - 1) * (t - 2) / 2,
        -(t + 1) * t * (t - 2) / 2,
        (t + 1) * t * (t - 1) / 6,
    )


def ig_seed_table(s, mu):
    """(ln x seed, in_domain mask). Rows outside the table domain get NaN and in_domain=False."""
    table = _table()
    f_u = (np.log(mu) - TABLE_U[0]) / TABLE_HU
    f_w = (np.log(s) - TABLE_W[0]) / TABLE_HW
    # Test the domain on the fractional indices themselves, not on ln mu and ln s: the stencil
    # reads cells i..i+3, so i must stay below the cell count, and (w - W0) / h can round up to
    # exactly the cell count for a w just inside the domain. NaN fails every comparison.
    in_domain = (f_u >= 0) & (f_u < TABLE_CELLS_U) & (f_w >= 0) & (f_w < TABLE_CELLS_W)
    # Out-of-domain rows get index 0 so the gather stays in bounds; they are masked out below.
    f_u, f_w = np.where(in_domain, f_u, 0.0), np.where(in_domain, f_w, 0.0)
    i_u, i_w = f_u.astype(int), f_w.astype(int)  # table is offset by +1 for the padding node
    weights_u, weights_w = _lagrange_weights(f_u - i_u), _lagrange_weights(f_w - i_w)
    y = 0.0
    for a in range(4):
        row = 0.0
        for b in range(4):
            row = row + weights_w[b] * table[i_u + a, i_w + b]
        y = y + weights_u[a] * row
    return np.where(in_domain, y, np.nan), in_domain


def ig_quantile_survival(s, mu, n_iter=1, seed="table", halley=True, fallback_iter=6):
    """x such that S_IG(x; mu, 1) = s. s and mu broadcast against each other.

    seed="table": ig_seed_table, then n_iter steps. The default, one Halley step, reaches ~4e-13 on
    the real data; one Newton step reaches ~1e-8, two reach ~1e-13. Rows outside the table domain
    take the analytic seed and fallback_iter Newton steps. They converge, but see the module
    docstring for the accuracy floor at large mu and small s, which applies there and inside the
    table's far corner.
    seed="analytic": ig_seed for every row, then n_iter steps (Newton needs ~6).
    """
    shape = np.broadcast(s, mu).shape
    s, mu = (np.ravel(a).astype(float) for a in np.broadcast_arrays(s, mu))
    if seed == "analytic":
        return np.exp(_refine_log(np.log(ig_seed(s, mu)), s, mu, n_iter, halley=halley)).reshape(shape)
    y0, in_domain = ig_seed_table(s, mu)
    if in_domain.all():
        return np.exp(_refine_log(y0, s, mu, n_iter, halley=halley)).reshape(shape)
    y = np.empty_like(s)
    inside, outside = in_domain, ~in_domain
    y[inside] = _refine_log(y0[inside], s[inside], mu[inside], n_iter, halley=halley)
    y[outside] = _refine_log(np.log(ig_seed(s[outside], mu[outside])), s[outside], mu[outside], fallback_iter)
    return np.exp(y).reshape(shape)


def survival_inputs(target_price, spot, strike, time_to_expiry, rate, dividend):
    """(s, mu) for ig_quantile_survival, with s = 1 - p computed directly and clipped to [1e-14, 1 - 1e-14].

    p = (1 - c) / eta_k, so s = c for k > 0 and (expm1(k) + c) / e^k for k <= 0. For k > 0 that is
    exact, where forming 1 - p rounds a tiny c onto a 1.1e-16 grid. For k <= 0 it still subtracts
    the price from intrinsic value, so its accuracy is bounded by how precisely the quote is
    represented (~1e-16 / s relative); the gain there is dropping the extra rounding of p.
    """
    discount = np.exp(-rate * time_to_expiry)
    forward = spot * np.exp((rate - dividend) * time_to_expiry)
    normalized_price = target_price / (discount * forward)
    log_moneyness = np.log(strike / forward)
    s = np.where(log_moneyness > 0, normalized_price, (np.expm1(log_moneyness) + normalized_price) / np.exp(log_moneyness))
    with np.errstate(divide="ignore"):  # k = 0 gives mu = inf, handled in implied_vol_from_quantile
        mu = 2.0 / np.abs(log_moneyness)
    return np.clip(s, 1e-14, 1 - 1e-14), mu


def implied_vol_from_quantile(target_price, spot, strike, time_to_expiry, rate, dividend, n_iter=1, quantile=None):
    """Schadner IV via ig_quantile_survival. Exactly at the forward (mu = inf) it returns the
    closed form (2/sqrt(T)) Phi^-1((1 + c)/2): the analytic seed reduces to the mu -> inf limit
    and the refinement steps are non-finite there, so the seed is kept. Within ~1e-7 of the
    forward the survival function loses precision (module docstring).

    Same normalization and clip as implied_volatility_schadner_vectorized, except the notebook's
    1e-10 at-the-forward threshold and its no-arbitrage validity mask, which the caller applies.

    quantile is the function that turns (s, mu) into x. It defaults to ig_quantile_survival and can
    be ig_quantile_cpp.ig_quantile_survival to run the same pipeline with the C++ quantile.
    """
    s, mu = survival_inputs(target_price, spot, strike, time_to_expiry, rate, dividend)
    quantile = quantile or ig_quantile_survival
    return (2.0 / np.sqrt(time_to_expiry)) / np.sqrt(quantile(s, mu, n_iter))


def _best_of(fn, repeats=20):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return min(times), out


def _load_real_inputs(data_dir):
    import pandas as pd

    files = [
        "Q1/max_spx_options_filtered_2023-01-04.csv",
        "Q2/max_spx_options_filtered_2023-06-30.csv",
        "Q3/max_spx_options_filtered_2023-08-15.csv",
        "Q4/max_spx_options_filtered_2023-11-14.csv",
    ]
    data = pd.concat(
        [pd.read_csv(Path(data_dir) / f).assign(quarter=f[:2]) for f in files], ignore_index=True
    )
    rate, dividend = 0.05, 0.0
    calls, puts = data[data.option_type == "call"], data[data.option_type == "put"]
    pairs = calls.merge(
        puts,
        on=["quarter", "strike", "expire_date", "underlying_last", "dte"],
        suffixes=("_call", "_put"),
    )
    t = pairs.dte / 365
    call_mid = (pairs.bid_call + pairs.ask_call) / 2
    put_mid = (pairs.bid_put + pairs.ask_put) / 2
    parity_call = put_mid + pairs.underlying_last * np.exp(-dividend * t) - pairs.strike * np.exp(-rate * t)
    price = (call_mid + parity_call) / 2
    return (
        price.to_numpy(float),
        pairs.underlying_last.to_numpy(float),
        pairs.strike.to_numpy(float),
        t.to_numpy(float),
        rate,
        dividend,
    )


if __name__ == "__main__":
    from scipy.stats import invgauss

    DATA_DIR = Path(__file__).resolve().parents[2] / "fim500project" / "SPXIV-Surface" / "data"
    price, spot, strike, t, rate, dividend = _load_real_inputs(DATA_DIR)

    s, mu = survival_inputs(price, spot, strike, t, rate, dividend)
    p = 1 - s
    clipped = s < 1.0001e-14
    print(f"rows {len(s):,}; clipped at p=1-1e-14: {clipped.sum():,}; mu {mu.min():.2g}..{mu.max():.2g}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scipy_seconds, x_scipy = _best_of(lambda: invgauss.ppf(p, mu=mu))
    print(f"scipy invgauss.ppf: {scipy_seconds * 1e3:8.3f} ms  ({scipy_seconds / len(s) * 1e9:7.0f} ns/row)")

    configs = [
        ("analytic", 6, False),
        ("table", 0, False),
        ("table", 1, False),
        ("table", 2, False),
        ("table", 1, True),
        ("table", 2, True),
    ]
    for seed, n_iter, halley in configs:
        # n_iter=0 with the table shows the raw seed accuracy (no refinement at all).
        seconds, x_mine = _best_of(lambda a=seed, n=n_iter, hl=halley: ig_quantile_survival(s, mu, n, seed=a, halley=hl))
        # Truth for the residual check is the survival function itself, which stays accurate
        # at tiny s (scipy's ppf takes p = 1 - s and has already lost the digits there).
        rel_resid = np.abs(ig_survival(x_mine, mu) / s - 1)
        rel_vs_scipy = np.abs(x_mine / x_scipy - 1)[~clipped]
        method = "Halley" if halley else "Newton"
        print(
            f"{seed:8s} {n_iter} {method:6s}: {seconds * 1e3:7.3f} ms ({seconds / len(s) * 1e9:6.0f} ns/row, "
            f"{scipy_seconds / seconds:5.1f}x scipy) | survival resid max {rel_resid.max():.1e} | "
            f"vs scipy (unclipped) max {rel_vs_scipy.max():.1e}"
        )

    seed_seconds, _ = _best_of(lambda: ig_seed_table(s, mu))
    print(f"table seed alone: {seed_seconds * 1e3:.3f} ms ({seed_seconds / len(s) * 1e9:.0f} ns/row)")

    # End to end: IV agreement with the scipy-based pipeline on rows away from the forward.
    iv_mine = implied_vol_from_quantile(price, spot, strike, t, rate, dividend)
    iv_scipy = (2 / np.sqrt(t)) / np.sqrt(x_scipy)
    diff = np.abs(iv_mine - iv_scipy)[~clipped]
    print(f"IV (table, 1 Halley step) vs scipy-based IV, unclipped rows: max |diff| {diff.max():.2e}, median {np.median(diff):.2e}")
