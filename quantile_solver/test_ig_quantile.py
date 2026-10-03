"""Behavior tests for ig_quantile.py. Run from this folder: python -m unittest -v"""

import unittest
from pathlib import Path

import mpmath as mp
import numpy as np

import gen_reference
import ig_quantile as q
import ig_quantile_cpp
import mp_truth



class TableDomainEdges(unittest.TestCase):
    def test_s_one_ulp_below_top_of_table_does_not_read_out_of_bounds(self):
        # (w - W0) / h rounds up to exactly the last cell boundary for this s.
        s = np.exp(np.nextafter(q.TABLE_W[1], -np.inf))
        for mu in (10.0, np.exp(np.nextafter(q.TABLE_U[1], 0.0))):
            x = q.ig_quantile_survival(np.array([s]), np.array([mu]))
            self.assertTrue(np.isfinite(x).all())
            self.assertLess(abs(q.ig_survival(x, mu) / s - 1)[0], 1e-6)

    def test_scalar_mu_with_an_off_table_row(self):
        x = q.ig_quantile_survival(np.array([0.01, 0.5]), 10.0)  # s = 0.5 is above the table
        self.assertEqual(x.shape, (2,))
        self.assertTrue(np.isfinite(x).all())

    def test_scalar_inputs_return_a_scalar_shape(self):
        self.assertEqual(q.ig_quantile_survival(0.01, 10.0).shape, ())


class SurvivalInputs(unittest.TestCase):
    def test_otm_s_is_the_normalized_price_exactly(self):
        # Deep OTM call, c ~ 2.5e-13. s = c exactly; forming 1 - p rounds it onto a 1.1e-16 grid,
        # a ~4e-4 relative error here. (The ITM branch subtracts price from intrinsic value, so its
        # accuracy is bounded by how precisely the quote is represented; no test can ask for more.)
        rate, dividend, spot, strike, t = 0.05, 0.0, 4000.0, 6000.0, 0.25
        price = 1e-9
        s, mu = q.survival_inputs(
            np.array([price]), np.array([spot]), np.array([strike]), np.array([t]), rate, dividend
        )
        forward = spot * np.exp((rate - dividend) * t)
        c = price / (np.exp(-rate * t) * forward)
        self.assertLess(c, 1e-12)
        self.assertEqual(s[0], c)
        self.assertAlmostEqual(mu[0], 2 / abs(np.log(strike / forward)), places=12)


class AtTheForward(unittest.TestCase):
    def test_exactly_at_the_forward_returns_the_closed_form(self):
        from scipy.special import ndtri
        from scipy.stats import norm

        spot, t, rate, sigma = 4000.0, 0.25, 0.05, 0.18
        strike = forward = spot * np.exp(rate * t)
        d1 = 0.5 * sigma * np.sqrt(t)
        price = np.exp(-rate * t) * (forward * norm.cdf(d1) - strike * norm.cdf(-d1))
        got = q.implied_vol_from_quantile(
            np.array([price]), np.array([spot]), np.array([strike]), np.array([t]), rate, 0.0
        )
        c = price / (np.exp(-rate * t) * forward)
        self.assertAlmostEqual(got[0], 2 / np.sqrt(t) * ndtri((1 + c) / 2), places=12)


class Accuracy(unittest.TestCase):
    def test_one_halley_step_matches_mpmath_on_real_data_range(self):
        # Spread across the region real SPX rows occupy (ln mu 1..16, s from 1e-8 to 0.08).
        cases = [(35.0, 1e-4), (1.5, 0.05), (300.0, 1e-3), (3e3, 1e-6), (6.8e6, 1e-3), (50.0, 1e-8)]
        for mu, s in cases:
            x = float(q.ig_quantile_survival(np.array([s]), np.array([mu]))[0])
            truth = mp_truth.quantile(s, mu, x)
            self.assertLess(abs(float(x / truth - 1)), 1e-11, f"mu={mu} s={s}")


class CppLibrary(unittest.TestCase):
    """The ctypes-wrapped C++ quantile must reproduce the Python module."""

    @classmethod
    def setUpClass(cls):
        try:
            ig_quantile_cpp.build()
        except (FileNotFoundError, RuntimeError) as exc:
            raise unittest.SkipTest(f"no C++ compiler or build failed: {exc}")

    def test_matches_python_on_real_rows(self):
        price, spot, strike, t, rate, dividend = q._load_real_inputs(gen_reference.DATA_DIR)
        s, mu = q.survival_inputs(price, spot, strike, t, rate, dividend)
        keep = np.isfinite(mu) & (s > 1.0001e-14)
        x_py = q.ig_quantile_survival(s[keep], mu[keep])
        x_cpp = ig_quantile_cpp.ig_quantile_survival(s[keep], mu[keep])
        self.assertLess(np.max(np.abs(x_cpp / x_py - 1)), 1e-10)

    def test_table_edge_input_does_not_misbehave(self):
        s = np.exp(np.nextafter(q.TABLE_W[1], -np.inf))
        x = ig_quantile_cpp.ig_quantile_survival(np.array([s, s]), np.array([10.0, np.exp(np.nextafter(q.TABLE_U[1], 0.0))]))
        self.assertTrue(np.isfinite(x).all())
        self.assertLess(np.max(np.abs(q.ig_survival(x, np.array([10.0, np.exp(np.nextafter(q.TABLE_U[1], 0.0))])) / s - 1)), 1e-6)

    def test_scalar_and_shape_are_preserved(self):
        self.assertEqual(ig_quantile_cpp.ig_quantile_survival(0.01, 10.0).shape, ())
        self.assertEqual(ig_quantile_cpp.ig_quantile_survival(np.full((2, 3), 0.01), 10.0).shape, (2, 3))

    def test_full_iv_with_cpp_quantile_matches_python_iv(self):
        price, spot, strike, t, rate, dividend = q._load_real_inputs(gen_reference.DATA_DIR)
        iv_py = q.implied_vol_from_quantile(price, spot, strike, t, rate, dividend)
        iv_cpp = q.implied_vol_from_quantile(price, spot, strike, t, rate, dividend, quantile=ig_quantile_cpp.ig_quantile_survival)
        both = np.isfinite(iv_py) & np.isfinite(iv_cpp) & (np.abs(np.log(strike / (spot * np.exp(rate * t)))) > 1e-7)
        self.assertLess(np.max(np.abs(iv_cpp - iv_py)[both]), 1e-9)

    def test_exactly_at_the_forward_matches_python(self):
        s_, mu_ = np.array([0.04]), np.array([np.inf])
        self.assertAlmostEqual(ig_quantile_cpp.ig_quantile_survival(s_, mu_)[0], q.ig_quantile_survival(s_, mu_)[0], places=9)


class GeneratedHeader(unittest.TestCase):
    def test_ig_table_h_matches_the_python_table(self):
        on_disk = (Path(gen_reference.__file__).parent / "ig_table.h").read_text()
        self.assertEqual(on_disk, gen_reference.table_header_text(), "stale ig_table.h: rerun gen_reference.py")


if __name__ == "__main__":
    unittest.main()
