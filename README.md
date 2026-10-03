# Schadner Explicit Black-Scholes IV Solution
based off https://arxiv.org/pdf/2604.24480v4
Summary of results in schadner_results_summary.ipynb

TLDR: Implied volatility from an option price now takes about 65 ns per option, down from about 700 µs with bisection, accurate to about 1e-13 relative error. Timings done on one laptop with ~18,000 real SPX option quotes, with Python handing price mapping and C++ handling arithmetic. 
