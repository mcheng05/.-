"""ctypes wrapper that calls the C++ quantile (ig_quantile.hpp) on NumPy arrays.

build() compiles ig_quantile_capi.cpp into a shared library next to this file if the library is
missing or older than its sources. ig_quantile_survival has the same signature and result shape as
ig_quantile.ig_quantile_survival for the table path, so it can be passed to
ig_quantile.implied_vol_from_quantile(quantile=...) to run the full price-to-IV pipeline with the
C++ quantile. Needs clang++ or g++ on the path.
"""

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / ("libig_quantile.dylib" if sys.platform == "darwin" else "libig_quantile.so")
SOURCES = [HERE / name for name in ("ig_quantile_capi.cpp", "ig_quantile.hpp", "ig_table.h")]

_lib = None


def build(force=False):
    """Compile the shared library if it is missing or stale. Raises FileNotFoundError without a compiler."""
    if not force and LIB_PATH.exists() and LIB_PATH.stat().st_mtime >= max(p.stat().st_mtime for p in SOURCES):
        return LIB_PATH
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise FileNotFoundError("clang++ or g++ not found on PATH")
    result = subprocess.run(
        [compiler, "-O3", "-march=native", "-std=c++17", "-shared", "-fPIC", str(HERE / "ig_quantile_capi.cpp"), "-o", str(LIB_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"building {LIB_PATH.name} failed:\n{result.stderr}")
    global _lib
    _lib = None  # reload the rebuilt library
    return LIB_PATH


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(build()))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        lib.ig_quantile_survival_array.argtypes = [array, array, array, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
        lib.ig_quantile_survival_array.restype = None
        _lib = lib
    return _lib


def ig_quantile_survival(s, mu, n_iter=1, halley=True):
    """x with S_IG(x; mu, 1) = s, computed by the C++ port. s and mu broadcast against each other."""
    shape = np.broadcast(s, mu).shape
    s_flat, mu_flat = (np.ascontiguousarray(np.ravel(a), dtype=np.float64) for a in np.broadcast_arrays(s, mu))
    x = np.empty_like(s_flat)
    _load().ig_quantile_survival_array(s_flat, mu_flat, x, s_flat.size, n_iter, int(halley))
    return x.reshape(shape)
