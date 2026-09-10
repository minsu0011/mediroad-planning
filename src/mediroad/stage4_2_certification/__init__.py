"""Stage 4.2C solver-certification hardening overlay for MEDIROAD."""

import os

# Apply before NumPy/SciPy imports. HiGHS owns the native CPU threads.
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("PYTHONHASHSEED", "42")

from .version import __version__

__all__ = ["__version__"]
