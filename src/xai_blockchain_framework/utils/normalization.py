"""Numerical normalization helpers.

Two flavors are provided:

- :func:`min_max_normalize` rescales values to ``[0, 1]`` linearly.
- :func:`log_normalize` applies ``log10(1 + x)`` before min-max rescaling,
  which is essential when a metric spans several orders of magnitude
  (e.g. Lipschitz values can range from 0.01 to nearly 500,000).

Both functions support the ``lower_better`` convention: when True, the
returned values are inverted so that 1.0 always corresponds to the best
score.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def min_max_normalize(values: ArrayLike, lower_better: bool = False) -> NDArray[np.float64]:
    """Linearly rescale ``values`` to the unit interval.

    Parameters
    ----------
    values : array-like
        Input values (1-D).
    lower_better : bool, default False
        When True, invert the result so that the smallest input maps to 1.0.

    Returns
    -------
    numpy.ndarray
        Normalized values of shape ``(n,)``. When all inputs are equal, the
        function returns ``0.5`` for every entry to avoid division by zero.
    """
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    span = hi - lo
    if span < 1e-10:
        return np.full_like(arr, 0.5)
    out = (arr - lo) / span
    return 1.0 - out if lower_better else out


def log_normalize(values: ArrayLike, lower_better: bool = False) -> NDArray[np.float64]:
    """Apply ``log10(1 + x)`` then min-max normalize to ``[0, 1]``.

    This is the recommended normalization for metrics whose raw scale is
    unbounded above and spans multiple orders of magnitude (Lipschitz
    stability, Coefficient of Variation). Without the log transform, a few
    extreme values compress the rest of the range into a tiny band near 0.

    Parameters
    ----------
    values : array-like
        Non-negative input values (1-D).
    lower_better : bool, default False
        When True, invert so that the smallest input maps to 1.0.

    Returns
    -------
    numpy.ndarray
        Normalized values of shape ``(n,)``.

    Raises
    ------
    ValueError
        If any input value is strictly negative.
    """
    arr = np.asarray(values, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("log_normalize requires non-negative inputs.")
    logged = np.log10(1.0 + arr)
    return min_max_normalize(logged, lower_better=lower_better)
