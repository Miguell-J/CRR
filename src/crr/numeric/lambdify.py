"""Lambdification helpers for numerical geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from crr.core.tensor import Tensor


def lambdify_christoffel(christoffel: Tensor | Any, coordinates: tuple[sp.Symbol, ...]):
    """Return a callable evaluating Christoffel symbols with shape ``(n, n, n)``."""

    components = christoffel.components if isinstance(christoffel, Tensor) else christoffel
    array = sp.MutableDenseNDimArray(components)
    function = sp.lambdify(coordinates, array.tolist(), modules="numpy")

    def evaluate(x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        point = np.asarray(x, dtype=float)
        if point.shape != (len(coordinates),):
            raise ValueError("x must have shape (dimension,).")
        return np.asarray(function(*point), dtype=float)

    return evaluate


def lambdify_metric(metric: "Metric"):
    """Return a callable evaluating metric components with shape ``(n, n)``."""

    function = sp.lambdify(metric.coordinates, metric.components.tolist(), modules="numpy")

    def evaluate(x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        point = np.asarray(x, dtype=float)
        if point.shape != (metric.dimension,):
            raise ValueError("x must have shape (dimension,).")
        return np.asarray(function(*point), dtype=float)

    return evaluate


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
