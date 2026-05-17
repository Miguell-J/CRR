"""Numerical integration utilities."""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.integrate import nquad


def integrate_scalar_numeric(
    metric: "Metric",
    scalar: sp.Expr,
    ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
    params: dict[sp.Symbol, object] | None = None,
    nquad_options: dict[str, object] | None = None,
    absolute_density: bool = True,
) -> float:
    """Numerically integrate scalar * volume density over a coordinate box."""

    if len(ranges) != metric.dimension:
        raise ValueError("One integration range is required for each coordinate.")
    if not 1 <= len(ranges) <= 3:
        raise ValueError("Numeric integration currently supports dimensions 1, 2, and 3.")

    substitutions = params or {}
    coordinates = tuple(item[0] for item in ranges)
    if coordinates != metric.coordinates:
        raise ValueError("Integration ranges must be ordered like the metric coordinates.")

    density = metric.volume_density(absolute=absolute_density)
    integrand = (sp.sympify(scalar) * density).subs(substitutions)
    bounds = []
    for _coord, lower, upper in ranges:
        bounds.append([float(sp.N(sp.sympify(lower).subs(substitutions))), float(sp.N(sp.sympify(upper).subs(substitutions)))])

    function = sp.lambdify(coordinates, integrand, modules="numpy")

    def wrapped(*args: float) -> float:
        return float(np.asarray(function(*args), dtype=float))

    value, _error = nquad(wrapped, bounds, opts=nquad_options)
    return float(value)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
