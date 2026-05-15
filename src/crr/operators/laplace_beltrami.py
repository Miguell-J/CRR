"""Laplace-Beltrami operator."""

from __future__ import annotations

import sympy as sp


def laplace_beltrami(metric: "Metric", scalar_field: sp.Expr, simplify: bool = False) -> sp.Expr:
    """Return the Laplace-Beltrami operator applied to a scalar field."""

    n = metric.dimension
    coords = metric.coordinates
    g_inv = metric.inverse()
    volume_density = sp.sqrt(abs(metric.determinant()))
    value = sp.S.Zero
    for i in range(n):
        inner = sp.S.Zero
        for j in range(n):
            inner += g_inv[i, j] * sp.diff(scalar_field, coords[j])
        value += sp.diff(volume_density * inner, coords[i])
    value = value / volume_density
    return sp.simplify(value) if simplify else value


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
