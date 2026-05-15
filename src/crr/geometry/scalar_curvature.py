"""Scalar curvature computation."""

from __future__ import annotations

import sympy as sp

from crr.symbolic.simplify import maybe_simplify


def scalar_curvature(metric: "Metric", simplify: bool = False) -> sp.Expr:
    """Compute scalar curvature R = g^jl R_jl."""

    n = metric.dimension
    g_inv = metric.inverse()
    ricci = metric.ricci_tensor().components
    value = sp.S.Zero
    for j in range(n):
        for l in range(n):
            value += g_inv[j, l] * ricci[j, l]
    return maybe_simplify(value, simplify)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
