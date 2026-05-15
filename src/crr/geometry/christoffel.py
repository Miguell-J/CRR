"""Christoffel symbol computation."""

from __future__ import annotations

import sympy as sp

from crr.core.tensor import Tensor
from crr.symbolic.simplify import maybe_simplify


def christoffel_symbols(metric: "Metric", simplify: bool = False) -> Tensor:
    """Compute Christoffel symbols Gamma^k_{ij} for a metric."""

    n = metric.dimension
    coords = metric.coordinates
    g = metric.components
    g_inv = metric.inverse()
    gamma = sp.MutableDenseNDimArray.zeros(n, n, n)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                value = sp.S.Zero
                for l in range(n):
                    value += g_inv[k, l] * (
                        sp.diff(g[j, l], coords[i])
                        + sp.diff(g[i, l], coords[j])
                        - sp.diff(g[i, j], coords[l])
                    )
                gamma[k, i, j] = maybe_simplify(sp.Rational(1, 2) * value, simplify)

    return Tensor(gamma, name=r"\Gamma", index_signature="^k_ij")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
