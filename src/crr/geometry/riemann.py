"""Riemann curvature tensor computation."""

from __future__ import annotations

import sympy as sp

from crr.core.tensor import Tensor
from crr.symbolic.simplify import maybe_simplify


def riemann_tensor(metric: "Metric", simplify: bool = False) -> Tensor:
    """Compute Riemann tensor R^i_{jkl} using the CRR convention."""

    n = metric.dimension
    coords = metric.coordinates
    gamma = metric.christoffel_symbols().components
    riemann = sp.MutableDenseNDimArray.zeros(n, n, n, n)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    value = sp.diff(gamma[i, l, j], coords[k]) - sp.diff(gamma[i, k, j], coords[l])
                    for m in range(n):
                        value += gamma[i, k, m] * gamma[m, l, j]
                        value -= gamma[i, l, m] * gamma[m, k, j]
                    riemann[i, j, k, l] = maybe_simplify(value, simplify)

    return Tensor(riemann, name="R", index_signature="^i_jkl")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
