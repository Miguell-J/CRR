"""Einstein tensor computation."""

from __future__ import annotations

import sympy as sp

from crr.core.tensor import Tensor
from crr.symbolic.simplify import maybe_simplify


def einstein_tensor(metric: "Metric", simplify: bool = False) -> Tensor:
    """Compute Einstein tensor G_ij = R_ij - 1/2 g_ij R."""

    n = metric.dimension
    ricci = metric.ricci_tensor().components
    scalar = metric.scalar_curvature()
    components = sp.MutableDenseNDimArray.zeros(n, n)
    for i in range(n):
        for j in range(n):
            components[i, j] = maybe_simplify(
                ricci[i, j] - sp.Rational(1, 2) * metric.components[i, j] * scalar,
                simplify,
            )
    return Tensor(components, name="G", index_signature="_ij")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
