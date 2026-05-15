"""Ricci tensor computation."""

from __future__ import annotations

import sympy as sp

from crr.core.tensor import Tensor
from crr.symbolic.simplify import maybe_simplify


def ricci_tensor(metric: "Metric", simplify: bool = False) -> Tensor:
    """Compute Ricci tensor R_jl = R^i_{jil}."""

    n = metric.dimension
    riemann = metric.riemann_tensor().components
    ricci = sp.MutableDenseNDimArray.zeros(n, n)

    for j in range(n):
        for l in range(n):
            value = sp.S.Zero
            for i in range(n):
                value += riemann[i, j, i, l]
            ricci[j, l] = maybe_simplify(value, simplify)

    return Tensor(ricci, name="R", index_signature="_ij")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
