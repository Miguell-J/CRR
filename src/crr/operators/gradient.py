"""Gradient operator."""

from __future__ import annotations

import sympy as sp

from crr.core.tensor import Tensor


def gradient(metric: "Metric", scalar_field: sp.Expr, simplify: bool = False) -> Tensor:
    """Return contravariant gradient components grad(f)^i = g^ij d_j f."""

    n = metric.dimension
    coords = metric.coordinates
    g_inv = metric.inverse()
    components = sp.MutableDenseNDimArray.zeros(n)
    for i in range(n):
        value = sp.S.Zero
        for j in range(n):
            value += g_inv[i, j] * sp.diff(scalar_field, coords[j])
        components[i] = sp.simplify(value) if simplify else value
    return Tensor(components, name=r"\nabla f", index_signature="^i")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
