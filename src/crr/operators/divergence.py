"""Divergence operator."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp


def divergence(metric: "Metric", vector_field: Sequence[sp.Expr], simplify: bool = False) -> sp.Expr:
    """Return divergence of a contravariant vector field."""

    n = metric.dimension
    if len(vector_field) != n:
        raise ValueError("Vector field length must match manifold dimension.")
    coords = metric.coordinates
    volume_density = sp.sqrt(abs(metric.determinant()))
    value = sp.S.Zero
    for i in range(n):
        value += sp.diff(volume_density * vector_field[i], coords[i])
    value = value / volume_density
    return sp.simplify(value) if simplify else value


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
