"""Covariant derivative computations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sympy as sp

from crr.core.tensor import Tensor


def covariant_derivative_scalar(metric: "Metric", scalar_field: sp.Expr, simplify: bool = False) -> Tensor:
    """Compute nabla_i f = partial_i f for a scalar field."""

    components = sp.MutableDenseNDimArray.zeros(metric.dimension)
    for i, coord in enumerate(metric.coordinates):
        value = sp.diff(scalar_field, coord)
        components[i] = sp.simplify(value) if simplify else value
    return Tensor(components, name=r"\nabla f", index_signature="_i")


def covariant_derivative_vector(
    metric: "Metric",
    vector_field: Sequence[Any] | Tensor,
    simplify: bool = False,
) -> Tensor:
    """Compute nabla_j V^i = partial_j V^i + Gamma^i_jk V^k."""

    vector = _as_rank1_components(vector_field, metric.dimension, "vector field")
    gamma = metric.christoffel_symbols().components
    components = sp.MutableDenseNDimArray.zeros(metric.dimension, metric.dimension)

    for i in range(metric.dimension):
        for j, coord in enumerate(metric.coordinates):
            value = sp.diff(vector[i], coord)
            for k in range(metric.dimension):
                value += gamma[i, j, k] * vector[k]
            components[i, j] = sp.simplify(value) if simplify else value
    return Tensor(components, name=r"\nabla V", index_signature="^i_j")


def covariant_derivative_covector(
    metric: "Metric",
    covector_field: Sequence[Any] | Tensor,
    simplify: bool = False,
) -> Tensor:
    """Compute nabla_j omega_i = partial_j omega_i - Gamma^k_ji omega_k."""

    covector = _as_rank1_components(covector_field, metric.dimension, "covector field")
    gamma = metric.christoffel_symbols().components
    components = sp.MutableDenseNDimArray.zeros(metric.dimension, metric.dimension)

    for i in range(metric.dimension):
        for j, coord in enumerate(metric.coordinates):
            value = sp.diff(covector[i], coord)
            for k in range(metric.dimension):
                value -= gamma[k, j, i] * covector[k]
            components[i, j] = sp.simplify(value) if simplify else value
    return Tensor(components, name=r"\nabla \omega", index_signature="_i_j")


def covariant_derivative_covariant_2tensor(
    metric: "Metric",
    tensor_field: Any,
    simplify: bool = False,
) -> Tensor:
    """Compute nabla_k T_ij for a covariant rank-2 tensor."""

    tensor = _as_rank2_components(tensor_field, metric.dimension)
    gamma = metric.christoffel_symbols().components
    components = sp.MutableDenseNDimArray.zeros(metric.dimension, metric.dimension, metric.dimension)

    for i in range(metric.dimension):
        for j in range(metric.dimension):
            for k, coord in enumerate(metric.coordinates):
                value = sp.diff(tensor[i, j], coord)
                for m in range(metric.dimension):
                    value -= gamma[m, k, i] * tensor[m, j]
                    value -= gamma[m, k, j] * tensor[i, m]
                components[i, j, k] = sp.simplify(value) if simplify else value
    return Tensor(components, name=r"\nabla T", index_signature="_i_j_k")


def _as_rank1_components(field: Sequence[Any] | Tensor, dimension: int, label: str) -> list[sp.Expr]:
    if isinstance(field, Tensor):
        if field.rank != 1 or field.shape[0] != dimension:
            raise ValueError(f"{label} must be a rank-1 tensor with one component per coordinate.")
        return [field[i] for i in range(dimension)]
    if len(field) != dimension:
        raise ValueError(f"{label} must have one component per coordinate.")
    return [sp.sympify(component) for component in field]


def _as_rank2_components(field: Any, dimension: int) -> sp.MutableDenseNDimArray:
    if isinstance(field, Tensor):
        if field.rank != 2 or field.shape != (dimension, dimension):
            raise ValueError("tensor field must be rank 2 and match the metric dimension.")
        return field.components

    matrix = sp.Matrix(field)
    if matrix.shape != (dimension, dimension):
        raise ValueError("tensor field must be a square matrix matching the metric dimension.")
    return sp.MutableDenseNDimArray(matrix.tolist())


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
