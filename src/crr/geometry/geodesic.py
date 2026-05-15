"""Geodesic equation utilities."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp


def geodesic_acceleration(
    metric: "Metric",
    velocity_symbols: Sequence[sp.Symbol] | None = None,
    simplify: bool = False,
) -> list[sp.Expr]:
    """Return a^k = - Gamma^k_ij v^i v^j for symbolic velocities."""

    n = metric.dimension
    velocities = tuple(velocity_symbols) if velocity_symbols is not None else sp.symbols(f"v0:{n}")
    if len(velocities) != n:
        raise ValueError("One velocity symbol is required for each coordinate.")

    gamma = metric.christoffel_symbols().components
    acceleration: list[sp.Expr] = []
    for k in range(n):
        value = sp.S.Zero
        for i in range(n):
            for j in range(n):
                value -= gamma[k, i, j] * velocities[i] * velocities[j]
        acceleration.append(sp.simplify(value) if simplify else value)
    return acceleration


def geodesic_equations(
    metric: "Metric",
    parameter: sp.Symbol | None = None,
    coordinate_functions: Sequence[sp.Function] | None = None,
    parameter_symbol: sp.Symbol | None = None,
    simplify: bool = False,
) -> list[sp.Expr]:
    """Return geodesic equations for coordinate functions x^i(parameter)."""

    n = metric.dimension
    parameter = parameter_symbol or parameter or sp.Symbol("lambda")
    if coordinate_functions is None:
        coordinate_functions = [sp.Function(str(coord)) for coord in metric.coordinates]
    if len(coordinate_functions) != n:
        raise ValueError("One coordinate function is required for each manifold dimension.")

    gamma = metric.christoffel_symbols().components
    substitutions = {
        coord: func(parameter) for coord, func in zip(metric.coordinates, coordinate_functions, strict=True)
    }
    equations: list[sp.Expr] = []
    for k in range(n):
        xk = coordinate_functions[k](parameter)
        value = sp.diff(xk, parameter, 2)
        for i in range(n):
            for j in range(n):
                value += gamma[k, i, j].subs(substitutions) * sp.diff(
                    coordinate_functions[i](parameter), parameter
                ) * sp.diff(coordinate_functions[j](parameter), parameter)
        equations.append(sp.simplify(value) if simplify else value)
    return equations


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
