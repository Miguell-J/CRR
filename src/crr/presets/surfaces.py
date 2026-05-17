"""Preset parametrized surfaces."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from crr.core.parametrized import ParametrizedMap


def plane_surface(
    name: str = "Plane",
    coordinates: Sequence[sp.Symbol] | None = None,
) -> ParametrizedMap:
    """Return the plane F(u, v) = (u, v, 0)."""

    u, v = _coordinates(coordinates, ("u", "v"))
    return ParametrizedMap(name=name, parameters=[u, v], components=[u, v, 0])


def sphere_surface(
    radius: object = 1,
    name: str = "Sphere",
    coordinates: Sequence[sp.Symbol] | None = None,
) -> ParametrizedMap:
    """Return the standard sphere parametrization in Euclidean R3."""

    theta, phi = _coordinates(coordinates, ("theta", "phi"))
    r = sp.sympify(radius)
    return ParametrizedMap(
        name=name,
        parameters=[theta, phi],
        components=[
            r * sp.sin(theta) * sp.cos(phi),
            r * sp.sin(theta) * sp.sin(phi),
            r * sp.cos(theta),
        ],
    )


def cylinder_surface(
    radius: object = 1,
    name: str = "Cylinder",
    coordinates: Sequence[sp.Symbol] | None = None,
) -> ParametrizedMap:
    """Return the standard cylinder parametrization."""

    theta, z = _coordinates(coordinates, ("theta", "z"))
    r = sp.sympify(radius)
    return ParametrizedMap(
        name=name,
        parameters=[theta, z],
        components=[r * sp.cos(theta), r * sp.sin(theta), z],
    )


def torus_surface(
    major_radius: object | None = None,
    minor_radius: object | None = None,
    name: str = "Torus",
    coordinates: Sequence[sp.Symbol] | None = None,
) -> ParametrizedMap:
    """Return the standard torus parametrization."""

    u, v = _coordinates(coordinates, ("u", "v"))
    R = sp.symbols("R", positive=True) if major_radius is None else sp.sympify(major_radius)
    r = sp.symbols("r", positive=True) if minor_radius is None else sp.sympify(minor_radius)
    return ParametrizedMap(
        name=name,
        parameters=[u, v],
        components=[
            (R + r * sp.cos(v)) * sp.cos(u),
            (R + r * sp.cos(v)) * sp.sin(u),
            r * sp.sin(v),
        ],
    )


def catenoid_surface(
    scale: object | None = None,
    name: str = "Catenoid",
    coordinates: Sequence[sp.Symbol] | None = None,
) -> ParametrizedMap:
    """Return a catenoid parametrization."""

    u, v = _coordinates(coordinates, ("u", "v"))
    a = sp.symbols("a", positive=True) if scale is None else sp.sympify(scale)
    return ParametrizedMap(
        name=name,
        parameters=[u, v],
        components=[
            a * sp.cosh(v / a) * sp.cos(u),
            a * sp.cosh(v / a) * sp.sin(u),
            v,
        ],
    )


def helicoid_surface(
    pitch: object | None = None,
    name: str = "Helicoid",
    coordinates: Sequence[sp.Symbol] | None = None,
) -> ParametrizedMap:
    """Return a helicoid parametrization."""

    u, v = _coordinates(coordinates, ("u", "v"))
    a = sp.symbols("a", positive=True) if pitch is None else sp.sympify(pitch)
    return ParametrizedMap(
        name=name,
        parameters=[u, v],
        components=[v * sp.cos(u), v * sp.sin(u), a * u],
    )


def mobius_strip_surface(
    name: str = "MobiusStrip",
    coordinates: Sequence[sp.Symbol] | None = None,
) -> ParametrizedMap:
    """Return a local parametrization of a Mobius strip."""

    u, v = _coordinates(coordinates, ("u", "v"))
    factor = 1 + sp.Rational(1, 2) * v * sp.cos(u / 2)
    return ParametrizedMap(
        name=name,
        parameters=[u, v],
        components=[
            factor * sp.cos(u),
            factor * sp.sin(u),
            sp.Rational(1, 2) * v * sp.sin(u / 2),
        ],
    )


def _coordinates(coordinates: Sequence[sp.Symbol] | None, names: tuple[str, ...]) -> tuple[sp.Symbol, ...]:
    if coordinates is None:
        symbols = sp.symbols(" ".join(names))
        return symbols if isinstance(symbols, tuple) else (symbols,)
    coordinates_tuple = tuple(coordinates)
    if len(coordinates_tuple) != len(names):
        raise ValueError(f"Expected {len(names)} coordinate symbols.")
    return coordinates_tuple
