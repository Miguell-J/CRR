"""Preset Riemannian metrics."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from crr.core.manifold import Manifold
from crr.core.metric import Metric


def euclidean_metric(
    dim: int = 2,
    coordinates: Sequence[sp.Symbol] | None = None,
    name: str | None = None,
) -> Metric:
    """Return the Euclidean metric in the requested dimension."""

    if dim < 1:
        raise ValueError("dim must be positive.")
    coords = _coordinates(coordinates, tuple(f"x{i}" for i in range(dim)))
    if len(coords) != dim:
        raise ValueError("coordinates length must match dim.")
    return Metric(Manifold(name or f"R{dim}", dim, coords), sp.eye(dim))


def polar_metric(
    coordinates: Sequence[sp.Symbol] | None = None,
    name: str = "PolarPlane",
) -> Metric:
    """Return the Euclidean plane metric in polar coordinates."""

    r, theta = _coordinates(coordinates, ("r", "theta"))
    return Metric(Manifold(name, 2, [r, theta]), [[1, 0], [0, r**2]])


def sphere_metric(
    radius: object = 1,
    coordinates: Sequence[sp.Symbol] | None = None,
    name: str = "Sphere",
) -> Metric:
    """Return the round 2-sphere metric."""

    theta, phi = _coordinates(coordinates, ("theta", "phi"))
    r = sp.sympify(radius)
    return Metric(Manifold(name, 2, [theta, phi]), [[r**2, 0], [0, r**2 * sp.sin(theta) ** 2]])


def hyperbolic_half_plane_metric(
    coordinates: Sequence[sp.Symbol] | None = None,
    name: str = "HyperbolicHalfPlane",
) -> Metric:
    """Return the Poincare half-plane metric."""

    x, y = _coordinates(coordinates, ("x", "y"))
    return Metric(Manifold(name, 2, [x, y]), [[1 / y**2, 0], [0, 1 / y**2]])


def poincare_disk_metric(
    coordinates: Sequence[sp.Symbol] | None = None,
    name: str = "PoincareDisk",
) -> Metric:
    """Return the Poincare disk metric in polar coordinates."""

    r, theta = _coordinates(coordinates, ("r", "theta"))
    conformal = 4 / (1 - r**2) ** 2
    return Metric(Manifold(name, 2, [r, theta]), [[conformal, 0], [0, conformal * r**2]])


def cylinder_metric(
    radius: object = 1,
    coordinates: Sequence[sp.Symbol] | None = None,
    name: str = "Cylinder",
) -> Metric:
    """Return the induced metric on a circular cylinder."""

    theta, z = _coordinates(coordinates, ("theta", "z"))
    r = sp.sympify(radius)
    return Metric(Manifold(name, 2, [theta, z]), [[r**2, 0], [0, 1]])


def torus_metric(
    major_radius: object | None = None,
    minor_radius: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    name: str = "Torus",
) -> Metric:
    """Return the standard torus induced metric."""

    u, v = _coordinates(coordinates, ("u", "v"))
    R = sp.symbols("R", positive=True) if major_radius is None else sp.sympify(major_radius)
    r = sp.symbols("r", positive=True) if minor_radius is None else sp.sympify(minor_radius)
    return Metric(Manifold(name, 2, [u, v]), [[(R + r * sp.cos(v)) ** 2, 0], [0, r**2]])


def _coordinates(coordinates: Sequence[sp.Symbol] | None, names: tuple[str, ...]) -> tuple[sp.Symbol, ...]:
    if coordinates is None:
        symbols = sp.symbols(" ".join(names))
        return symbols if isinstance(symbols, tuple) else (symbols,)
    coordinates_tuple = tuple(coordinates)
    if len(coordinates_tuple) != len(names):
        raise ValueError(f"Expected {len(names)} coordinate symbols.")
    return coordinates_tuple
