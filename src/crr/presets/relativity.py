"""Preset relativity metrics."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from crr.core.manifold import Manifold
from crr.core.metric import Metric


def minkowski_metric(
    dim: int = 4,
    signature: str = "-+++",
    coordinates: Sequence[sp.Symbol] | None = None,
    name: str = "Minkowski",
) -> Metric:
    """Return a flat Minkowski metric."""

    if dim < 2:
        raise ValueError("dim must be at least 2.")
    if coordinates is None:
        coords = sp.symbols("t x y z") if dim == 4 else sp.symbols(" ".join(["t", *(f"x{i}" for i in range(1, dim))]))
    else:
        coords = tuple(coordinates)
    if len(coords) != dim:
        raise ValueError("coordinates length must match dim.")

    signs = _signature_signs(signature, dim)
    return Metric(Manifold(name, dim, coords), sp.diag(*signs))


def schwarzschild_metric(
    mass: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    signature: str = "-+++",
    name: str = "Schwarzschild",
) -> Metric:
    """Return the Schwarzschild metric in standard coordinates with G=c=1."""

    t, r, theta, phi = _coordinates4(coordinates)
    M = sp.symbols("M", positive=True) if mass is None else sp.sympify(mass)
    f = 1 - 2 * M / r
    components = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(theta) ** 2)
    if signature == "+---":
        components = -components
    elif signature != "-+++":
        raise ValueError('signature must be "-+++" or "+---".')
    return Metric(Manifold(name, 4, [t, r, theta, phi]), components)


def flrw_metric(
    scale_factor: object | None = None,
    curvature: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    signature: str = "-+++",
    name: str = "FLRW",
) -> Metric:
    """Return the FLRW metric in spherical spatial coordinates."""

    t, r, theta, phi = _coordinates4(coordinates)
    a = sp.Function("a")(t) if scale_factor is None else sp.sympify(scale_factor)
    k = sp.symbols("k") if curvature is None else sp.sympify(curvature)
    spatial = sp.diag(
        a**2 / (1 - k * r**2),
        a**2 * r**2,
        a**2 * r**2 * sp.sin(theta) ** 2,
    )
    components = sp.diag(-1, spatial[0, 0], spatial[1, 1], spatial[2, 2])
    if signature == "+---":
        components = -components
    elif signature != "-+++":
        raise ValueError('signature must be "-+++" or "+---".')
    return Metric(Manifold(name, 4, [t, r, theta, phi]), components)


def robertson_walker_metric(
    scale_factor: object | None = None,
    curvature: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    signature: str = "-+++",
    name: str = "RobertsonWalker",
) -> Metric:
    """Alias for the FLRW metric."""

    return flrw_metric(
        scale_factor=scale_factor,
        curvature=curvature,
        coordinates=coordinates,
        signature=signature,
        name=name,
    )


def reissner_nordstrom_metric(
    mass: object | None = None,
    charge: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    signature: str = "-+++",
    name: str = "ReissnerNordstrom",
) -> Metric:
    """Return the Reissner-Nordstrom metric in standard coordinates with G=c=1."""

    t, r, theta, phi = _coordinates4(coordinates)
    M = sp.symbols("M", positive=True) if mass is None else sp.sympify(mass)
    Q = sp.symbols("Q", real=True) if charge is None else sp.sympify(charge)
    f = 1 - 2 * M / r + Q**2 / r**2
    components = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(theta) ** 2)
    return Metric(Manifold(name, 4, [t, r, theta, phi]), _apply_signature(components, signature))


def kerr_metric(
    mass: object | None = None,
    spin: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    signature: str = "-+++",
    name: str = "Kerr",
) -> Metric:
    """Return the Kerr metric in Boyer-Lindquist coordinates with G=c=1."""

    t, r, theta, phi = _coordinates4(coordinates)
    M = sp.symbols("M", positive=True) if mass is None else sp.sympify(mass)
    a = sp.symbols("a", real=True) if spin is None else sp.sympify(spin)
    sigma = r**2 + a**2 * sp.cos(theta) ** 2
    delta = r**2 - 2 * M * r + a**2
    components = sp.zeros(4, 4)
    components[0, 0] = -(1 - 2 * M * r / sigma)
    components[0, 3] = components[3, 0] = -2 * M * a * r * sp.sin(theta) ** 2 / sigma
    components[1, 1] = sigma / delta
    components[2, 2] = sigma
    components[3, 3] = (r**2 + a**2 + 2 * M * a**2 * r * sp.sin(theta) ** 2 / sigma) * sp.sin(theta) ** 2
    return Metric(Manifold(name, 4, [t, r, theta, phi]), _apply_signature(components, signature))


def eddington_finkelstein_metric(
    mass: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    signature: str = "-+++",
    ingoing: bool = True,
    name: str = "EddingtonFinkelstein",
) -> Metric:
    """Return ingoing or outgoing Eddington-Finkelstein Schwarzschild coordinates."""

    if coordinates is None:
        time, r, theta, phi = sp.symbols(("v" if ingoing else "u") + " r theta phi")
    else:
        time, r, theta, phi = _coordinates4(coordinates)
    M = sp.symbols("M", positive=True) if mass is None else sp.sympify(mass)
    f = 1 - 2 * M / r
    cross = 1 if ingoing else -1
    components = sp.zeros(4, 4)
    components[0, 0] = -f
    components[0, 1] = components[1, 0] = cross
    components[2, 2] = r**2
    components[3, 3] = r**2 * sp.sin(theta) ** 2
    return Metric(Manifold(name, 4, [time, r, theta, phi]), _apply_signature(components, signature))


def kruskal_szekeres_metric(
    mass: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    radial: object | None = None,
    signature: str = "-+++",
    name: str = "KruskalSzekeres",
) -> Metric:
    """Return the Kruskal-Szekeres Schwarzschild metric.

    The areal radius is implicit in Kruskal coordinates. By default this preset
    uses ``r(U, V)`` as an unspecified function; pass ``radial=...`` to use a
    specific expression.
    """

    if coordinates is None:
        U, V, theta, phi = sp.symbols("U V theta phi")
    else:
        U, V, theta, phi = _coordinates4(coordinates)
    M = sp.symbols("M", positive=True) if mass is None else sp.sympify(mass)
    rho = sp.Function("r")(U, V) if radial is None else sp.sympify(radial)
    coefficient = -32 * M**3 * sp.exp(-rho / (2 * M)) / rho
    components = sp.zeros(4, 4)
    components[0, 1] = components[1, 0] = coefficient / 2
    components[2, 2] = rho**2
    components[3, 3] = rho**2 * sp.sin(theta) ** 2
    return Metric(Manifold(name, 4, [U, V, theta, phi]), _apply_signature(components, signature))


def kruskal_metric(
    mass: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    radial: object | None = None,
    signature: str = "-+++",
    name: str = "KruskalSzekeres",
) -> Metric:
    """Backward-compatible alias for :func:`kruskal_szekeres_metric`."""

    return kruskal_szekeres_metric(
        mass=mass,
        coordinates=coordinates,
        radial=radial,
        signature=signature,
        name=name,
    )


def godel_metric(
    parameter: object | None = None,
    scale: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    signature: str = "-+++",
    name: str = "Godel",
) -> Metric:
    """Return a common Godel universe metric form.

    Coordinates are ``(t, x, y, z)`` and
    ``ds^2 = a^2[-dt^2 + dx^2 - 1/2 exp(2x) dy^2 + dz^2 - 2 exp(x) dt dy]``.
    """

    if coordinates is None:
        t, x, y, z = sp.symbols("t x y z")
    else:
        t, x, y, z = _coordinates4(coordinates)
    if parameter is not None and scale is not None:
        raise ValueError("Use either parameter or scale, not both.")
    a_value = parameter if parameter is not None else scale
    a = sp.symbols("a", positive=True) if a_value is None else sp.sympify(a_value)
    components = sp.zeros(4, 4)
    components[0, 0] = -a**2
    components[0, 2] = components[2, 0] = -a**2 * sp.exp(x)
    components[1, 1] = a**2
    components[2, 2] = -a**2 * sp.exp(2 * x) / 2
    components[3, 3] = a**2
    return Metric(Manifold(name, 4, [t, x, y, z]), _apply_signature(components, signature))


def de_sitter_metric(
    radius: object | None = None,
    curvature_radius: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    chart: str = "static",
    signature: str = "-+++",
    name: str = "deSitter",
) -> Metric:
    """Return de Sitter spacetime in static coordinates."""

    if chart != "static":
        raise NotImplementedError('Only chart="static" is currently implemented for de_sitter_metric.')
    if radius is not None and curvature_radius is not None:
        raise ValueError("Use either radius or curvature_radius, not both.")
    t, r, theta, phi = _coordinates4(coordinates)
    radius_value = radius if radius is not None else curvature_radius
    L = sp.symbols("L", positive=True) if radius_value is None else sp.sympify(radius_value)
    f = 1 - r**2 / L**2
    components = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(theta) ** 2)
    return Metric(Manifold(name, 4, [t, r, theta, phi]), _apply_signature(components, signature))


def anti_de_sitter_metric(
    radius: object | None = None,
    curvature_radius: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    chart: str = "static",
    signature: str = "-+++",
    name: str = "AntiDeSitter",
) -> Metric:
    """Return Anti-de Sitter spacetime in static global coordinates."""

    if chart != "static":
        raise NotImplementedError('Only chart="static" is currently implemented for anti_de_sitter_metric.')
    if radius is not None and curvature_radius is not None:
        raise ValueError("Use either radius or curvature_radius, not both.")
    t, r, theta, phi = _coordinates4(coordinates)
    radius_value = radius if radius is not None else curvature_radius
    L = sp.symbols("L", positive=True) if radius_value is None else sp.sympify(radius_value)
    f = 1 + r**2 / L**2
    components = sp.diag(-f, 1 / f, r**2, r**2 * sp.sin(theta) ** 2)
    return Metric(Manifold(name, 4, [t, r, theta, phi]), _apply_signature(components, signature))


def alcubierre_metric(
    velocity: object | None = None,
    shape_function: object | None = None,
    coordinates: Sequence[sp.Symbol] | None = None,
    signature: str = "-+++",
    name: str = "Alcubierre",
) -> Metric:
    """Return a simple Alcubierre warp-drive metric with shift in the x direction.

    Coordinates are ``(t, x, y, z)`` and
    ``ds^2 = -dt^2 + (dx - v_s f dt)^2 + dy^2 + dz^2``.
    """

    if coordinates is None:
        t, x, y, z = sp.symbols("t x y z")
    else:
        t, x, y, z = _coordinates4(coordinates)
    v_s = sp.symbols("v_s", real=True) if velocity is None else sp.sympify(velocity)
    f = sp.symbols("f_s", real=True) if shape_function is None else sp.sympify(shape_function)
    components = sp.eye(4)
    components[0, 0] = -1 + v_s**2 * f**2
    components[0, 1] = components[1, 0] = -v_s * f
    return Metric(Manifold(name, 4, [t, x, y, z]), _apply_signature(components, signature))


def _signature_signs(signature: str, dim: int) -> list[int]:
    if signature == "-+++":
        return [-1, *([1] * (dim - 1))]
    if signature == "+---":
        return [1, *([-1] * (dim - 1))]
    raise ValueError('signature must be "-+++" or "+---".')


def _apply_signature(components: sp.Matrix, signature: str) -> sp.Matrix:
    if signature == "-+++":
        return components
    if signature == "+---":
        return -components
    raise ValueError('signature must be "-+++" or "+---".')


def _coordinates4(coordinates: Sequence[sp.Symbol] | None) -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]:
    if coordinates is None:
        return sp.symbols("t r theta phi")
    coordinates_tuple = tuple(coordinates)
    if len(coordinates_tuple) != 4:
        raise ValueError("Expected 4 coordinate symbols.")
    return coordinates_tuple  # type: ignore[return-value]
