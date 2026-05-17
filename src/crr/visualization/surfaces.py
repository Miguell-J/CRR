"""Visualization helpers for parametrized surfaces."""

from __future__ import annotations

import numpy as np
import sympy as sp

from crr.visualization.utils import (
    apply_params,
    get_figure_and_axis,
    lambdify_grid,
    make_grid_2d,
    maybe_show,
    save_figure,
)


def plot_surface(
    surface: "ParametrizedMap",
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    resolution: int = 80,
    params: dict[sp.Symbol, object] | None = None,
    ax=None,
    show: bool = True,
    color=None,
    alpha: float = 1.0,
    wireframe: bool = False,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot a 2D parametrized surface in R3."""

    _validate_surface(surface)
    fig, ax = get_figure_and_axis(ax=ax, projection="3d")
    U, V = make_grid_2d(u_range, v_range, resolution=resolution)
    X, Y, Z = _surface_coordinate_grids(surface, U, V, params=params)

    if wireframe:
        ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha)
    else:
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)
    ax.set_title(title or surface.name)
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def plot_curvature(
    surface: "ParametrizedMap",
    curvature: str = "gaussian",
    u_range: tuple[float, float] | None = None,
    v_range: tuple[float, float] | None = None,
    resolution: int = 80,
    params: dict[sp.Symbol, object] | None = None,
    intrinsic: bool = False,
    ax=None,
    show: bool = True,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot Gaussian or mean curvature as a colormap on the surface."""

    _validate_surface(surface)
    if u_range is None or v_range is None:
        raise ValueError("u_range and v_range are required for curvature plotting.")

    fig, ax = get_figure_and_axis(ax=ax, projection="3d")
    U, V = make_grid_2d(u_range, v_range, resolution=resolution)
    X, Y, Z = _surface_coordinate_grids(surface, U, V, params=params)
    expr = _curvature_expression(surface, curvature=curvature, intrinsic=intrinsic)
    C = lambdify_grid(expr, surface.parameters, (U, V), params=params)

    import matplotlib

    norm = matplotlib.colors.Normalize(vmin=np.nanmin(C), vmax=np.nanmax(C))
    colors = matplotlib.colormaps["viridis"](norm(C))
    plot = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0, antialiased=True)
    plot.set_array(C.ravel())
    plot.set_cmap("viridis")
    fig.colorbar(plot, ax=ax, shrink=0.6, pad=0.1)
    ax.set_title(title or f"{surface.name} {curvature} curvature")
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def plot_geodesic(
    surface: "ParametrizedMap",
    solution: "GeodesicSolution",
    ax=None,
    show: bool = True,
    color=None,
    linewidth: float = 2,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot a geodesic embedded by a parametrized surface."""

    points = surface.embed_geodesic(solution)
    fig, ax = get_figure_and_axis(ax=ax, projection="3d" if points.shape[1] == 3 else None)
    if points.shape[1] == 3:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=linewidth)
    else:
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth)
    ax.set_title(title or f"Geodesic on {surface.name}")
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def plot_surface_with_geodesic(
    surface: "ParametrizedMap",
    solution: "GeodesicSolution",
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    resolution: int = 80,
    params: dict[sp.Symbol, object] | None = None,
    show: bool = True,
    surface_alpha: float = 0.6,
    geodesic_color=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot a surface and an embedded geodesic together."""

    fig, ax = plot_surface(
        surface,
        u_range,
        v_range,
        resolution=resolution,
        params=params,
        show=False,
        alpha=surface_alpha,
        title=title or f"{surface.name} with geodesic",
    )
    plot_geodesic(surface, solution, ax=ax, show=False, color=geodesic_color)
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def _surface_coordinate_grids(surface: "ParametrizedMap", U: np.ndarray, V: np.ndarray, params=None):
    return tuple(lambdify_grid(component, surface.parameters, (U, V), params=params) for component in surface.components)


def _curvature_expression(surface: "ParametrizedMap", curvature: str, intrinsic: bool) -> sp.Expr:
    if curvature == "gaussian":
        if intrinsic:
            return sp.Rational(1, 2) * surface.pullback_metric(simplify=True).scalar_curvature(simplify=True)
        return surface.gaussian_curvature_extrinsic(simplify=True)
    if curvature == "mean":
        return surface.mean_curvature(simplify=True)
    raise ValueError('curvature must be "gaussian" or "mean".')


def _validate_surface(surface: "ParametrizedMap") -> None:
    if surface.dimension != 2 or surface.ambient_dimension != 3:
        raise ValueError("Surface plotting requires a 2D parametrized surface in R3.")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.parametrized import ParametrizedMap
    from crr.numeric.geodesic_solver import GeodesicSolution
