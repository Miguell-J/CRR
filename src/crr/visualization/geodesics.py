"""Lightweight geodesic plotting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_geodesic_coordinates(solution: "GeodesicSolution"):
    """Plot geodesic coordinates against the integration parameter."""

    plt = _pyplot()
    fig, ax = plt.subplots()
    for i in range(solution.x.shape[1]):
        ax.plot(solution.t, solution.x[:, i], label=f"x{i}")
    ax.set_xlabel("t")
    ax.set_ylabel("coordinate")
    ax.legend()
    return fig, ax


def plot_phase_component(solution: "GeodesicSolution", i: int):
    """Plot coordinate component x_i against velocity component v_i."""

    if i < 0 or i >= solution.x.shape[1]:
        raise IndexError("component index out of range.")
    plt = _pyplot()
    fig, ax = plt.subplots()
    ax.plot(solution.x[:, i], solution.v[:, i])
    ax.set_xlabel(f"x{i}")
    ax.set_ylabel(f"v{i}")
    return fig, ax


def plot_embedded_geodesic(surface: "ParametrizedMap", solution: "GeodesicSolution"):
    """Plot a geodesic embedded by a parametrized map."""

    points = surface.embed_geodesic(solution)
    plt = _pyplot()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d" if points.shape[1] == 3 else None)
    if points.shape[1] == 3:
        ax.plot(points[:, 0], points[:, 1], points[:, 2])
    else:
        ax.plot(points[:, 0], points[:, 1])
    return fig, ax


def plot_surface_with_geodesic(
    surface: "ParametrizedMap",
    solution: "GeodesicSolution",
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    resolution: int = 50,
):
    """Plot a Euclidean R3 parametrized surface with an embedded geodesic."""

    if surface.dimension != 2 or surface.ambient_dimension != 3:
        raise ValueError("surface plotting currently requires a 2D surface in R3.")

    plt = _pyplot()
    u_values = np.linspace(u_range[0], u_range[1], resolution)
    v_values = np.linspace(v_range[0], v_range[1], resolution)
    uu, vv = np.meshgrid(u_values, v_values)
    grid = np.column_stack([uu.ravel(), vv.ravel()])
    surface_points = surface.evaluate(grid).reshape(resolution, resolution, 3)
    geodesic_points = surface.embed_geodesic(solution)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        surface_points[:, :, 0],
        surface_points[:, :, 1],
        surface_points[:, :, 2],
        alpha=0.35,
        linewidth=0,
    )
    ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], geodesic_points[:, 2], color="black")
    return fig, ax


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Plotting requires matplotlib. Install with: python -m pip install "crr[viz]"') from exc
    return plt


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.parametrized import ParametrizedMap
    from crr.numeric.geodesic_solver import GeodesicSolution
