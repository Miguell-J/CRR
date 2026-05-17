"""Lightweight geodesic plotting helpers."""

from __future__ import annotations

from crr.visualization.utils import get_figure_and_axis, maybe_show, save_figure


def plot_geodesic_coordinates(
    solution: "GeodesicSolution",
    show: bool = True,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot geodesic coordinates against the integration parameter."""

    fig, ax = get_figure_and_axis(ax=ax)
    for i in range(solution.x.shape[1]):
        ax.plot(solution.t, solution.x[:, i], label=f"x{i}")
    ax.set_xlabel("t")
    ax.set_ylabel("coordinate")
    ax.set_title(title or "Geodesic coordinates")
    ax.legend()
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def plot_geodesic_velocities(
    solution: "GeodesicSolution",
    show: bool = True,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot geodesic velocity components against the integration parameter."""

    fig, ax = get_figure_and_axis(ax=ax)
    for i in range(solution.v.shape[1]):
        ax.plot(solution.t, solution.v[:, i], label=f"v{i}")
    ax.set_xlabel("t")
    ax.set_ylabel("velocity")
    ax.set_title(title or "Geodesic velocities")
    ax.legend()
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def plot_geodesic_energy(
    solution: "GeodesicSolution",
    show: bool = True,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot geodesic energy."""

    fig, ax = get_figure_and_axis(ax=ax)
    ax.plot(solution.t, solution.energy())
    ax.set_xlabel("t")
    ax.set_ylabel("energy")
    ax.set_title(title or "Geodesic energy")
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def plot_speed_squared(
    solution: "GeodesicSolution",
    show: bool = True,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot squared speed."""

    fig, ax = get_figure_and_axis(ax=ax)
    ax.plot(solution.t, solution.speed_squared())
    ax.set_xlabel("t")
    ax.set_ylabel("speed squared")
    ax.set_title(title or "Speed squared")
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def plot_phase_component(
    solution: "GeodesicSolution",
    i: int,
    show: bool = True,
    ax=None,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot coordinate component x_i against velocity component v_i."""

    if i < 0 or i >= solution.x.shape[1]:
        raise IndexError("component index out of range.")
    fig, ax = get_figure_and_axis(ax=ax)
    ax.plot(solution.x[:, i], solution.v[:, i])
    ax.set_xlabel(f"x{i}")
    ax.set_ylabel(f"v{i}")
    ax.set_title(title or f"Phase component {i}")
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def plot_embedded_geodesic(surface: "ParametrizedMap", solution: "GeodesicSolution", **kwargs):
    """Plot a geodesic embedded by a parametrized map."""

    from crr.visualization.surfaces import plot_geodesic

    return plot_geodesic(surface, solution, **kwargs)


def plot_surface_with_geodesic(
    surface: "ParametrizedMap",
    solution: "GeodesicSolution",
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    resolution: int = 50,
    **kwargs,
):
    """Plot a Euclidean R3 parametrized surface with an embedded geodesic."""

    from crr.visualization.surfaces import plot_surface_with_geodesic as _plot

    return _plot(surface, solution, u_range, v_range, resolution=resolution, **kwargs)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.parametrized import ParametrizedMap
    from crr.numeric.geodesic_solver import GeodesicSolution
