"""2D scalar, density, and vector field visualization."""

from __future__ import annotations

import numpy as np
import sympy as sp

from crr.visualization.utils import (
    get_figure_and_axis,
    lambdify_grid,
    make_grid_2d,
    maybe_show,
    save_figure,
)


def plot_scalar_field_2d(
    scalar,
    coordinates: tuple[sp.Symbol, sp.Symbol] | list[sp.Symbol],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: int = 100,
    params: dict[sp.Symbol, object] | None = None,
    ax=None,
    show: bool = True,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot a scalar field on a 2D coordinate grid."""

    fig, ax = get_figure_and_axis(ax=ax)
    X, Y = make_grid_2d(x_range, y_range, resolution=resolution)
    Z = lambdify_grid(scalar, coordinates, (X, Y), params=params)
    image = ax.pcolormesh(X, Y, Z, shading="auto")
    fig.colorbar(image, ax=ax)
    ax.set_xlabel(str(coordinates[0]))
    ax.set_ylabel(str(coordinates[1]))
    ax.set_title(title or "Scalar field")
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax


def plot_density_2d(*args, **kwargs):
    """Plot a density field on a 2D coordinate grid."""

    kwargs.setdefault("title", "Density")
    return plot_scalar_field_2d(*args, **kwargs)


def plot_vector_field_2d(
    vector,
    coordinates: tuple[sp.Symbol, sp.Symbol] | list[sp.Symbol],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: int = 25,
    params: dict[sp.Symbol, object] | None = None,
    ax=None,
    show: bool = True,
    normalize: bool = False,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot a 2D vector field with quiver arrows."""

    if len(vector) != 2:
        raise ValueError("A 2D vector field requires two components.")
    fig, ax = get_figure_and_axis(ax=ax)
    X, Y = make_grid_2d(x_range, y_range, resolution=resolution)
    U = lambdify_grid(vector[0], coordinates, (X, Y), params=params)
    V = lambdify_grid(vector[1], coordinates, (X, Y), params=params)
    if normalize:
        magnitude = np.sqrt(U**2 + V**2)
        magnitude[magnitude == 0] = 1
        U = U / magnitude
        V = V / magnitude
    ax.quiver(X, Y, U, V)
    ax.set_xlabel(str(coordinates[0]))
    ax.set_ylabel(str(coordinates[1]))
    ax.set_title(title or "Vector field")
    save_figure(fig, save_path)
    maybe_show(fig, show=show)
    return fig, ax
