"""Shared visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp


def require_matplotlib():
    """Import matplotlib.pyplot or raise a friendly optional-dependency error."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Visualization requires matplotlib. Install with: pip install -e ".[viz]"') from exc
    return plt


def make_grid_2d(
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two meshgrids over a rectangular parameter domain."""

    u = np.linspace(float(u_range[0]), float(u_range[1]), resolution)
    v = np.linspace(float(v_range[0]), float(v_range[1]), resolution)
    return np.meshgrid(u, v)


def apply_params(expr: Any, params: dict[sp.Symbol, object] | None = None) -> Any:
    """Substitute parameter values into a SymPy expression or sequence."""

    if params is None:
        return expr
    if isinstance(expr, (list, tuple)):
        return [apply_params(item, params=params) for item in expr]
    return sp.sympify(expr).subs(params)


def lambdify_grid(
    expr: Any,
    symbols: tuple[sp.Symbol, sp.Symbol] | list[sp.Symbol],
    grids: tuple[np.ndarray, np.ndarray],
    params: dict[sp.Symbol, object] | None = None,
) -> np.ndarray:
    """Evaluate a SymPy expression on two coordinate grids."""

    prepared = apply_params(expr, params=params)
    function = sp.lambdify(tuple(symbols), prepared, modules="numpy")
    values = np.asarray(function(*grids), dtype=float)
    if values.ndim == 0:
        values = np.full_like(grids[0], float(values), dtype=float)
    return values


def maybe_show(fig: Any, show: bool = True):
    """Show a matplotlib figure when requested."""

    if show:
        require_matplotlib().show()
    return fig


def save_figure(fig: Any, path: str | Path | None, dpi: int = 150) -> None:
    """Save a figure if a path is provided."""

    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")


def get_figure_and_axis(ax: Any = None, projection: str | None = None):
    """Return a ``(fig, ax)`` pair, creating one if needed."""

    plt = require_matplotlib()
    if ax is not None:
        return ax.figure, ax
    fig = plt.figure()
    if projection == "3d":
        return fig, fig.add_subplot(111, projection="3d")
    return fig, fig.add_subplot(111)
