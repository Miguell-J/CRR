"""Visualization helpers for differential forms."""

from __future__ import annotations

from crr.visualization.fields import plot_density_2d, plot_scalar_field_2d, plot_vector_field_2d


def plot_form_2d(
    form: "DifferentialForm",
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: int = 30,
    params=None,
    ax=None,
    show: bool = True,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot a 0-, 1-, or 2-form on a 2D coordinate chart."""

    if form.manifold.dimension != 2:
        raise ValueError("plot_form_2d requires a form on a 2D manifold.")
    coordinates = form.manifold.coordinates
    if form.degree == 0:
        return plot_scalar_field_2d(
            form.components.get((), 0),
            coordinates,
            x_range,
            y_range,
            resolution=resolution,
            params=params,
            ax=ax,
            show=show,
            title=title or "0-form",
            save_path=save_path,
        )
    if form.degree == 1:
        return plot_vector_field_2d(
            [form[(0,)], form[(1,)]],
            coordinates,
            x_range,
            y_range,
            resolution=resolution,
            params=params,
            ax=ax,
            show=show,
            title=title or "1-form components",
            save_path=save_path,
        )
    if form.degree == 2:
        return plot_density_2d(
            form[(0, 1)],
            coordinates,
            x_range,
            y_range,
            resolution=resolution,
            params=params,
            ax=ax,
            show=show,
            title=title or "2-form density",
            save_path=save_path,
        )
    raise NotImplementedError("Only 0-, 1-, and 2-forms on 2D manifolds can be plotted.")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.forms import DifferentialForm
