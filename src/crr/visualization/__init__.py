"""Optional visualization helpers for CRR."""

from crr.visualization.geodesics import (
    plot_embedded_geodesic,
    plot_geodesic_coordinates,
    plot_geodesic_energy,
    plot_geodesic_velocities,
    plot_phase_component,
    plot_speed_squared,
    plot_surface_with_geodesic,
)
from crr.visualization.fields import plot_density_2d, plot_scalar_field_2d, plot_vector_field_2d
from crr.visualization.forms import plot_form_2d
from crr.visualization.surfaces import plot_curvature, plot_geodesic, plot_surface
from crr.visualization.utils import (
    apply_params,
    lambdify_grid,
    make_grid_2d,
    maybe_show,
    require_matplotlib,
    save_figure,
)

__all__ = [
    "plot_geodesic_coordinates",
    "plot_geodesic_velocities",
    "plot_geodesic_energy",
    "plot_speed_squared",
    "plot_phase_component",
    "plot_embedded_geodesic",
    "plot_surface_with_geodesic",
    "plot_surface",
    "plot_curvature",
    "plot_geodesic",
    "plot_scalar_field_2d",
    "plot_density_2d",
    "plot_vector_field_2d",
    "plot_form_2d",
    "require_matplotlib",
    "make_grid_2d",
    "lambdify_grid",
    "apply_params",
    "maybe_show",
    "save_figure",
]
