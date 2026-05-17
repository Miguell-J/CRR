import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crr import DifferentialForm, Manifold
from crr.presets import sphere_metric, sphere_surface, torus_surface
from crr.visualization import (
    plot_form_2d,
    plot_scalar_field_2d,
    plot_vector_field_2d,
    require_matplotlib,
)


def _close(fig):
    plt.close(fig)


def test_require_matplotlib_returns_pyplot():
    plt_module = require_matplotlib()
    assert hasattr(plt_module, "figure")


def test_surface_plotting_returns_figure_and_axis():
    sphere = sphere_surface()
    fig, ax = sphere.plot_surface((0, np.pi), (0, 2 * np.pi), resolution=8, show=False)

    assert fig is not None
    assert ax is not None
    _close(fig)


def test_curvature_plotting_returns_figure_and_axis():
    torus = torus_surface(major_radius=2, minor_radius=0.7)
    fig, ax = torus.plot_curvature(
        u_range=(0, 2 * np.pi),
        v_range=(0, 2 * np.pi),
        resolution=8,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    _close(fig)


def test_geodesic_plotting_returns_figure_and_axis():
    surface = sphere_surface()
    metric = sphere_metric(radius=1)
    solution = metric.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 0.5), num_points=12)

    for method in [
        solution.plot_coordinates,
        solution.plot_velocities,
        solution.plot_energy,
        solution.plot_speed_squared,
    ]:
        fig, ax = method(show=False)
        assert fig is not None
        assert ax is not None
        _close(fig)

    fig, ax = solution.plot_phase_component(0, show=False)
    assert fig is not None
    assert ax is not None
    _close(fig)

    fig, ax = surface.plot_geodesic(solution, show=False)
    assert fig is not None
    assert ax is not None
    _close(fig)

    fig, ax = surface.plot_surface_with_geodesic(
        solution,
        (0, np.pi),
        (0, 2 * np.pi),
        resolution=8,
        show=False,
    )
    assert fig is not None
    assert ax is not None
    _close(fig)


def test_field_plotting_returns_figure_and_axis():
    x, y = sp.symbols("x y")
    fig, ax = plot_scalar_field_2d(x**2 - y**2, [x, y], (-1, 1), (-1, 1), resolution=10, show=False)
    assert fig is not None
    assert ax is not None
    _close(fig)

    fig, ax = plot_vector_field_2d([-y, x], [x, y], (-1, 1), (-1, 1), resolution=8, show=False)
    assert fig is not None
    assert ax is not None
    _close(fig)


def test_form_plotting_returns_figure_and_axis():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    forms = [
        DifferentialForm.scalar(manifold, x**2 - y**2),
        -y * dx + x * dy,
        sp.sin(x * y) * dx.wedge(dy),
    ]

    for form in forms:
        fig, ax = plot_form_2d(form, (-1, 1), (-1, 1), resolution=8, show=False)
        assert fig is not None
        assert ax is not None
        _close(fig)


def test_save_path_writes_png(tmp_path):
    path = tmp_path / "sphere.png"
    fig, ax = sphere_surface().plot_surface(
        (0, np.pi),
        (0, 2 * np.pi),
        resolution=8,
        show=False,
        save_path=path,
    )

    assert fig is not None
    assert ax is not None
    assert path.exists()
    assert path.stat().st_size > 0
    _close(fig)


def test_visualization_notebook_scripts_smoke_run():
    scripts = [
        "notebooks/08_visualizing_surfaces.py",
        "notebooks/09_visualizing_curvature.py",
        "notebooks/10_visualizing_geodesics.py",
        "notebooks/11_visualizing_forms.py",
    ]

    for script in scripts:
        completed = subprocess.run([sys.executable, script], check=True, capture_output=True, text=True)
        assert completed.stdout
