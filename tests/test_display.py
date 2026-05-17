import subprocess
import sys

import sympy as sp

from crr import DifferentialForm, Manifold
from crr.display.pretty import display_latex, format_nonzero_components
from crr.presets import euclidean_metric, sphere_surface


def test_display_latex_fallback_returns_string():
    assert display_latex(r"x^2") == r"x^2"


def test_tensor_markdown_table_contains_components():
    x, y = sp.symbols("x y")
    metric = euclidean_metric(dim=2, coordinates=[x, y])
    ricci = metric.ricci_tensor(simplify=True)
    gamma = metric.christoffel_symbols()

    assert isinstance(gamma.to_markdown_table(), str)
    assert "| Component | Value |" in gamma.to_markdown_table()
    assert ricci.to_latex_components() == "0"


def test_format_nonzero_components_for_tensor():
    theta, phi = sp.symbols("theta phi")
    sphere = sphere_surface(coordinates=[theta, phi])
    gamma = sphere.pullback_metric(simplify=True).christoffel_symbols(simplify=True)
    formatted = format_nonzero_components(gamma, latex=False)

    assert "Gamma" in formatted


def test_differential_form_latex_and_markdown():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    form = (x + y) * DifferentialForm.basis(manifold, 0).wedge(DifferentialForm.basis(manifold, 1))

    assert "wedge" in form.to_latex()
    markdown = form.to_markdown()
    assert "| Component | Value |" in markdown
    assert "(0, 1)" in markdown


def test_repr_and_str_are_useful():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    metric = euclidean_metric(dim=2, coordinates=[x, y])
    form = DifferentialForm.basis(manifold, 0)
    surface = sphere_surface()
    solution = metric.solve_geodesic([0, 0], [1, 0], (0, 1), num_points=4)

    for obj in [manifold, metric, form, surface, solution]:
        assert repr(obj)
        assert str(obj)


def test_notebook_scripts_smoke_run():
    scripts = [
        "notebooks/01_sphere_geometry.py",
        "notebooks/02_torus_geometry.py",
        "notebooks/03_gauss_bonnet.py",
        "notebooks/04_numeric_geodesics.py",
        "notebooks/05_differential_forms.py",
        "notebooks/06_maxwell_forms.py",
        "notebooks/07_intro_relativity.py",
    ]

    for script in scripts:
        completed = subprocess.run([sys.executable, script], check=True, capture_output=True, text=True)
        assert completed.stdout
