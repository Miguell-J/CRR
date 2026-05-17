import sympy as sp

from crr import DifferentialForm, Manifold, Metric, ParametrizedMap, Tensor
from crr.audit import compare_benchmarks_to_baseline, load_benchmark_baseline, run_benchmarks, save_benchmark_baseline
from crr.display import display_object, object_to_latex, object_to_markdown
from crr.gauge import abelian_gauge_potential, identity_inner_product, su2_connection, yang_mills_action_density
from crr.geometry.global_invariants import gauss_bonnet_check
from crr.lie import AlgebraValuedForm, su2_algebra
from crr.presets import sphere_metric, sphere_surface


def test_public_object_display_consistency():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    metric = Metric(manifold, [[1, 0], [0, 1]])
    tensor = Tensor([[1, 0], [0, 0]], name="T")
    surface = ParametrizedMap("Plane", [x, y], [x, y, 0])
    form = x * DifferentialForm.basis(manifold, 1)
    solution = metric.solve_geodesic([0, 0], [1, 0], (0, 1), num_points=5)
    gb = gauss_bonnet_check(sphere_surface(radius=1), [(sp.Symbol("theta"), 0, sp.pi), (sp.Symbol("phi"), 0, 2 * sp.pi)], euler_characteristic=2)
    algebra = su2_algebra()
    element = algebra.basis_element(0)
    avf = AlgebraValuedForm(algebra, [form, DifferentialForm.zero(manifold, 1), DifferentialForm.zero(manifold, 1)])
    gauge = su2_connection(manifold, [form, DifferentialForm.basis(manifold, 0), DifferentialForm.zero(manifold, 1)])
    curvature = gauge.curvature()

    objects = [manifold, metric, tensor, surface, form, solution, gb, algebra, element, avf, gauge, curvature]
    for obj in objects:
        assert repr(obj)
        assert str(obj)
        assert hasattr(obj, "to_latex")
        assert isinstance(obj.to_latex(), str)
        assert obj.to_latex()
        assert hasattr(obj, "display")
        assert isinstance(obj.display(), str)

    markdown_objects = [manifold, metric, tensor, surface, form, solution, gb, algebra, element, avf, gauge, curvature]
    for obj in markdown_objects:
        assert hasattr(obj, "to_markdown")
        assert isinstance(obj.to_markdown(), str)


def test_backward_compatible_display_aliases():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    tensor = Tensor([[1, 0], [0, 0]], name="T")
    algebra = su2_algebra()

    assert tensor.to_markdown() == tensor.to_markdown_table()
    assert tensor.display_nonzero() == tensor.display()
    assert algebra.to_markdown() == algebra.to_markdown_table()
    assert DifferentialForm.basis(manifold, 0).display_components()


def test_display_helper_fallbacks():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    form = x * DifferentialForm.basis(manifold, 1)

    assert object_to_latex(form) == form.to_latex()
    assert object_to_markdown(form) == form.to_markdown()
    assert display_object(form) == form.to_latex()

    class Plain:
        def __str__(self):
            return "plain"

    assert object_to_latex(Plain()) == "plain"
    assert object_to_markdown(Plain()) == "plain"
    assert display_object(Plain()) == "plain"


def test_benchmark_baseline_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"

    results = run_benchmarks(quick=True)
    saved = save_benchmark_baseline(path, quick=True)
    loaded = load_benchmark_baseline(saved)
    comparisons = compare_benchmarks_to_baseline(saved, quick=True, tolerance=1000)

    assert results
    assert saved == path
    assert set(loaded) >= {"sphere_curvature", "gauge_su2_curvature"}
    assert comparisons
    assert all(comparison.success for comparison in comparisons)
    assert all(not comparison.regression for comparison in comparisons)


def test_gauge_identity_inner_product_and_action_density():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    metric = Metric(manifold, [[1, 0], [0, 1]])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)

    u1 = abelian_gauge_potential(manifold, x * dy)
    assert identity_inner_product(u1.algebra).shape == (1, 1)
    assert yang_mills_action_density(u1.curvature(), metric, simplify=True).degree == 2

    su2 = su2_connection(manifold, [dx, dy, DifferentialForm.zero(manifold, 1)])
    assert identity_inner_product(su2.algebra).shape == (3, 3)
    assert yang_mills_action_density(su2.curvature(), metric, inner_product=identity_inner_product(su2.algebra), simplify=True).degree == 2


def test_stabilization_docs_exist():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]

    assert (root / "docs" / "SIMPLIFICATION_AND_CACHING.md").exists()
    assert (root / "docs" / "API_STABILITY.md").exists()
