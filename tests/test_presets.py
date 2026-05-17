import numpy as np
import sympy as sp

from crr.presets import (
    alcubierre_metric,
    anti_de_sitter_metric,
    cylinder_surface,
    de_sitter_metric,
    eddington_finkelstein_metric,
    euclidean_metric,
    flrw_metric,
    godel_metric,
    hyperbolic_half_plane_metric,
    kerr_metric,
    kruskal_metric,
    minkowski_metric,
    poincare_disk_metric,
    reissner_nordstrom_metric,
    robertson_walker_metric,
    schwarzschild_metric,
    sphere_metric,
    sphere_surface,
    torus_metric,
    torus_surface,
)


def test_sphere_surface_curvature_area_and_total_gaussian_curvature():
    R = sp.symbols("R", positive=True)
    sphere = sphere_surface(radius=R)
    theta, phi = sphere.parameters
    metric = sphere.pullback_metric(simplify=True)

    assert sp.simplify(metric.scalar_curvature(simplify=True) - 2 / R**2) == 0
    assert sp.simplify(sphere.integrate_area([(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], simplify=True) - 4 * sp.pi * R**2) == 0
    assert sp.simplify(sphere.integrate_gaussian_curvature([(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], simplify=True) - 4 * sp.pi) == 0


def test_sphere_metric_scalar_curvature():
    R = sp.symbols("R", positive=True)
    metric = sphere_metric(radius=R)

    assert sp.simplify(metric.scalar_curvature(simplify=True) - 2 / R**2) == 0


def test_cylinder_surface_area_and_curvature():
    R, h = sp.symbols("R h", positive=True)
    cylinder = cylinder_surface(radius=R)
    theta, z = cylinder.parameters

    assert sp.simplify(cylinder.pullback_metric(simplify=True).scalar_curvature(simplify=True)) == 0
    assert sp.simplify(cylinder.gaussian_curvature_extrinsic(simplify=True)) == 0
    assert sp.simplify(cylinder.integrate_area([(theta, 0, 2 * sp.pi), (z, 0, h)], simplify=True) - 2 * sp.pi * R * h) == 0


def test_torus_surface_and_metric_scalar_curvatures_and_total_gaussian_curvature():
    u, v = sp.symbols("u v")
    R, r = sp.symbols("R r", positive=True)
    expected = 2 * sp.cos(v) / (r * (R + r * sp.cos(v)))

    surface = torus_surface(major_radius=R, minor_radius=r, coordinates=[u, v])
    metric = torus_metric(major_radius=R, minor_radius=r, coordinates=[u, v])

    assert sp.simplify(surface.pullback_metric(simplify=True).scalar_curvature(simplify=True) - expected) == 0
    assert sp.simplify(metric.scalar_curvature(simplify=True) - expected) == 0
    standard_surface = torus_surface(major_radius=2, minor_radius=1, coordinates=[u, v])
    assert sp.simplify(standard_surface.integrate_gaussian_curvature([(u, 0, 2 * sp.pi), (v, 0, 2 * sp.pi)], simplify=True)) == 0


def test_hyperbolic_half_plane_metric_scalar_curvature():
    assert sp.simplify(hyperbolic_half_plane_metric().scalar_curvature(simplify=True) + 2) == 0


def test_poincare_disk_metric_scalar_curvature():
    assert sp.simplify(poincare_disk_metric().scalar_curvature(simplify=True) + 2) == 0


def test_euclidean_metric_scalar_curvature_in_dimensions_two_and_three():
    assert sp.simplify(euclidean_metric(dim=2).scalar_curvature(simplify=True)) == 0
    assert sp.simplify(euclidean_metric(dim=3).scalar_curvature(simplify=True)) == 0


def test_minkowski_metric_scalar_curvature():
    assert sp.simplify(minkowski_metric().scalar_curvature(simplify=True)) == 0


def test_schwarzschild_metric_representative_ricci_components_vanish():
    metric = schwarzschild_metric()
    ricci = metric.ricci_tensor(simplify=True)

    for index in [(0, 0), (1, 1), (2, 2), (3, 3), (0, 1)]:
        assert sp.simplify(ricci[index]) == 0


def test_flrw_metric_builds_and_scalar_curvature_computes():
    metric = flrw_metric()
    scalar = metric.scalar_curvature()

    assert scalar is not None
    assert scalar != 0


def test_preset_sphere_equator_geodesic_keeps_theta():
    metric = sphere_metric(radius=1)

    solution = metric.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 4), num_points=80)

    assert solution.success
    np.testing.assert_allclose(solution.x[:, 0], np.pi / 2, atol=1e-7)


def test_extended_relativity_presets_build_with_expected_shapes():
    for factory in [
        kerr_metric,
        godel_metric,
        robertson_walker_metric,
        kruskal_metric,
        eddington_finkelstein_metric,
        reissner_nordstrom_metric,
        de_sitter_metric,
        anti_de_sitter_metric,
        alcubierre_metric,
    ]:
        metric = factory()
        assert metric.components.shape == (4, 4)
        assert metric.determinant() != 0


def test_reissner_nordstrom_and_eddington_finkelstein_scalar_curvature_vanish():
    assert sp.simplify(reissner_nordstrom_metric().scalar_curvature(simplify=True)) == 0
    assert sp.simplify(eddington_finkelstein_metric().scalar_curvature(simplify=True)) == 0


def test_de_sitter_and_anti_de_sitter_scalar_curvatures():
    L = sp.symbols("L", positive=True)

    assert sp.simplify(de_sitter_metric(curvature_radius=L).scalar_curvature(simplify=True) - 12 / L**2) == 0
    assert sp.simplify(anti_de_sitter_metric(curvature_radius=L).scalar_curvature(simplify=True) + 12 / L**2) == 0


def test_kerr_preset_reduces_to_schwarzschild_components_when_spin_zero():
    t, r, theta, phi = sp.symbols("t r theta phi")
    M = sp.symbols("M", positive=True)
    kerr = kerr_metric(mass=M, spin=0, coordinates=[t, r, theta, phi])
    schwarzschild = schwarzschild_metric(mass=M, coordinates=[t, r, theta, phi])

    assert sp.simplify(kerr.components[0, 0] - schwarzschild.components[0, 0]) == 0
    assert sp.simplify(kerr.components[1, 1] - schwarzschild.components[1, 1]) == 0
    assert sp.simplify(kerr.components[0, 3]) == 0


def test_alcubierre_metric_reduces_to_minkowski_when_velocity_zero():
    metric = alcubierre_metric(velocity=0)

    assert metric.components == minkowski_metric().components
