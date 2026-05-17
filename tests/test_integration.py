import numpy as np
import sympy as sp

from crr import Manifold, Metric, ParametrizedMap, gauss_bonnet_check


def unit_sphere():
    theta, phi = sp.symbols("theta phi")
    return theta, phi, ParametrizedMap(
        name="UnitSphere",
        parameters=[theta, phi],
        components=[
            sp.sin(theta) * sp.cos(phi),
            sp.sin(theta) * sp.sin(phi),
            sp.cos(theta),
        ],
    )


def test_unit_sphere_area_integrates_to_four_pi():
    theta, phi, sphere = unit_sphere()

    area = sphere.integrate_area([(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], simplify=True)

    assert sp.simplify(area - 4 * sp.pi) == 0


def test_unit_sphere_total_gaussian_curvature_integrates_to_four_pi():
    theta, phi, sphere = unit_sphere()

    total_curvature = sphere.integrate_gaussian_curvature(
        [(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)],
        simplify=True,
    )

    assert sp.simplify(total_curvature - 4 * sp.pi) == 0


def test_gauss_bonnet_sphere_passes_for_chi_two():
    theta, phi, sphere = unit_sphere()

    result = gauss_bonnet_check(
        sphere,
        [(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)],
        euler_characteristic=2,
    )

    assert result.passed is True
    assert result.difference == 0


def test_torus_total_gaussian_curvature_integrates_to_zero():
    u, v = sp.symbols("u v")
    torus = ParametrizedMap(
        name="Torus",
        parameters=[u, v],
        components=[
            (2 + sp.cos(v)) * sp.cos(u),
            (2 + sp.cos(v)) * sp.sin(u),
            sp.sin(v),
        ],
    )

    total_curvature = torus.integrate_gaussian_curvature(
        [(u, 0, 2 * sp.pi), (v, 0, 2 * sp.pi)],
        simplify=True,
    )

    assert sp.simplify(total_curvature) == 0


def test_gauss_bonnet_torus_passes_for_chi_zero():
    u, v = sp.symbols("u v")
    torus = ParametrizedMap(
        name="Torus",
        parameters=[u, v],
        components=[
            (2 + sp.cos(v)) * sp.cos(u),
            (2 + sp.cos(v)) * sp.sin(u),
            sp.sin(v),
        ],
    )

    result = gauss_bonnet_check(
        torus,
        [(u, 0, 2 * sp.pi), (v, 0, 2 * sp.pi)],
        euler_characteristic=0,
    )

    assert result.passed is True
    assert result.difference == 0


def test_cylinder_area_integrates_to_expected_formula():
    theta, z, radius, height = sp.symbols("theta z R h", positive=True)
    cylinder = ParametrizedMap(
        name="Cylinder",
        parameters=[theta, z],
        components=[
            radius * sp.cos(theta),
            radius * sp.sin(theta),
            z,
        ],
    )

    area = cylinder.integrate_area([(theta, 0, 2 * sp.pi), (z, 0, height)], simplify=True)

    assert sp.simplify(area - 2 * sp.pi * radius * height) == 0


def test_euclidean_rectangle_volume_integrates_to_area():
    x, y, a, b = sp.symbols("x y a b", positive=True)
    metric = Metric(Manifold("R2", 2, [x, y]), [[1, 0], [0, 1]])

    area = metric.integrate_volume([(x, 0, a), (y, 0, b)], simplify=True)

    assert sp.simplify(area - a * b) == 0


def test_scalar_integral_on_unit_square():
    x, y = sp.symbols("x y")
    metric = Metric(Manifold("R2", 2, [x, y]), [[1, 0], [0, 1]])

    value = metric.integrate_scalar(x + y, [(x, 0, 1), (y, 0, 1)], simplify=True)

    assert sp.simplify(value - 1) == 0


def test_numeric_scalar_integral_matches_symbolic_result():
    x, y = sp.symbols("x y")
    metric = Metric(Manifold("R2", 2, [x, y]), [[1, 0], [0, 1]])

    symbolic = metric.integrate_scalar(x**2 + y, [(x, 0, 1), (y, 0, 2)], simplify=True)
    numeric = metric.integrate_scalar_numeric(x**2 + y, [(x, 0, 1), (y, 0, 2)])

    np.testing.assert_allclose(numeric, float(symbolic), atol=1e-10)
