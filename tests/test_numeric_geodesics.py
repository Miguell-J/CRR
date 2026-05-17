import numpy as np
import pytest
import sympy as sp

from crr import Manifold, Metric, ParametrizedMap, solve_geodesic


def test_euclidean_plane_geodesic_is_line():
    x, y = sp.symbols("x y")
    metric = Metric(Manifold("R2", 2, [x, y]), [[1, 0], [0, 1]])

    solution = metric.solve_geodesic([0, 0], [1, 2], (0, 3), num_points=50)

    assert solution.success
    final_x, final_v = solution.final_state()
    np.testing.assert_allclose(final_x, [3, 6], atol=1e-8)
    np.testing.assert_allclose(final_v, [1, 2], atol=1e-8)


def test_unit_sphere_equator_geodesic_keeps_theta_and_energy():
    theta, phi = sp.symbols("theta phi")
    metric = Metric(Manifold("S2", 2, [theta, phi]), [[1, 0], [0, sp.sin(theta) ** 2]])

    solution = solve_geodesic(
        metric,
        x0=[np.pi / 2, 0],
        v0=[0, 1],
        t_span=(0, 6),
        num_points=120,
    )

    assert solution.success
    np.testing.assert_allclose(solution.x[:, 0], np.pi / 2, atol=1e-7)
    energy = solution.energy()
    np.testing.assert_allclose(energy, energy[0], atol=1e-7)


def test_cylinder_geodesic_coordinates_are_linear():
    theta, z = sp.symbols("theta z")
    radius = 2
    metric = Metric(Manifold("Cylinder", 2, [theta, z]), [[radius**2, 0], [0, 1]])

    solution = metric.solve_geodesic([0, 1], [0.25, -0.5], (0, 4), num_points=80)

    assert solution.success
    np.testing.assert_allclose(solution.x[:, 0], 0.25 * solution.t, atol=1e-8)
    np.testing.assert_allclose(solution.x[:, 1], 1 - 0.5 * solution.t, atol=1e-8)
    np.testing.assert_allclose(solution.v, np.column_stack([np.full_like(solution.t, 0.25), np.full_like(solution.t, -0.5)]), atol=1e-8)


def test_geodesic_solution_shapes_and_helpers():
    x, y = sp.symbols("x y")
    metric = Metric(Manifold("R2", 2, [x, y]), [[1, 0], [0, 1]])

    solution = metric.solve_geodesic([0, 0], [1, 0], (0, 1), num_points=11)
    t, coords, velocities = solution.to_numpy()
    final_x, final_v = solution.final_state()

    assert t.shape == (11,)
    assert coords.shape == (11, 2)
    assert velocities.shape == (11, 2)
    assert final_x.shape == (2,)
    assert final_v.shape == (2,)
    assert solution.coordinates().shape == (11, 2)
    assert solution.velocities().shape == (11, 2)
    assert solution.energy().shape == (11,)


def test_parametrized_map_embeds_sphere_equator_geodesic():
    theta, phi = sp.symbols("theta phi")
    sphere = ParametrizedMap(
        name="UnitSphere",
        parameters=[theta, phi],
        components=[
            sp.sin(theta) * sp.cos(phi),
            sp.sin(theta) * sp.sin(phi),
            sp.cos(theta),
        ],
    )
    metric = sphere.pullback_metric(simplify=True)

    solution = metric.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 2), num_points=50)
    embedded = sphere.embed_geodesic(solution)

    assert embedded.shape == (50, 3)
    np.testing.assert_allclose(np.linalg.norm(embedded, axis=1), 1, atol=1e-8)


def test_geodesic_solution_reports_energy_drift():
    x, y = sp.symbols("x y")
    metric = Metric(Manifold("R2", 2, [x, y]), [[1, 0], [0, 1]])

    solution = metric.solve_geodesic([0, 0], [1, 0], (0, 1), num_points=11)

    np.testing.assert_allclose(solution.energy_drift(), 0, atol=1e-12)
    assert solution.max_energy_drift() == pytest.approx(0, abs=1e-12)


def test_solve_geodesic_rejects_invalid_numeric_inputs():
    x, y = sp.symbols("x y")
    metric = Metric(Manifold("R2", 2, [x, y]), [[1, 0], [0, 1]])

    with pytest.raises(ValueError, match="t_span"):
        metric.solve_geodesic([0, 0], [1, 0], (0, float("inf")))
    with pytest.raises(ValueError, match="distinct"):
        metric.solve_geodesic([0, 0], [1, 0], (1, 1))
    with pytest.raises(ValueError, match="finite"):
        metric.solve_geodesic([0, float("nan")], [1, 0], (0, 1))
