import sympy as sp
import pytest

from crr import Manifold, Metric, ParametrizedMap


def test_euclidean_plane_is_flat():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    metric = Metric(manifold, [[1, 0], [0, 1]])

    assert metric.christoffel_symbols().nonzero_components() == {}
    assert sp.simplify(metric.scalar_curvature()) == 0


def test_unit_two_sphere_scalar_curvature_is_two():
    theta, phi = sp.symbols("theta phi")
    manifold = Manifold("S2", 2, [theta, phi])
    metric = Metric(manifold, [[1, 0], [0, sp.sin(theta) ** 2]])

    assert sp.simplify(metric.scalar_curvature(simplify=True) - 2) == 0


def test_poincare_half_plane_scalar_curvature_is_minus_two():
    x, y = sp.symbols("x y", positive=True)
    manifold = Manifold("H2", 2, [x, y])
    metric = Metric(manifold, [[1 / y**2, 0], [0, 1 / y**2]])

    assert sp.simplify(metric.scalar_curvature(simplify=True) + 2) == 0


def test_schwarzschild_representative_ricci_components_vanish():
    t, r, theta, phi, mass = sp.symbols("t r theta phi M", positive=True)
    f = 1 - 2 * mass / r
    manifold = Manifold("Schwarzschild", 4, [t, r, theta, phi])
    metric = Metric(
        manifold,
        [
            [-f, 0, 0, 0],
            [0, 1 / f, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta) ** 2],
        ],
    )

    ricci = metric.ricci_tensor(simplify=True)

    for index in [(0, 0), (1, 1), (2, 2), (3, 3), (0, 1)]:
        assert sp.simplify(ricci[index]) == 0


def test_einstein_tensor_formula():
    theta, phi = sp.symbols("theta phi")
    manifold = Manifold("S2", 2, [theta, phi])
    metric = Metric(manifold, [[1, 0], [0, sp.sin(theta) ** 2]])

    einstein = metric.einstein_tensor(simplify=True)
    ricci = metric.ricci_tensor(simplify=True)
    scalar = metric.scalar_curvature(simplify=True)

    for i in range(metric.dimension):
        for j in range(metric.dimension):
            expected = ricci[i, j] - sp.Rational(1, 2) * metric.components[i, j] * scalar
            assert sp.simplify(einstein[i, j] - expected) == 0


def test_tensor_index_operations_and_trace():
    theta, phi = sp.symbols("theta phi")
    manifold = Manifold("S2", 2, [theta, phi])
    metric = Metric(manifold, [[1, 0], [0, sp.sin(theta) ** 2]])

    ricci = metric.ricci_tensor(simplify=True)
    mixed = ricci.raise_index(metric=metric, index=0, simplify=True)

    assert sp.simplify(mixed[0, 0] - 1) == 0
    assert sp.simplify(mixed[1, 1] - 1) == 0
    assert sp.simplify(ricci.trace(metric=metric, simplify=True) - 2) == 0
    assert ricci.simplify().equals(metric.ricci_tensor(simplify=True))


def test_euclidean_covariant_derivative_metric_and_geodesic_acceleration():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    metric = Metric(manifold, [[1, 0], [0, 1]])

    nabla_metric = metric.covariant_derivative_covariant_2tensor(metric.components, simplify=True)
    assert nabla_metric.nonzero_components() == {}
    assert metric.geodesic_acceleration(simplify=True) == [0, 0]


def test_sphere_metric_compatibility_and_geodesic_acceleration():
    theta, phi = sp.symbols("theta phi")
    v_theta, v_phi = sp.symbols("v_theta v_phi")
    manifold = Manifold("S2", 2, [theta, phi])
    metric = Metric(manifold, [[1, 0], [0, sp.sin(theta) ** 2]])

    assert sp.simplify(metric.scalar_curvature(simplify=True) - 2) == 0

    nabla_metric = metric.covariant_derivative_covariant_2tensor(metric.components, simplify=True)
    assert nabla_metric.nonzero_components() == {}

    acceleration = metric.geodesic_acceleration([v_theta, v_phi], simplify=True)
    assert sp.simplify(acceleration[0] - sp.sin(theta) * sp.cos(theta) * v_phi**2) == 0
    assert sp.simplify(acceleration[1] + 2 * sp.cot(theta) * v_theta * v_phi) == 0


def test_scalar_covariant_derivative_on_euclidean_plane():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    metric = Metric(manifold, [[1, 0], [0, 1]])

    derivative = metric.covariant_derivative_scalar(x**2 + y**2, simplify=True)

    assert derivative.tolist() == [2 * x, 2 * y]


def test_vector_covariant_derivative_on_euclidean_plane_is_jacobian():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    metric = Metric(manifold, [[1, 0], [0, 1]])

    derivative = metric.covariant_derivative_vector([x**2, x * y], simplify=True)

    assert derivative[0, 0] == 2 * x
    assert derivative[0, 1] == 0
    assert derivative[1, 0] == y
    assert derivative[1, 1] == x


def test_embedded_sphere_induced_metric_and_scalar_curvature():
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

    expected = sp.Matrix([[1, 0], [0, sp.sin(theta) ** 2]])
    induced = sphere.induced_metric_components(simplify=True)

    assert induced.equals(expected)
    assert sp.simplify(sphere.pullback_metric(simplify=True).scalar_curvature(simplify=True) - 2) == 0


def test_cylinder_induced_metric_and_scalar_curvature():
    theta, z, radius = sp.symbols("theta z R", positive=True)
    cylinder = ParametrizedMap(
        name="Cylinder",
        parameters=[theta, z],
        components=[
            radius * sp.cos(theta),
            radius * sp.sin(theta),
            z,
        ],
    )

    expected = sp.Matrix([[radius**2, 0], [0, 1]])
    induced = cylinder.induced_metric_components(simplify=True)

    assert induced.equals(expected)
    assert sp.simplify(cylinder.pullback_metric(simplify=True).scalar_curvature(simplify=True)) == 0


def test_torus_induced_metric_determinant_and_area_density():
    u, v, major_radius, minor_radius = sp.symbols("u v R r", positive=True)
    torus = ParametrizedMap(
        name="Torus",
        parameters=[u, v],
        components=[
            (major_radius + minor_radius * sp.cos(v)) * sp.cos(u),
            (major_radius + minor_radius * sp.cos(v)) * sp.sin(u),
            minor_radius * sp.sin(v),
        ],
    )

    expected = sp.Matrix(
        [
            [(major_radius + minor_radius * sp.cos(v)) ** 2, 0],
            [0, minor_radius**2],
        ]
    )
    induced = torus.induced_metric_components(simplify=True)
    metric = torus.pullback_metric(simplify=True)
    expected_determinant = minor_radius**2 * (major_radius + minor_radius * sp.cos(v)) ** 2

    assert induced.equals(expected)
    assert sp.simplify(metric.determinant(simplify=True) - expected_determinant) == 0
    assert sp.simplify(torus.area_density(simplify=True) - sp.sqrt(expected_determinant)) == 0


def test_plane_embedding_into_r3_has_identity_induced_metric():
    u, v = sp.symbols("u v")
    plane = ParametrizedMap(
        name="PlaneInR3",
        parameters=[u, v],
        components=[u, v, 0],
    )

    assert plane.induced_metric_components(simplify=True).equals(sp.eye(2))


def test_non_euclidean_ambient_metric_pullback_substitutes_components():
    t, x, y = sp.symbols("t x y")
    curve = ParametrizedMap(
        name="Parabola",
        parameters=[t],
        components=[t, t**2],
        ambient_coordinates=[x, y],
        ambient_metric=[[1, 0], [0, x**2]],
    )

    induced = curve.induced_metric_components(simplify=True)

    assert induced.shape == (1, 1)
    assert sp.simplify(induced[0, 0] - (1 + 4 * t**4)) == 0


def test_sphere_extrinsic_gaussian_curvature_matches_intrinsic_curvature():
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

    gaussian = sphere.gaussian_curvature_extrinsic(simplify=True)
    scalar = sphere.pullback_metric(simplify=True).scalar_curvature(simplify=True)
    mean = sphere.mean_curvature(simplify=True)

    assert sp.simplify(gaussian - 1) == 0
    assert sp.simplify(scalar / 2 - 1) == 0
    assert sp.simplify(gaussian - scalar / 2) == 0
    assert sp.simplify(mean**2 - 1) == 0


def test_cylinder_extrinsic_curvatures():
    theta, z, radius = sp.symbols("theta z R", positive=True)
    cylinder = ParametrizedMap(
        name="Cylinder",
        parameters=[theta, z],
        components=[
            radius * sp.cos(theta),
            radius * sp.sin(theta),
            z,
        ],
    )

    gaussian = cylinder.gaussian_curvature_extrinsic(simplify=True)
    scalar = cylinder.pullback_metric(simplify=True).scalar_curvature(simplify=True)
    mean = cylinder.mean_curvature(simplify=True)

    assert sp.simplify(gaussian) == 0
    assert sp.simplify(scalar) == 0
    assert sp.simplify(mean**2 - sp.Rational(1, 4) / radius**2) == 0


def test_torus_extrinsic_gaussian_curvature_matches_intrinsic_curvature():
    u, v, major_radius, minor_radius = sp.symbols("u v R r", positive=True)
    torus = ParametrizedMap(
        name="Torus",
        parameters=[u, v],
        components=[
            (major_radius + minor_radius * sp.cos(v)) * sp.cos(u),
            (major_radius + minor_radius * sp.cos(v)) * sp.sin(u),
            minor_radius * sp.sin(v),
        ],
    )
    expected = sp.cos(v) / (minor_radius * (major_radius + minor_radius * sp.cos(v)))
    gaussian = torus.gaussian_curvature_extrinsic(simplify=True)
    scalar = torus.pullback_metric(simplify=True).scalar_curvature(simplify=True)

    assert sp.simplify(gaussian - expected) == 0
    assert sp.simplify(scalar / 2 - expected) == 0
    assert sp.simplify(gaussian - scalar / 2) == 0


def test_plane_extrinsic_geometry_is_flat():
    u, v = sp.symbols("u v")
    plane = ParametrizedMap(
        name="Plane",
        parameters=[u, v],
        components=[u, v, 0],
    )

    normal = plane.normal_vector(simplify=True)
    second_form = plane.second_fundamental_form(simplify=True)

    assert normal == sp.Matrix([0, 0, 1])
    assert second_form == sp.zeros(2, 2)
    assert sp.simplify(plane.gaussian_curvature_extrinsic(simplify=True)) == 0
    assert sp.simplify(plane.mean_curvature(simplify=True)) == 0


def test_extrinsic_methods_reject_invalid_dimensions():
    t = sp.symbols("t")
    curve = ParametrizedMap(
        name="Curve",
        parameters=[t],
        components=[t, t**2],
    )

    with pytest.raises(ValueError, match="2-dimensional parameter domain"):
        curve.normal_vector()
    with pytest.raises(ValueError, match="2-dimensional parameter domain"):
        curve.second_fundamental_form()

    u, v = sp.symbols("u v")
    surface_in_r4 = ParametrizedMap(
        name="SurfaceInR4",
        parameters=[u, v],
        components=[u, v, u**2, v**2],
    )

    with pytest.raises(ValueError, match="Euclidean R3"):
        surface_in_r4.normal_vector()
