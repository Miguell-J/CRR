import sympy as sp
import pytest

from crr import DifferentialForm, Manifold
from crr.gauge import (
    GaugeCurvature,
    GaugePotential,
    abelian_gauge_potential,
    abelian_gauge_transform,
    covariant_exterior_derivative,
    pure_gauge_abelian,
    su2_connection,
    u1_connection_from_potential,
    yang_mills_action_density,
)
from crr.lie import AlgebraValuedForm, abelian_lie_algebra, su2_algebra
from crr.presets import euclidean_metric


def test_gauge_potential_validation():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    dx = DifferentialForm.basis(manifold, 0)
    scalar = DifferentialForm.scalar(manifold, x)

    with pytest.raises(ValueError):
        GaugePotential(abelian_lie_algebra(1), [scalar])
    with pytest.raises(ValueError):
        GaugePotential(su2_algebra(), [dx])
    with pytest.raises(TypeError):
        GaugePotential(abelian_lie_algebra(1), [x])

    assert isinstance(abelian_gauge_potential(manifold, dx), GaugePotential)
    assert isinstance(su2_connection(manifold, [dx, dx, DifferentialForm.zero(manifold, 1)]), GaugePotential)


def test_abelian_curvature_equals_exterior_derivative_and_bracket_vanishes():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    potential = x * dy
    connection = abelian_gauge_potential(manifold, potential)

    curvature = connection.curvature()

    assert isinstance(curvature, GaugeCurvature)
    assert curvature.components[0].equals(potential.exterior_derivative())
    assert connection.bracket_wedge(connection).equals(AlgebraValuedForm.zero(connection.algebra, manifold, 2))
    assert curvature.components[0].equals(dx.wedge(dy))


def test_abelian_bianchi_and_pure_gauge_curvature_zero():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    dy = DifferentialForm.basis(manifold, 1)
    connection = abelian_gauge_potential(manifold, x * dy)

    assert connection.bianchi_identity().equals(AlgebraValuedForm.zero(connection.algebra, manifold, 2))

    pure = pure_gauge_abelian(manifold, x * y)
    assert pure.curvature().components[0].equals(DifferentialForm.zero(manifold, 2))


def test_abelian_gauge_transform_preserves_curvature():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    dy = DifferentialForm.basis(manifold, 1)
    connection = abelian_gauge_potential(manifold, x * dy)

    transformed = abelian_gauge_transform(connection, x * y)

    assert transformed.components[0].equals(connection.components[0] + DifferentialForm.scalar(manifold, x * y).exterior_derivative())
    assert transformed.curvature().equals(connection.curvature())


def test_su2_curvature_has_bracket_contribution_when_dA_zero():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    zero = DifferentialForm.zero(manifold, 1)
    connection = su2_connection(manifold, [dx, dy, zero])

    curvature = connection.curvature()

    assert curvature.degree == 2
    assert curvature.components[0].equals(DifferentialForm.zero(manifold, 2))
    assert curvature.components[1].equals(DifferentialForm.zero(manifold, 2))
    assert curvature.components[2].equals(dx.wedge(dy))


def test_covariant_exterior_derivative_degree_and_abelian_reduction():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    connection = abelian_gauge_potential(manifold, x * dy)
    omega = AlgebraValuedForm(connection.algebra, [DifferentialForm.scalar(manifold, x * y)])

    result = covariant_exterior_derivative(connection, omega)

    assert result.degree == 1
    assert result.equals(omega.exterior_derivative())

    su2 = su2_connection(manifold, [dx, dy, DifferentialForm.zero(manifold, 1)])
    su2_omega = AlgebraValuedForm(su2.algebra, [DifferentialForm.scalar(manifold, x), DifferentialForm.scalar(manifold, 0), DifferentialForm.scalar(manifold, 0)])
    assert su2.covariant_exterior_derivative(su2_omega).degree == 1


def test_bianchi_identity_for_abelian_and_simple_su2():
    x, y, z = sp.symbols("x y z")
    manifold = Manifold("R3", 3, [x, y, z])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    zero = DifferentialForm.zero(manifold, 1)

    abelian = abelian_gauge_potential(manifold, x * dy)
    assert abelian.bianchi_identity().equals(AlgebraValuedForm.zero(abelian.algebra, manifold, 3))

    su2 = su2_connection(manifold, [dx, dy, zero])
    assert su2.bianchi_identity().equals(AlgebraValuedForm.zero(su2.algebra, manifold, 3))


def test_yang_mills_action_density_u1_euclidean_r2():
    x, y = sp.symbols("x y")
    metric = euclidean_metric(dim=2, coordinates=[x, y])
    manifold = metric.manifold
    dy = DifferentialForm.basis(manifold, 1)
    connection = abelian_gauge_potential(manifold, x * dy)
    curvature = connection.curvature()

    density = yang_mills_action_density(curvature, metric, simplify=True)

    assert density.degree == 2
    assert density.equals(DifferentialForm(manifold, 2, {(0, 1): 1}))


def test_gauge_exports_and_u1_preset():
    from crr.gauge import GaugeCurvature as ImportedCurvature
    from crr.gauge import GaugePotential as ImportedPotential
    from crr.gauge import abelian_gauge_potential as imported_abelian
    from crr.gauge import yang_mills_action_density as imported_density

    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    connection = u1_connection_from_potential(manifold, [0, x])

    assert ImportedPotential is GaugePotential
    assert ImportedCurvature is GaugeCurvature
    assert imported_abelian is abelian_gauge_potential
    assert imported_density is yang_mills_action_density
    assert connection.components[0].equals(x * DifferentialForm.basis(manifold, 1))
