import sympy as sp

from crr import DifferentialForm, Manifold
from crr.presets import euclidean_metric, sphere_metric


def test_form_construction_canonicalizes_and_combines_components():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])

    form = DifferentialForm(manifold, 2, {(1, 0): 2, (0, 1): 3, (0, 0): 10})

    assert form.components == {(0, 1): 1}
    assert form[(1, 0)] == -1
    assert form[(0, 0)] == 0


def test_wedge_antisymmetry_associativity_and_graded_commutativity():
    x, y, z = sp.symbols("x y z")
    manifold = Manifold("R3", 3, [x, y, z])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    dz = DifferentialForm.basis(manifold, 2)

    assert dx.wedge(dy).equals(-dy.wedge(dx))
    assert dx.wedge(dx).equals(DifferentialForm.zero(manifold, 2))
    assert dx.wedge(dy).wedge(dz).equals(dx.wedge(dy.wedge(dz)))
    assert dx.wedge(dy.wedge(dz)).equals(dy.wedge(dz).wedge(dx))


def test_exterior_derivative_properties():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    f = DifferentialForm.scalar(manifold, x * y)
    df = f.exterior_derivative()

    assert df.equals(DifferentialForm.one_form(manifold, [y, x]))
    assert df.exterior_derivative().equals(DifferentialForm.zero(manifold, 2))

    alpha = x * DifferentialForm.basis(manifold, 1)
    assert alpha.exterior_derivative().equals(DifferentialForm(manifold, 2, {(0, 1): 1}))
    assert DifferentialForm(manifold, 2, {(0, 1): x}).exterior_derivative().equals(DifferentialForm.zero(manifold, 2))


def test_hodge_star_euclidean_r2():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    metric = euclidean_metric(dim=2, coordinates=[x, y])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    volume = dx.wedge(dy)

    assert dx.hodge_star(metric, simplify=True).equals(dy)
    assert dy.hodge_star(metric, simplify=True).equals(-dx)
    assert volume.hodge_star(metric, simplify=True).equals(DifferentialForm.scalar(manifold, 1))
    assert dx.hodge_star(metric, simplify=True).hodge_star(metric, simplify=True).equals(-dx)


def test_hodge_star_euclidean_r3():
    x, y, z = sp.symbols("x y z")
    manifold = Manifold("R3", 3, [x, y, z])
    metric = euclidean_metric(dim=3, coordinates=[x, y, z])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    dz = DifferentialForm.basis(manifold, 2)

    assert dx.hodge_star(metric, simplify=True).equals(dy.wedge(dz))
    assert dy.hodge_star(metric, simplify=True).equals(dz.wedge(dx))
    assert dz.hodge_star(metric, simplify=True).equals(dx.wedge(dy))
    assert dx.wedge(dy).hodge_star(metric, simplify=True).equals(dz)


def test_hodge_star_inner_product_identity_for_basis_form():
    x, y, z = sp.symbols("x y z")
    manifold = Manifold("R3", 3, [x, y, z])
    metric = euclidean_metric(dim=3, coordinates=[x, y, z])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    dz = DifferentialForm.basis(manifold, 2)
    volume = dx.wedge(dy).wedge(dz)

    assert dx.wedge(dx.hodge_star(metric, simplify=True)).equals(volume)


def test_hodge_laplacian_scalar_sign_convention():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    metric = euclidean_metric(dim=2, coordinates=[x, y])
    f = DifferentialForm.scalar(manifold, x**2 + y**2)

    assert f.hodge_laplacian(metric, simplify=True).equals(DifferentialForm.scalar(manifold, -4))


def test_musical_isomorphisms_euclidean_metric():
    x, y, P, Q = sp.symbols("x y P Q")
    metric = euclidean_metric(dim=2, coordinates=[x, y])

    one_form = metric.flat([P, Q], simplify=True)

    assert one_form.equals(DifferentialForm.one_form(metric.manifold, [P, Q]))
    assert metric.sharp(one_form, simplify=True) == [P, Q]


def test_top_form_integration():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    form = DifferentialForm(manifold, 2, {(0, 1): x + y})

    assert sp.simplify(form.integrate([(x, 0, 1), (y, 0, 1)], simplify=True) - 1) == 0


def test_sphere_volume_form_from_hodge_star():
    theta, phi = sp.symbols("theta phi")
    metric = sphere_metric(radius=1, coordinates=[theta, phi])
    one = DifferentialForm.scalar(metric.manifold, 1)
    volume = one.hodge_star(metric, simplify=True)

    assert volume.equals(DifferentialForm(metric.manifold, 2, {(0, 1): sp.sin(theta)}))
    assert sp.simplify(volume.integrate([(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], simplify=True) - 4 * sp.pi) == 0
