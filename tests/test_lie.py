import sympy as sp

from crr import DifferentialForm, Manifold
from crr.lie import (
    AlgebraValuedForm,
    abelian_lie_algebra,
    heisenberg_algebra,
    matrix_commutator,
    matrix_lie_algebra_from_basis,
    sl2r_algebra,
    so3_algebra,
    su2_algebra,
)


def test_so3_brackets_and_identities():
    algebra = so3_algebra()
    e1, e2, e3 = [algebra.basis_element(i) for i in range(3)]

    assert e1.bracket(e2).equals(e3)
    assert e2.bracket(e1).equals(-e3)
    assert algebra.check_antisymmetry()
    assert algebra.check_jacobi()

    killing = algebra.killing_form_matrix(simplify=True)
    assert killing == sp.diag(-2, -2, -2)
    assert killing == killing.T


def test_su2_convention_matches_compact_real_so3_structure():
    algebra = su2_algebra()
    t1, t2, t3 = [algebra.basis_element(i) for i in range(3)]

    assert algebra.dimension == 3
    assert t1.bracket(t2).equals(t3)
    assert algebra.check_jacobi()


def test_sl2r_brackets_and_jacobi():
    algebra = sl2r_algebra()
    H, E, F = [algebra.basis_element(i) for i in range(3)]

    assert H.bracket(E).equals(2 * E)
    assert H.bracket(F).equals(-2 * F)
    assert E.bracket(F).equals(H)
    assert algebra.check_jacobi()


def test_heisenberg_brackets_and_killing_degeneracy():
    algebra = heisenberg_algebra()
    X, Y, Z = [algebra.basis_element(i) for i in range(3)]

    assert X.bracket(Y).equals(Z)
    assert X.bracket(Z).equals(algebra.element([0, 0, 0]))
    assert Y.bracket(Z).equals(algebra.element([0, 0, 0]))
    assert algebra.check_jacobi()
    assert algebra.killing_form_matrix(simplify=True) == sp.zeros(3)


def test_abelian_algebra_is_zero_bracket():
    algebra = abelian_lie_algebra(3)
    x = algebra.element([1, 2, 3])
    y = algebra.element([4, 5, 6])

    assert algebra.is_abelian()
    assert algebra.check_jacobi()
    assert x.bracket(y).equals(algebra.element([0, 0, 0]))


def test_lie_algebra_element_operations_and_bilinearity():
    algebra = so3_algebra()
    e1, e2, e3 = [algebra.basis_element(i) for i in range(3)]
    a, b = sp.symbols("a b")

    assert (e1 + 2 * e2).equals(algebra.element([1, 2, 0]))
    assert (a * e1).equals(algebra.element([a, 0, 0]))
    assert (a * e1).bracket(b * e2).equals(a * b * e3)


def test_matrix_commutator_and_matrix_lie_algebra_from_basis():
    H = sp.Matrix([[1, 0], [0, -1]])
    E = sp.Matrix([[0, 1], [0, 0]])
    F = sp.Matrix([[0, 0], [1, 0]])

    assert matrix_commutator(H, E) == 2 * E

    algebra = matrix_lie_algebra_from_basis("sl2_from_matrices", [H, E, F], ["H", "E", "F"])
    H_el, E_el, F_el = [algebra.basis_element(i) for i in range(3)]
    assert H_el.bracket(E_el).equals(2 * E_el)
    assert H_el.bracket(F_el).equals(-2 * F_el)
    assert E_el.bracket(F_el).equals(H_el)


def test_algebra_valued_forms_for_nonabelian_and_abelian_cases():
    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)

    algebra = su2_algebra()
    A = AlgebraValuedForm(algebra, [x * dx, y * dy, DifferentialForm.zero(manifold, 1)])
    dA = A.exterior_derivative()
    bracket = A.bracket_wedge(A).simplify()

    assert dA.degree == 2
    assert bracket.degree == 2
    assert all(isinstance(component, DifferentialForm) for component in bracket.components)
    assert bracket.components[2].equals(2 * x * y * dx.wedge(dy))

    abelian = abelian_lie_algebra(3)
    A0 = AlgebraValuedForm(abelian, [x * dx, y * dy, DifferentialForm.zero(manifold, 1)])
    assert A0.bracket_wedge(A0).equals(AlgebraValuedForm.zero(abelian, manifold, 2))
