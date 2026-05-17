"""Basic so(3) Lie algebra operations."""

import _bootstrap  # noqa: F401

from crr.lie import so3_algebra


so3 = so3_algebra()
e1, e2, e3 = [so3.basis_element(i) for i in range(3)]

print(so3)
print("[e1, e2] =", e1.bracket(e2))
print("[e2, e1] =", e2.bracket(e1))
print("Jacobi:", so3.check_jacobi())
print("Antisymmetry:", so3.check_antisymmetry())
print("Killing form:")
print(so3.killing_form_matrix(simplify=True))
