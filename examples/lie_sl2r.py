"""Basic sl(2,R) Lie algebra operations."""

import _bootstrap  # noqa: F401

from crr.lie import sl2r_algebra


sl2 = sl2r_algebra()
H, E, F = [sl2.basis_element(i) for i in range(3)]

print(sl2)
print("[H, E] =", H.bracket(E))
print("[H, F] =", H.bracket(F))
print("[E, F] =", E.bracket(F))
print("ad_H:")
print(sl2.adjoint_matrix(0))
print("Killing form:")
print(sl2.killing_form_matrix(simplify=True))
