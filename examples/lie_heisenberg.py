"""Three-dimensional Heisenberg algebra."""

import _bootstrap  # noqa: F401

from crr.lie import heisenberg_algebra


h = heisenberg_algebra()
X, Y, Z = [h.basis_element(i) for i in range(3)]

print(h)
print("[X, Y] =", X.bracket(Y))
print("[X, Z] =", X.bracket(Z))
print("[Y, Z] =", Y.bracket(Z))
print("Jacobi:", h.check_jacobi())
print("Killing form:")
print(h.killing_form_matrix(simplify=True))
