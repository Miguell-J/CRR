# %% [markdown]
# # Maxwell Forms Toy Example

# %%
import _bootstrap  # noqa: F401

import sympy as sp

from crr import DifferentialForm, Manifold

t, x, y, z = sp.symbols("t x y z")
M = Manifold("R4", 4, [t, x, y, z])

Phi = sp.Function("Phi")(t, x, y, z)
Ax = sp.Function("A_x")(t, x, y, z)
Ay = sp.Function("A_y")(t, x, y, z)
Az = sp.Function("A_z")(t, x, y, z)

A = DifferentialForm.one_form(M, [-Phi, Ax, Ay, Az])
F = A.exterior_derivative()

print("F = dA")
print(F)
print("dF")
print(F.exterior_derivative().simplify())
