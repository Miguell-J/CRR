# %% [markdown]
# # Differential Forms

# %%
import sympy as sp

from crr import DifferentialForm, Manifold
from crr.presets import euclidean_metric

x, y, z = sp.symbols("x y z")
M = Manifold("R3", 3, [x, y, z])
g = euclidean_metric(dim=3, coordinates=[x, y, z])

dx = DifferentialForm.basis(M, 0)
dy = DifferentialForm.basis(M, 1)
dz = DifferentialForm.basis(M, 2)

omega = dx.wedge(dy)
print("omega")
print(omega)
print("*omega")
print(omega.hodge_star(g, simplify=True))
print("volume")
print(dx.wedge(dy).wedge(dz))
