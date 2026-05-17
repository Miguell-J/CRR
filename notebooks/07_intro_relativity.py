# %% [markdown]
# # Intro Relativity Presets

# %%
import sympy as sp

from crr.presets import de_sitter_metric, minkowski_metric, schwarzschild_metric

eta = minkowski_metric()
print("Minkowski scalar curvature")
print(eta.scalar_curvature(simplify=True))


# %%
schwarzschild = schwarzschild_metric()
ricci = schwarzschild.ricci_tensor(simplify=True)
print("Schwarzschild representative Ricci components")
for index in [(0, 0), (1, 1), (2, 2), (3, 3)]:
    print(index, sp.simplify(ricci[index]))


# %%
L = sp.symbols("L", positive=True)
dS = de_sitter_metric(radius=L)
print("de Sitter scalar curvature")
print(dS.scalar_curvature(simplify=True))
