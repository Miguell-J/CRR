# %% [markdown]
# # Numeric Geodesics

# %%
import numpy as np

from crr.presets import sphere_metric

g = sphere_metric(radius=1)
solution = g.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 4), num_points=80)

print(solution)
print("Final state")
print(solution.final_state())
print("Energy drift")
energy = solution.energy()
print(float(np.max(np.abs(energy - energy[0]))))
