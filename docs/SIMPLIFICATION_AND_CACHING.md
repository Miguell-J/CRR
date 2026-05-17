# Simplification And Caching

CRR uses SymPy for exact symbolic computation. This makes formulas inspectable, but simplification can dominate runtime for larger metrics, embedded surfaces, and gauge-form expressions.

## When To Use `simplify=True`

Use `simplify=True` when the dimension is small, the geometry is standard, or you need a final closed-form expression for display or comparison. The unit sphere and Euclidean examples are good candidates.

Avoid aggressive simplification when exploring four-dimensional relativity metrics, computing full curvature tensors for complicated metrics, plotting dense numeric grids, or repeatedly building wedge/bracket products in gauge examples.

## Caching Behavior

`Metric` caches expensive derived quantities requested through object methods:

- inverse metric
- determinant
- Christoffel symbols
- Riemann tensor
- Ricci tensor
- scalar curvature
- Einstein tensor

The cache distinguishes simplified and unsimplified variants where supported. Reuse the same `Metric` object when exploring related quantities.

## Examples

Sphere curvature is small enough for direct simplification:

```python
from crr.presets import sphere_metric

g = sphere_metric(radius=1)
R = g.scalar_curvature(simplify=True)
```

For Kerr and other heavy relativity metrics, start with representative components:

```python
from crr.presets import kerr_metric

g = kerr_metric()
ricci = g.ricci_tensor(simplify=False)
component = ricci.components[0, 0]
```

Substitute numeric parameters before torus plotting:

```python
from crr.presets import torus_surface

torus = torus_surface(major_radius=2, minor_radius=0.7)
torus.plot_curvature((0, 6.28), (0, 6.28), curvature="gaussian", show=False)
```

For gauge forms, compute curvature first and simplify the result:

```python
F = A.curvature()
F_simplified = F.simplify()
```

## Notebook Workflow

- Build a metric or surface once per cell and reuse it.
- Inspect small summaries before displaying full tensors.
- Prefer `display_nonzero()` or Markdown component tables for sparse results.
- Use audit benchmarks after changing symbolic code paths.

Compute first, simplify late, and simplify only the object or component you need.
