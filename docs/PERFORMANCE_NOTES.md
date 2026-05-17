# CRR Performance Notes

CRR v1.0.2 includes lightweight benchmarks and baseline comparison helpers in `crr.audit.benchmarks`.

Run them with:

```bash
python -m crr.audit benchmarks
python examples/audit_benchmarks.py
```

Save and compare local baselines with:

```bash
python -m crr.audit benchmarks --quick --save-baseline /tmp/crr_benchmark_baseline.json
python -m crr.audit benchmarks --quick --compare /tmp/crr_benchmark_baseline.json
```

Benchmarks classify elapsed runtime as:

- fast: less than 0.5 seconds
- moderate: less than 5 seconds
- slow: 5 seconds or more

## Benchmark Cases

- `sphere_curvature`: builds the unit sphere metric and computes scalar curvature.
- `torus_curvature`: builds a numeric-radius torus and computes intrinsic scalar curvature and extrinsic Gaussian curvature.
- `schwarzschild_ricci_representative`: builds the Schwarzschild metric and computes a representative Ricci tensor access.
- `forms_hodge_star_r3`: computes Hodge stars of Euclidean R3 basis 1-forms.
- `lie_su2_jacobi`: checks the Jacobi identity for the `su2` preset.
- `gauge_su2_curvature`: computes curvature for a simple `su2` connection on R2.
- `geodesic_sphere_numeric`: solves a short equatorial geodesic on the sphere.

## Expensive Computations

- Full curvature tensors for four-dimensional relativity metrics can grow quickly.
- Aggressive `simplify=True` may cost more than the raw tensor computation.
- Parametrized surface curvature can expand substantially when parameters remain symbolic.
- Plotting symbolic fields is faster after substituting numeric parameters.

## Recommendations

- Use `simplify=True` selectively and late.
- Substitute numeric parameter values before plotting, benchmarking, or dense numerical sampling.
- Avoid full curvature for heavy GR metrics unless all components are needed.
- Inspect representative tensor components while exploring.
- Prefer lower-resolution plots first, then increase resolution after expressions are stable.
- Cache intermediate metrics, Christoffel symbols, and curvature tensors by reusing object methods rather than recomputing helper functions directly.
- Treat benchmark regressions as prompts for investigation, not hard failures, unless a project-specific CI policy opts into strict enforcement.
