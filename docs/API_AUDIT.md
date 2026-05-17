# CRR API Audit

CRR v1.0.2 includes a lightweight public API audit in `crr.audit`.

Run it with:

```bash
python -m crr.audit api
python examples/audit_api.py
```

## Public API Areas

### Top-level `crr`

- `Manifold`
- `Metric`
- `Tensor`
- `ParametrizedMap`
- `DifferentialForm`
- `GaugePotential`
- `GaugeCurvature`
- geodesic solving helpers
- common presets and Lie algebra exports

### `crr.forms`

- `DifferentialForm`
- `flat`
- `sharp`

### `crr.lie`

- `LieAlgebra`
- `LieAlgebraElement`
- `AlgebraValuedForm`
- algebra presets: `abelian_lie_algebra`, `so2_algebra`, `so3_algebra`, `su2_algebra`, `sl2r_algebra`, `heisenberg_algebra`

### `crr.gauge`

- `GaugePotential`
- `GaugeCurvature`
- `abelian_gauge_potential`
- `u1_connection_from_potential`
- `su2_connection`
- `pure_gauge_abelian`
- `covariant_exterior_derivative`
- `abelian_gauge_transform`
- `infinitesimal_gauge_transform`
- `yang_mills_action_density`
- `identity_inner_product`

### `crr.presets`

- surface presets such as `sphere_surface`, `torus_surface`, and `cylinder_surface`
- metric presets such as `sphere_metric`, `torus_metric`, and `hyperbolic_half_plane_metric`
- relativity presets such as `schwarzschild_metric`, `kerr_metric`, and `minkowski_metric`

### `crr.visualization`

- `plot_scalar_field_2d`
- `plot_vector_field_2d`
- `plot_form_2d`
- surface and geodesic plotting helpers

## Consistency Notes

- Most mathematical objects provide `__repr__`.
- Symbolic objects generally provide `simplify`, `equals`, or both.
- Public mathematical objects prefer `to_latex`, `to_markdown`, and `display` where appropriate.
- Existing component-specific names such as `to_markdown_table`, `to_latex_components`, and `display_nonzero` remain supported as backward-compatible aliases.
- Visualization imports are public, but matplotlib remains optional until plotting is called.

## Risks And Refactor Candidates

- Top-level exports are broad; future releases should keep backward compatibility while considering clearer namespace documentation.
- Some display APIs use `display_*` methods while others use `to_latex`; a future consistency pass could document or normalize these patterns.
- Heavy symbolic relativity and surface expressions need benchmark-aware examples.
- Gauge APIs are local and intentionally avoid global bundle abstractions for now.
