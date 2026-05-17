# CRR Capabilities

CRR v1.0.2 is a local-coordinate symbolic/numeric lab for differential geometry and mathematical physics. It combines SymPy-based exact expressions with numerical geodesic solving, coordinate-domain integration, display helpers, audit tools, benchmark baselines, and optional matplotlib visualization.

## What CRR Can Do Now

### Core geometry

- Build `Manifold`, `Metric`, `Tensor`, and `ParametrizedMap` objects.
- Compute Christoffel symbols, Riemann tensor, Ricci tensor, scalar curvature, and Einstein tensor.
- Compute covariant derivatives of scalars, vectors, covectors, and covariant 2-tensors.

### Geodesics

- Derive symbolic geodesic equations.
- Numerically solve geodesic initial-value problems.
- Inspect `GeodesicSolution` coordinates, velocities, energy, and final state.
- Embed surface geodesics back into ambient space.

### Embedded surfaces and integration

- Work with parametrized surfaces such as spheres, cylinders, tori, catenoids, helicoids, and local Mobius strip charts.
- Compute induced metrics, area density, normal vectors, second fundamental forms, Gaussian curvature, and mean curvature.
- Integrate scalar and curvature densities over explicit coordinate domains.
- Run Gauss-Bonnet checks for chart-domain examples.

### Relativity

- Use local metric presets including Minkowski, Schwarzschild, FLRW/Robertson-Walker, Reissner-Nordstrom, Kerr, Eddington-Finkelstein, Kruskal-Szekeres, Godel, de Sitter, anti-de Sitter, and Alcubierre-style metrics.
- Compute curvature objects from these metrics when symbolic complexity is manageable.

### Differential forms

- Construct sparse coordinate-basis differential forms.
- Compute wedge products, exterior derivatives, Hodge star, codifferential, Hodge Laplacian, top-form integration, and musical isomorphisms.

### Lie algebras and gauge fields

- Represent finite-dimensional Lie algebras by symbolic structure constants.
- Check antisymmetry and Jacobi identities, compute brackets and Killing forms.
- Work with algebra-valued differential forms and bracket-wedge products.
- Represent gauge potentials and curvature 2-forms.
- Compute abelian transformations, infinitesimal non-abelian transformations, Bianchi expressions, and local Yang-Mills action densities.
- Use `identity_inner_product(algebra)` for the default positive identity convention used by compact gauge examples.

### Visualization and display

- Plot surfaces, curvature maps, geodesics, scalar fields, density fields, vector fields, and 2D forms with optional matplotlib.
- Use LaTeX and Markdown display helpers in notebooks with plain-string fallbacks.

## Examples By Category

- Core geometry: `examples/sphere.py`, `examples/schwarzschild.py`
- Geodesics: `examples/geodesics_sphere.py`, `examples/numeric_geodesic_sphere.py`
- Embedded surfaces: `examples/extrinsic_sphere.py`, `examples/extrinsic_torus.py`
- Integration: `examples/integrate_sphere_area.py`, `examples/gauss_bonnet_sphere.py`
- Relativity: `examples/extended_relativity_presets.py`, `examples/kerr_and_rn_limits.py`
- Forms: `examples/forms_basic.py`, `examples/hodge_star_euclidean.py`
- Lie algebras: `examples/lie_so3.py`, `examples/algebra_valued_forms.py`
- Gauge fields: `examples/gauge_u1_maxwell.py`, `examples/gauge_su2_curvature.py`
- Audit: `examples/audit_capabilities.py`, `examples/audit_api.py`, `examples/audit_benchmarks.py`

## Stable Features

The stable v1.0.2 surface includes the core metric/tensor API, differential forms, Lie algebra structures, local gauge curvature, preset metrics/surfaces, coordinate-domain integration, numerical geodesics, lightweight display helpers, and audit/benchmark reporting helpers.

## Experimental Features

- Pseudo-Riemannian Hodge star and Yang-Mills sign conventions.
- Local Yang-Mills action-density signs; the default Lie-algebra inner product is the identity matrix, while Killing forms are sign- and normalization-dependent.
- Visualization for heavy symbolic expressions.
- Advanced relativity metrics where full curvature simplification may be expensive.

## Known Limitations

- CRR is local-coordinate based; it does not implement a global atlas system.
- There is no principal bundle topology, transition-function machinery, or finite non-abelian gauge transformation support.
- There is no automatic Chern-Weil class computation.
- Symbolic simplification can dominate runtime for large expressions.
- Numerical geodesics are standard ODE solves, not symplectic or variational integrators.
- Visualization is matplotlib-only and chart-based.

## Suggested Scientific Workflows

- Use presets to prototype known geometries, then move to custom metrics or parametrizations.
- Compute representative curvature components first for expensive metrics.
- Use `simplify=True` selectively after verifying the raw computation.
- Substitute numeric parameters before plotting or benchmarking.
- Use forms and gauge fields in small local charts before attempting larger symbolic systems.

## Suggested Jupyter Notebooks

- `notebooks/01_sphere_geometry.py`
- `notebooks/04_numeric_geodesics.py`
- `notebooks/05_differential_forms.py`
- `notebooks/06_maxwell_forms.py`
- `notebooks/08_visualizing_surfaces.py`
- `notebooks/11_visualizing_forms.py`

## Roadmap To v1.1+

- More robust symbolic simplification strategies and caching.
- Finite non-abelian gauge transformations.
- Chern-Weil and Chern-Simons helpers.
- Better pseudo-Riemannian form and gauge conventions.
- More explicit global chart/atlas abstractions.
- Broader benchmark coverage and performance regression tracking.
