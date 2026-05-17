# API Stability

CRR v1.0.2 treats the current public API as stable unless explicitly marked experimental.

## Public API Categories

- Core geometry: `Manifold`, `Metric`, `Tensor`, `ParametrizedMap`
- Geometry operations: Christoffel, Riemann, Ricci, scalar curvature, Einstein tensor, covariant derivatives
- Geodesics: symbolic equations, numerical solving, `GeodesicSolution`
- Embedded surfaces and integration helpers
- Preset metrics and surfaces
- Differential forms and musical maps
- Lie algebras and algebra-valued forms
- Local gauge fields
- Display and visualization helpers
- Audit reports, API checks, and benchmarks

## Stable APIs

Stable APIs should remain backward compatible before v2.0:

- object constructors for core geometry, forms, Lie algebras, and gauge fields;
- documented methods such as `simplify`, `equals`, `to_latex`, `to_markdown`, and `display`;
- preset function names;
- example-backed workflows in `examples/`;
- audit APIs added in v1.0.1 and benchmark baseline APIs added in v1.0.2.

## Experimental APIs

- pseudo-Riemannian Hodge star sign conventions;
- local Yang-Mills action-density sign and normalization conventions;
- visualization details and matplotlib styling;
- performance benchmark thresholds and baseline schema;
- future global geometry and bundle abstractions.

## Backward Compatibility Policy Before v2.0

- Do not remove public methods without a deprecation period.
- Prefer aliases when normalizing names.
- Keep examples runnable as integration checks.
- Preserve existing mathematical conventions unless a change is explicitly versioned and documented.

## Naming Conventions

- `to_latex()` returns a string and should not require IPython.
- `to_markdown()` returns a Markdown string.
- `display()` may render in IPython if available, but must return a string outside IPython.
- `nonzero_components()` returns sparse component data where that concept is meaningful.
- `to_markdown_table()` and `to_latex_components()` remain component-table aliases for tensor-like objects.
- `display_nonzero()` displays or returns nonzero component summaries.
- `plot_*` functions may require optional visualization dependencies.

## Risks

- Top-level exports are broad and convenient, but namespace documentation must stay clear.
- Some heavy symbolic operations are stable but expensive.
- Gauge APIs intentionally model local trivializations, not global principal bundles.
