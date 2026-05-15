# CRR: Curvature, Ricci, and Riemann

CRR is a small symbolic and numerical differential geometry package for computing geometric objects from a coordinate chart, a metric tensor, or a parametrized embedding. The v0.5 scope focuses on readable SymPy-based computations plus basic numerical geodesic solving for mathematical physics, general relativity, Riemannian geometry, embedded surfaces, and geometric analysis.

## Installation

From the project root:

```bash
python -m pip install -e ".[dev]"
```

For runtime use only:

```bash
python -m pip install -e .
```

For optional plotting helpers:

```bash
python -m pip install -e ".[viz]"
```

## Quickstart

```python
import sympy as sp
from crr import Manifold, Metric

theta, phi = sp.symbols("theta phi")

M = Manifold(
    name="S2",
    dimension=2,
    coordinates=[theta, phi],
)

g = Metric(
    manifold=M,
    components=[
        [1, 0],
        [0, sp.sin(theta)**2],
    ],
)

print(g.scalar_curvature(simplify=True))  # 2
```

## Mathematical Conventions

Coordinates are written as `x^i`. Metric components are `g_ij`, and the inverse metric is `g^ij`.

Christoffel symbols:

```text
Gamma^k_{ij} = 1/2 g^{kl} (d_i g_{jl} + d_j g_{il} - d_l g_{ij})
```

Riemann tensor:

```text
R^i_{jkl} = d_k Gamma^i_{lj} - d_l Gamma^i_{kj}
            + Gamma^i_{km} Gamma^m_{lj}
            - Gamma^i_{lm} Gamma^m_{kj}
```

Ricci tensor:

```text
R_jl = R^i_{jil}
```

Scalar curvature:

```text
R = g^jl R_jl
```

Einstein tensor:

```text
G_ij = R_ij - 1/2 g_ij R
```

Geodesic acceleration:

```text
a^k = - Gamma^k_{ij} v^i v^j
```

Equivalently, geodesic equations are returned as left-hand sides:

```text
d2x^k/dlambda^2 + Gamma^k_{ij} dx^i/dlambda dx^j/dlambda = 0
```

These conventions fix the sign of curvature. With the CRR convention, the unit 2-sphere has scalar curvature `2`, and the Poincare half-plane with metric `(dx^2 + dy^2) / y^2` has scalar curvature `-2`.

## API Notes

`Metric` caches expensive objects: inverse metric, determinant, Christoffel symbols, Riemann tensor, Ricci tensor, scalar curvature, and Einstein tensor. Expensive simplification is optional:

```python
Ricci = g.ricci_tensor(simplify=True)
R = g.scalar_curvature(simplify=True)
```

Tensor objects support:

```python
tensor.nonzero_components()
tensor.to_latex()
tensor.simplify()
tensor.equals(other)
```

## v0.2 Tensor Operations

CRR includes basic dense tensor operations for the ranks used by the library:

```python
Ric = g.ricci_tensor()
Ric_mixed = Ric.raise_index(metric=g, index=0, simplify=True)
R = Ric.trace(metric=g, simplify=True)
Ric_again = Ric_mixed.lower_index(metric=g, index=0, simplify=True)
```

`raise_index` uses `g^ij`, `lower_index` uses `g_ij`, and `trace(metric=g)` computes `g^ij T_ij` for rank-2 covariant tensors. `contract(axis1, axis2)` is available for ordinary same-axis contraction without an extra metric.

## v0.2 Covariant Derivatives

Covariant derivatives are available as methods on `Metric` and as functions in `crr.geometry`:

```python
x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])
g = Metric(M, [[1, 0], [0, 1]])

df = g.covariant_derivative_scalar(x**2 + y**2, simplify=True)
nabla_V = g.covariant_derivative_vector([x**2, x*y], simplify=True)
nabla_g = g.covariant_derivative_covariant_2tensor(g.components, simplify=True)
```

Implemented conventions:

```text
nabla_i f = partial_i f
nabla_j V^i = partial_j V^i + Gamma^i_{jk} V^k
nabla_j omega_i = partial_j omega_i - Gamma^k_{ji} omega_k
nabla_k T_ij = partial_k T_ij - Gamma^m_{ki} T_mj - Gamma^m_{kj} T_im
```

## v0.2 Geodesics

For symbolic velocities:

```python
theta, phi = sp.symbols("theta phi")
v_theta, v_phi = sp.symbols("v_theta v_phi")
M = Manifold("S2", 2, [theta, phi])
g = Metric(M, [[1, 0], [0, sp.sin(theta)**2]])

acc = g.geodesic_acceleration([v_theta, v_phi], simplify=True)
print(acc)
```

For coordinate functions of an affine parameter:

```python
lam = sp.Symbol("lambda")
eqs = g.geodesic_equations(parameter_symbol=lam, simplify=True)
```

## v0.3 Parametrized Maps And Embedded Surfaces

CRR can build an induced metric from a parametrized map:

```text
F: U -> R^n
```

For Euclidean ambient space, the induced metric is:

```text
g_ab = sum_i partial_a F^i partial_b F^i
```

With an ambient metric `G_ij`, CRR computes the pullback metric:

```text
g_ab = G_ij(F(u)) partial_a F^i partial_b F^j
```

Example: the unit sphere embedded in Euclidean `R3`.

```python
import sympy as sp
from crr import ParametrizedMap

theta, phi = sp.symbols("theta phi")

sphere = ParametrizedMap(
    name="UnitSphere",
    parameters=[theta, phi],
    components=[
        sp.sin(theta) * sp.cos(phi),
        sp.sin(theta) * sp.sin(phi),
        sp.cos(theta),
    ],
)

g = sphere.pullback_metric(simplify=True)
print(g.components)
print(g.scalar_curvature(simplify=True))  # 2
```

`ParametrizedMap` provides:

```python
sphere.jacobian()
sphere.induced_metric_components(simplify=True)
sphere.first_fundamental_form(simplify=True)
sphere.pullback_metric(simplify=True)
sphere.volume_density(simplify=True)
sphere.area_density(simplify=True)
```

Because `pullback_metric()` returns a normal `Metric`, all existing CRR geometry APIs apply:

```python
Gamma = g.christoffel_symbols(simplify=True)
R = g.scalar_curvature(simplify=True)
```

For a non-Euclidean ambient metric, pass ambient coordinates and the metric matrix:

```python
t, x, y = sp.symbols("t x y")

curve = ParametrizedMap(
    name="Curve",
    parameters=[t],
    components=[t, t**2],
    ambient_coordinates=[x, y],
    ambient_metric=[[1, 0], [0, x**2]],
)

print(curve.induced_metric_components(simplify=True))
```

## v0.4 Extrinsic Geometry Of Surfaces

For a 2-dimensional parametrized surface in Euclidean `R3`,

```text
F(u, v) = (x(u,v), y(u,v), z(u,v))
```

CRR computes classical extrinsic surface geometry:

```python
surface.tangent_vectors(simplify=True)
surface.normal_vector(simplify=True)
surface.second_fundamental_form(simplify=True)
surface.shape_operator(simplify=True)
surface.gaussian_curvature_extrinsic(simplify=True)
surface.mean_curvature(simplify=True)
surface.principal_curvatures(simplify=True)
```

The normal orientation is fixed by:

```text
N = (F_u x F_v) / ||F_u x F_v||
```

Changing the parameter order reverses the normal, so the sign of the second fundamental form, shape operator, principal curvatures, and mean curvature can change. Gaussian curvature is orientation-independent. For a 2D surface, the intrinsic scalar curvature of the induced metric satisfies:

```text
R = 2K
```

Example:

```python
import sympy as sp
from crr import ParametrizedMap

theta, phi = sp.symbols("theta phi")

sphere = ParametrizedMap(
    name="UnitSphere",
    parameters=[theta, phi],
    components=[
        sp.sin(theta) * sp.cos(phi),
        sp.sin(theta) * sp.sin(phi),
        sp.cos(theta),
    ],
)

g = sphere.pullback_metric(simplify=True)
K = sphere.gaussian_curvature_extrinsic(simplify=True)
R = g.scalar_curvature(simplify=True)

print(K)                      # 1
print(sp.simplify(K - R / 2)) # 0
```

Extrinsic geometry is currently implemented only for 2D surfaces embedded in Euclidean `R3`. For other dimensions or non-Euclidean ambient metrics, these methods raise a clear error.

## v0.5 Numerical Geodesics

CRR can solve the geodesic equation numerically using SciPy:

```text
d2x^k/dt^2 + Gamma^k_ij(x) dx^i/dt dx^j/dt = 0
```

The solver uses the first-order system:

```text
dx^k/dt = v^k
dv^k/dt = - Gamma^k_ij(x) v^i v^j
```

Example on the unit sphere:

```python
import sympy as sp
from crr import Manifold, Metric

theta, phi = sp.symbols("theta phi")
g = Metric(
    Manifold("S2", 2, [theta, phi]),
    [[1, 0], [0, sp.sin(theta)**2]],
)

sol = g.solve_geodesic(
    x0=[sp.pi / 2, 0],
    v0=[0, 1],
    t_span=(0, 10),
    num_points=500,
)

print(sol.success)
print(sol.final_state())
print(sol.energy())
```

The same solver is available as a standalone function:

```python
from crr import solve_geodesic

sol = solve_geodesic(g, x0=[sp.pi / 2, 0], v0=[0, 1], t_span=(0, 10))
```

`GeodesicSolution` stores:

```python
sol.t
sol.x
sol.v
sol.coordinates()
sol.velocities()
sol.final_state()
sol.speed_squared()
sol.energy()
sol.to_numpy()
```

For embedded surfaces, map coordinate geodesics into ambient space:

```python
from crr import ParametrizedMap

sphere = ParametrizedMap(
    name="UnitSphere",
    parameters=[theta, phi],
    components=[
        sp.sin(theta) * sp.cos(phi),
        sp.sin(theta) * sp.sin(phi),
        sp.cos(theta),
    ],
)

sol = sphere.pullback_metric(simplify=True).solve_geodesic(
    x0=[sp.pi / 2, 0],
    v0=[0, 1],
    t_span=(0, 10),
)
points_in_r3 = sphere.embed_geodesic(sol)
```

Plotting is optional and requires matplotlib:

```python
sol.plot_coordinates()
```

## Examples

```bash
python examples/euclidean_plane.py
python examples/sphere.py
python examples/poincare_half_plane.py
python examples/schwarzschild.py
python examples/covariant_derivative_sphere.py
python examples/geodesics_sphere.py
python examples/embedded_sphere.py
python examples/torus.py
python examples/cylinder.py
python examples/extrinsic_sphere.py
python examples/extrinsic_cylinder.py
python examples/extrinsic_torus.py
python examples/minimal_surfaces.py
python examples/numeric_geodesic_sphere.py
python examples/numeric_geodesic_cylinder.py
python examples/numeric_geodesic_poincare.py
python examples/numeric_geodesic_torus.py
```

## Tests

```bash
pytest
```

## Limitations

CRR v0.5 is intentionally conservative. Numerical geodesics are solved in local coordinates, so coordinate singularities and chart boundaries remain the user's responsibility. Standard adaptive SciPy solvers may show numerical energy drift over long integrations; CRR does not yet include a symplectic or variational geodesic integrator. The package also does not yet include a full abstract tensor algebra system, differential forms, coordinate transformations between charts, sparse storage, arbitrary-codimension extrinsic geometry, non-Euclidean ambient extrinsic curvature, or optimized simplification strategies.
