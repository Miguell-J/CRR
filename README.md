# CRR: Curvature, Ricci, and Riemann

CRR is a small symbolic and numerical differential geometry package for computing geometric objects from a coordinate chart, a metric tensor, a parametrized embedding, coordinate-basis differential forms, or finite-dimensional Lie algebras. The v0.9 scope focuses on readable SymPy-based computations, numerical geodesic solving, coordinate-domain integration, convenient presets, exterior calculus, Jupyter-friendly display, optional matplotlib visualization, and basic Lie algebra tools for future gauge theory support.

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

For notebook use:

```bash
python -m pip install -e ".[dev,viz,notebook]"
jupyter lab
```

Notebook-style scripts in `notebooks/` include a small source-checkout bootstrap, so they can also be run directly before installation from the project root:

```bash
python notebooks/01_sphere_geometry.py
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

## v0.6 Integration And Global Quantities

For a metric `g`, CRR uses the coordinate volume density:

```text
sqrt(abs(det(g)))
```

For positive definite Riemannian metrics, this is the usual `sqrt(det(g))`. You can request the non-absolute symbolic density with `absolute=False`.

```python
x, y = sp.symbols("x y")
g = Metric(Manifold("R2", 2, [x, y]), [[1, 0], [0, 1]])

area = g.integrate_volume([(x, 0, 2), (y, 0, 3)], simplify=True)
value = g.integrate_scalar(x + y, [(x, 0, 1), (y, 0, 1)], simplify=True)
numeric = g.integrate_scalar_numeric(x + y, [(x, 0, 1), (y, 0, 1)])
```

For parametrized surfaces, CRR can integrate area, scalar fields, Gaussian curvature, and mean curvature over parameter domains:

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

area = sphere.integrate_area([
    (theta, 0, sp.pi),
    (phi, 0, 2 * sp.pi),
], simplify=True)

print(area)  # 4*pi
```

Total Gaussian curvature:

```python
total_K = sphere.integrate_gaussian_curvature([
    (theta, 0, sp.pi),
    (phi, 0, 2 * sp.pi),
], simplify=True)
```

Gauss-Bonnet helper:

```python
from crr import gauss_bonnet_check

result = gauss_bonnet_check(
    sphere,
    [(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)],
    euler_characteristic=2,
)

print(result.curvature_integral)
print(result.expected)
print(result.passed)
```

For a 2D surface, intrinsic scalar curvature and Gaussian curvature satisfy `R = 2K`, so `integrate_gaussian_curvature(..., intrinsic=True)` uses the pullback metric scalar curvature divided by 2. Use `intrinsic=False` to use the extrinsic Gaussian curvature method.

## v0.7 Presets

CRR includes a preset library for common metrics and embedded surfaces:

```python
from crr.presets import sphere_surface, sphere_metric

sphere = sphere_surface(radius=1)
g = sphere.pullback_metric()
print(g.scalar_curvature(simplify=True))

g2 = sphere_metric(radius=1)
print(g2.scalar_curvature(simplify=True))
```

Surface presets return `ParametrizedMap` objects:

```python
from crr.presets import (
    plane_surface,
    sphere_surface,
    cylinder_surface,
    torus_surface,
    catenoid_surface,
    helicoid_surface,
    mobius_strip_surface,
)
```

Metric presets return `Metric` objects:

```python
from crr.presets import (
    euclidean_metric,
    polar_metric,
    sphere_metric,
    hyperbolic_half_plane_metric,
    poincare_disk_metric,
    cylinder_metric,
    torus_metric,
)
```

Relativity presets:

```python
from crr.presets import (
    minkowski_metric,
    schwarzschild_metric,
    flrw_metric,
    robertson_walker_metric,
    reissner_nordstrom_metric,
    kerr_metric,
    eddington_finkelstein_metric,
    kruskal_szekeres_metric,
    godel_metric,
    de_sitter_metric,
    anti_de_sitter_metric,
    alcubierre_metric,
)

eta = minkowski_metric(signature="-+++")
sch = schwarzschild_metric()
flrw = flrw_metric()
```

All relativity presets support `signature="-+++"`; `signature="+---"` flips the full line element. Matrix cross terms follow the standard convention: a line-element term `2*g_ab dx^a dx^b` corresponds to symmetric matrix entries `g_ab = g_ba`.

Coordinate and convention summary:

- `minkowski_metric`: Cartesian-like coordinates, default `(t, x, y, z)`.
- `schwarzschild_metric`: standard Schwarzschild coordinates `(t, r, theta, phi)`, `G=c=1`.
- `flrw_metric` and `robertson_walker_metric`: the same standard cosmological metric form in spherical spatial coordinates.
- `reissner_nordstrom_metric`: standard static coordinates with `f(r) = 1 - 2M/r + Q**2/r**2`.
- `kerr_metric`: Boyer-Lindquist coordinates with `Sigma = r**2 + a**2*cos(theta)**2` and `Delta = r**2 - 2Mr + a**2`.
- `eddington_finkelstein_metric`: advanced ingoing coordinates by default with `g_vr = 1`; set `ingoing=False` for retarded outgoing coordinates with `g_ur = -1`.
- `kruskal_szekeres_metric`: coordinates `(U, V, theta, phi)` with implicit areal radius `r(U,V)`. The `dU dV` coefficient is represented by `g_UV = g_VU = -16*M**3*exp(-r/(2M))/r`, so the line element contains `2*g_UV dU dV`.
- `godel_metric`: coordinates `(t, x, y, z)` and `ds^2 = a^2[-dt^2 + dx^2 - 1/2 exp(2x)dy^2 + dz^2 - 2 exp(x)dt dy]`.
- `de_sitter_metric` and `anti_de_sitter_metric`: static spherical coordinates with radius `L`; scalar curvatures are `12/L**2` and `-12/L**2`.
- `alcubierre_metric`: simplified symbolic warp profile value `f_s` by default, with `ds^2 = -dt^2 + (dx - v_s f_s dt)^2 + dy^2 + dz^2`.

`kruskal_metric` remains available as a backward-compatible alias for `kruskal_szekeres_metric`.

Surface presets and metric presets are intentionally connected but separate. For example, `sphere_surface(radius=R).pullback_metric()` and `sphere_metric(radius=R)` describe the same intrinsic round metric in the standard chart. The surface preset also provides embedding-specific operations such as normals, second fundamental forms, and extrinsic Gaussian curvature.

Preset functions accept custom coordinate symbols where useful:

```python
theta, phi = sp.symbols("theta phi")
g = sphere_metric(radius=2, coordinates=[theta, phi])
```

## v0.8 Differential Forms

CRR includes a coordinate-basis `DifferentialForm` class for exterior calculus in dimensions 1 through 4 and beyond when expressions remain tractable.

```python
import sympy as sp
from crr import Manifold, DifferentialForm
from crr.presets import euclidean_metric

x, y, z = sp.symbols("x y z")
M = Manifold("R3", 3, [x, y, z])
g = euclidean_metric(dim=3, coordinates=[x, y, z])

dx = DifferentialForm.basis(M, 0)
dy = DifferentialForm.basis(M, 1)
dz = DifferentialForm.basis(M, 2)

omega = dx.wedge(dy)
print(omega)
print(omega.hodge_star(g))
```

Forms store antisymmetric components in canonical sorted index order:

```python
f = DifferentialForm.scalar(M, x*y)
df = f.exterior_derivative()
```

Supported operations include:

```python
alpha.wedge(beta)
alpha.exterior_derivative()
alpha.hodge_star(g, simplify=True)
alpha.codifferential(g, simplify=True)
alpha.hodge_laplacian(g, simplify=True)
alpha.integrate(ranges, simplify=True)
```

Hodge star convention:

```text
(*alpha)_{i_{k+1}...i_n}
= sqrt(det(g))/k! * alpha^{j_1...j_k}
  epsilon_{j_1...j_k i_{k+1}...i_n}
```

The coordinate order fixes the orientation. This convention is intended for Riemannian metrics; pseudo-Riemannian Hodge star behavior is experimental.

Codifferential convention:

```text
delta alpha = (-1)^(n*k+n+1) * d * alpha
Delta alpha = d delta alpha + delta d alpha
```

With this convention, on Euclidean `R2`, the Hodge Laplacian of `x**2 + y**2` is `-4`.

Metric musical maps are available:

```python
one_form = g.flat([sp.Symbol("P"), sp.Symbol("Q"), sp.Symbol("R")])
vector = g.sharp(one_form)
```

Toy Maxwell example:

```python
t, x, y, z = sp.symbols("t x y z")
M4 = Manifold("R4", 4, [t, x, y, z])
A = DifferentialForm.one_form(M4, [-sp.Function("Phi")(t, x, y, z), 0, 0, 0])
F = A.exterior_derivative()
print(F.exterior_derivative().simplify())  # dF = 0
```

## v0.8.1 Jupyter Display

CRR includes lightweight display helpers for notebook and scientific exploration workflows. IPython is optional: display methods return plain strings when notebook rendering is not available.

```python
from crr.presets import sphere_metric

g = sphere_metric(radius=1)
g.display_matrix()
g.display_christoffel_nonzero(simplify=True)
g.display_scalar_curvature(simplify=True)
```

Useful objects expose compact display or formatting helpers:

```python
tensor.to_markdown_table()
tensor.to_latex_components()
form.to_markdown()
form.display_components()
surface.display_parametrization()
surface.display_induced_metric(simplify=True)
```

Notebook-style scripts live in `notebooks/` and use `# %%` cell markers, so they can be opened as notebooks in VS Code, Jupyter-compatible editors, or run as plain Python scripts.

## v0.8.2 Visualization

Visualization is optional and matplotlib-based. Install it with:

```bash
python -m pip install -e ".[viz]"
```

Parametrized surfaces can be plotted directly:

```python
import numpy as np
from crr.presets import sphere_surface, torus_surface

sphere = sphere_surface(radius=1)
sphere.plot_surface((0, np.pi), (0, 2*np.pi), show=False)

torus = torus_surface(major_radius=2, minor_radius=0.7)
torus.plot_curvature(
    u_range=(0, 2*np.pi),
    v_range=(0, 2*np.pi),
    curvature="gaussian",
    show=False,
)
```

Numerical geodesics can be visualized in coordinates, phase components, energy, speed, or embedded on a surface:

```python
from crr.presets import sphere_metric, sphere_surface

g = sphere_metric(radius=1)
surface = sphere_surface(radius=1)
sol = g.solve_geodesic([np.pi/2, 0], [0, 1], (0, 2*np.pi), num_points=300)

sol.plot_coordinates(show=False)
sol.plot_energy(show=False)
surface.plot_surface_with_geodesic(sol, (0, np.pi), (0, 2*np.pi), show=False)
```

Scalar fields, density fields, vector fields, and 2D forms are also supported:

```python
import sympy as sp
from crr import DifferentialForm, Manifold
from crr.visualization import plot_scalar_field_2d, plot_form_2d

x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])
dx = DifferentialForm.basis(M, 0)
dy = DifferentialForm.basis(M, 1)

plot_scalar_field_2d(x**2 - y**2, [x, y], (-2, 2), (-2, 2), show=False)
plot_form_2d(-y*dx + x*dy, (-2, 2), (-2, 2), show=False)
```

Plotting functions return `(fig, ax)` and accept `save_path=...` for script and CI-style workflows. Visualization remains chart-based: coordinate singularities, local domains, and symbolic complexity are still visible in plots. There is no Plotly or interactive 3D backend yet; substitute numeric parameters before plotting heavy symbolic surfaces or curvature fields.

## v0.9 Lie Algebras and Algebra-Valued Forms

CRR includes a small symbolic Lie algebra module for finite-dimensional algebras specified by structure constants. The convention is:

```text
[e_i, e_j] = sum_k c^k_ij e_k
```

For elements `x = x^i e_i` and `y = y^j e_j`, the bracket is:

```text
[x, y]^k = c^k_ij x^i y^j
```

Basic usage:

```python
from crr.lie import su2_algebra

su2 = su2_algebra()
e1, e2, e3 = [su2.basis_element(i) for i in range(3)]

print(e1.bracket(e2))          # T3
print(su2.check_jacobi())      # True
print(su2.killing_form_matrix())
```

Available presets:

```python
from crr.lie import (
    abelian_lie_algebra,
    so2_algebra,
    so3_algebra,
    su2_algebra,
    sl2r_algebra,
    heisenberg_algebra,
)
```

The `su2_algebra()` preset currently uses the real compact convention structurally equivalent to `so(3)`, with `[T1,T2]=T3`, `[T2,T3]=T1`, and `[T3,T1]=T2`; no explicit matrix `i` factors are included.

Matrix helpers are available for simple matrix Lie algebras:

```python
from crr.lie import matrix_commutator, matrix_lie_algebra_from_basis
```

Algebra-valued forms are represented as:

```text
A = sum_a A^a e_a
```

where each `A^a` is a CRR `DifferentialForm`. The bracket wedge convention is:

```text
[A wedge B]^k = sum_ij c^k_ij A^i wedge B^j
```

No factor of `1/2` is included in `bracket_wedge`; gauge curvature conventions can choose it explicitly:

```python
import sympy as sp
from crr import Manifold, DifferentialForm
from crr.lie import AlgebraValuedForm, su2_algebra

x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])
dx = DifferentialForm.basis(M, 0)
dy = DifferentialForm.basis(M, 1)

su2 = su2_algebra()
A = AlgebraValuedForm(su2, [x*dx, y*dy, DifferentialForm.zero(M, 1)])

F_preview = A.exterior_derivative() + sp.Rational(1, 2) * A.bracket_wedge(A)
print(F_preview.to_latex())
```

This prepares the next gauge-field layer: connections can be represented as Lie-algebra-valued 1-forms, and curvature can be built as `F = dA + 1/2 [A wedge A]` or another convention chosen explicitly by a future `GaugeField` API.

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
python examples/integrate_sphere_area.py
python examples/gauss_bonnet_sphere.py
python examples/gauss_bonnet_torus.py
python examples/cylinder_area.py
python examples/scalar_density_integral.py
python examples/preset_sphere.py
python examples/preset_torus.py
python examples/preset_relativity.py
python examples/preset_geodesic.py
python examples/extended_relativity_presets.py
python examples/kerr_and_rn_limits.py
python examples/de_sitter_ads.py
python examples/alcubierre_metric.py
python examples/forms_basic.py
python examples/hodge_star_euclidean.py
python examples/maxwell_forms_flat.py
python examples/hodge_laplacian_scalar.py
python examples/sphere_area_form.py
python examples/plot_sphere_surface.py
python examples/plot_torus_curvature.py
python examples/plot_sphere_geodesic.py
python examples/plot_torus_geodesic.py
python examples/plot_forms_2d.py
python examples/plot_geodesic_energy.py
python examples/lie_so3.py
python examples/lie_sl2r.py
python examples/lie_heisenberg.py
python examples/algebra_valued_forms.py
python examples/lie_for_gauge_preview.py
python examples/gauge_u1_maxwell.py
python examples/gauge_abelian_transform.py
python examples/gauge_su2_curvature.py
python examples/gauge_bianchi_identity.py
python examples/yang_mills_action_density.py
```

## CRR v1.0 Gauge Fields

CRR v1.0 adds local symbolic gauge theory built on the existing differential-form and Lie-algebra infrastructure. A gauge potential is represented as a Lie-algebra-valued 1-form

```text
A = sum_a A^a e_a in Omega^1(M, g)
```

The default curvature convention is

```text
F_A = dA + 1/2 [A wedge A]
```

where the bracket wedge combines the exterior wedge product with the Lie algebra bracket. The alternate `convention="matrix"` option computes `F = dA + [A wedge A]` for matrix-style local conventions. For abelian gauge fields the bracket term vanishes, so Maxwell field strength is simply `F = dA`.

```python
import sympy as sp
from crr import Manifold, DifferentialForm
from crr.gauge import abelian_gauge_potential

x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])
A_form = x * DifferentialForm.basis(M, 1)
A = abelian_gauge_potential(M, A_form)
F = A.curvature()
print(F)
```

The gauge module also provides:

- `GaugePotential` and `GaugeCurvature`
- `covariant_exterior_derivative(A, omega)` for `d_A omega = d omega + [A wedge omega]`
- `GaugePotential.bianchi_identity()` for `d_A F`
- `abelian_gauge_transform(A, chi)` for `A -> A + d chi`
- `infinitesimal_gauge_transform(A, epsilon)` for `A -> A + d_A epsilon`
- `yang_mills_action_density(F, metric)` for the local top form `sum_ab <e_a,e_b> F^a wedge *F^b`
- presets for U(1), abelian pure gauges, and su(2) connections

The default Yang-Mills Lie-algebra inner product is the identity matrix, which is a practical positive convention for compact examples such as the real `su(2)` preset. Pass `inner_product=` explicitly for a different invariant form or normalization.

Gauge support is intentionally local: CRR does not yet model principal bundle topology, transition functions, finite non-abelian gauge transformations, automatic Chern-Weil classes, or full pseudo-Riemannian Yang-Mills sign conventions.

## CRR v1.0.1 Audit And Stabilization

CRR v1.0.1 adds a runnable audit layer for capability reporting, public API checks, examples validation, and lightweight benchmarks. This release does not add major mathematical features; it documents and verifies the scientific surface that exists now.

Audit documentation:

- [Capabilities](docs/CAPABILITIES.md)
- [API audit](docs/API_AUDIT.md)
- [Performance notes](docs/PERFORMANCE_NOTES.md)

Run the capability report:

```bash
python -m crr.audit capabilities
python examples/audit_capabilities.py
```

Run public API checks:

```bash
python -m crr.audit api
python examples/audit_api.py
```

Run lightweight benchmarks:

```bash
python -m crr.audit benchmarks
python examples/audit_benchmarks.py
```

CRR's current scientific scope is a local-coordinate symbolic/numeric laboratory for metrics, tensors, forms, Lie algebras, local gauge fields, embedded surfaces, geodesics, relativity presets, and chart-based visualization. It is not yet a full global differential geometry system: there is no automatic atlas, chart gluing, principal bundle topology, transition-function machinery, or global cohomology engine.

## CRR v1.0.2 API Polish

CRR v1.0.2 normalizes display and serialization method names across public objects, adds benchmark baseline save/compare helpers, and documents simplification, caching, and API stability conventions.

Additional documentation:

- [Simplification and caching](docs/SIMPLIFICATION_AND_CACHING.md)
- [API stability](docs/API_STABILITY.md)

Benchmark baselines:

```bash
python -m crr.audit benchmarks --quick --save-baseline /tmp/crr_benchmark_baseline.json
python -m crr.audit benchmarks --quick --compare /tmp/crr_benchmark_baseline.json
```

Recommended workflows:

- Symbolic exploration: compute first, simplify late, and inspect representative components before simplifying full tensors.
- Jupyter notebooks: use `display()`, `to_latex()`, `to_markdown()`, and `display_nonzero()` for compact outputs.
- Visualization: substitute numeric parameters before plotting and start with low resolution.
- Relativity metrics: avoid full curvature simplification until the needed components are identified.
- Gauge examples: use the identity Lie-algebra inner product for compact examples such as `su2` unless a specific invariant form is required.

## Tests And Quality Checks

Run the full test suite from an editable install:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m mypy
```

For source-checkout smoke checks without installing the package, use:

```bash
PYTHONPATH=src python -m crr.audit api
PYTHONPATH=src python -m crr.audit benchmarks --quick
PYTHONPATH=src python -m compileall -q src tests examples notebooks
```

The repository also includes a GitHub Actions workflow that installs the package, compiles the Python files, runs pytest, runs lint/type checks, and executes the API/benchmark audit commands on supported Python versions.
The repository also includes a GitHub Actions workflow that installs the package, compiles the Python files, runs pytest, and executes the API/benchmark audit commands on supported Python versions.

## Limitations

CRR v1.0 is intentionally conservative. Differential forms are coordinate-basis forms; there is no automatic global atlas, transition map, or cohomology engine. The Hodge star uses coordinate-order orientation and is primarily tested for Riemannian metrics. Lie algebras are finite-dimensional and structure-constant based; CRR does not yet implement Lie groups, exponential maps, principal bundle topology, transition functions, finite non-abelian gauge transformations, Chern-Weil classes, or full Yang-Mills equations. Presets are conveniences, not a global atlas system; they use standard coordinate charts and inherit their singularities and domain restrictions. Integrations and visualizations are over explicit coordinate domains; CRR does not automatically handle chart gluing, boundary identifications, singular coordinate endpoints, or global rendering of multiple charts. Symbolic integration may fail or return unevaluated SymPy integrals, and numeric integration is currently a simple SciPy fallback for low-dimensional boxes. Numerical geodesics are solved in local coordinates, so coordinate singularities and chart boundaries remain the user's responsibility. Standard adaptive SciPy solvers may show numerical energy drift over long integrations; CRR does not yet include a symplectic or variational geodesic integrator. Visualization is matplotlib-only and non-interactive by default. The package also does not yet include a full abstract tensor algebra system beyond the current dense coordinate components, sparse storage, arbitrary-codimension extrinsic geometry, non-Euclidean ambient extrinsic curvature, or optimized simplification strategies.
