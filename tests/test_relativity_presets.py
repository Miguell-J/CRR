import sympy as sp

import crr
import crr.presets as presets
from crr.presets import (
    alcubierre_metric,
    anti_de_sitter_metric,
    de_sitter_metric,
    eddington_finkelstein_metric,
    flrw_metric,
    godel_metric,
    kerr_metric,
    kruskal_szekeres_metric,
    minkowski_metric,
    reissner_nordstrom_metric,
    robertson_walker_metric,
    schwarzschild_metric,
)


EXTENDED_PRESET_NAMES = [
    "minkowski_metric",
    "schwarzschild_metric",
    "flrw_metric",
    "robertson_walker_metric",
    "reissner_nordstrom_metric",
    "kerr_metric",
    "kruskal_szekeres_metric",
    "eddington_finkelstein_metric",
    "de_sitter_metric",
    "anti_de_sitter_metric",
    "godel_metric",
    "alcubierre_metric",
]


def assert_symmetric(metric):
    assert metric.components == metric.components.T


def test_extended_relativity_presets_export_from_presets_and_top_level():
    for name in EXTENDED_PRESET_NAMES:
        assert callable(getattr(presets, name))
        assert callable(getattr(crr, name))


def test_reissner_nordstrom_reduces_to_schwarzschild_when_charge_zero():
    t, r, theta, phi = sp.symbols("t r theta phi")
    M = sp.symbols("M", positive=True)
    rn = reissner_nordstrom_metric(mass=M, charge=0, coordinates=[t, r, theta, phi])
    sch = schwarzschild_metric(mass=M, coordinates=[t, r, theta, phi])

    assert_symmetric(rn)
    assert rn.components.equals(sch.components)


def test_kerr_reduces_to_schwarzschild_when_spin_zero():
    t, r, theta, phi = sp.symbols("t r theta phi")
    M = sp.symbols("M", positive=True)
    kerr = kerr_metric(mass=M, spin=0, coordinates=[t, r, theta, phi])
    sch = schwarzschild_metric(mass=M, coordinates=[t, r, theta, phi])

    assert_symmetric(kerr)
    assert sp.simplify(kerr.components[0, 0] - sch.components[0, 0]) == 0
    assert sp.simplify(kerr.components[1, 1] - sch.components[1, 1]) == 0
    assert sp.simplify(kerr.components[0, 3]) == 0


def test_robertson_walker_matches_flrw_components():
    t, r, theta, phi = sp.symbols("t r theta phi")
    a = sp.Function("a")(t)
    k = sp.symbols("k")
    rw = robertson_walker_metric(scale_factor=a, curvature=k, coordinates=[t, r, theta, phi])
    flrw = flrw_metric(scale_factor=a, curvature=k, coordinates=[t, r, theta, phi])

    assert rw.components.equals(flrw.components)


def test_eddington_finkelstein_cross_terms_and_angular_components():
    ingoing = eddington_finkelstein_metric(ingoing=True)
    outgoing = eddington_finkelstein_metric(ingoing=False)
    _v, r_in, theta_in, _phi = ingoing.coordinates
    _u, r_out, theta_out, _phi_out = outgoing.coordinates

    assert_symmetric(ingoing)
    assert_symmetric(outgoing)
    assert ingoing.components[0, 1] == 1
    assert outgoing.components[0, 1] == -1
    assert ingoing.components[2, 2] == r_in**2
    assert ingoing.components[3, 3] == r_in**2 * sp.sin(theta_in) ** 2
    assert outgoing.components[2, 2] == r_out**2
    assert outgoing.components[3, 3] == r_out**2 * sp.sin(theta_out) ** 2


def test_de_sitter_and_anti_de_sitter_scalar_curvatures():
    L = sp.symbols("L", positive=True)

    assert sp.simplify(de_sitter_metric(radius=L).scalar_curvature(simplify=True) - 12 / L**2) == 0
    assert sp.simplify(anti_de_sitter_metric(radius=L).scalar_curvature(simplify=True) + 12 / L**2) == 0


def test_kruskal_szekeres_components_and_cross_term_convention():
    metric = kruskal_szekeres_metric()
    U, V, theta, _phi = metric.coordinates
    M = sp.symbols("M", positive=True)
    radius = sp.Function("r")(U, V)
    expected_cross = -16 * M**3 * sp.exp(-radius / (2 * M)) / radius

    assert_symmetric(metric)
    assert sp.simplify(metric.components[0, 1] - expected_cross) == 0
    assert metric.components[2, 2] == radius**2
    assert metric.components[3, 3] == radius**2 * sp.sin(theta) ** 2


def test_godel_metric_cross_term_convention():
    metric = godel_metric()
    _t, x, _y, _z = metric.coordinates
    a = sp.symbols("a", positive=True)

    assert_symmetric(metric)
    assert metric.components[0, 2] == -a**2 * sp.exp(x)
    assert metric.components[2, 0] == -a**2 * sp.exp(x)


def test_alcubierre_metric_cross_term_and_minkowski_limits():
    v_s, f_s = sp.symbols("v_s f_s", real=True)
    metric = alcubierre_metric(velocity=v_s, shape_function=f_s)

    assert_symmetric(metric)
    assert metric.components[0, 1] == -v_s * f_s
    assert alcubierre_metric(velocity=0).components.equals(minkowski_metric().components)
    assert alcubierre_metric(shape_function=0).components.equals(minkowski_metric().components)
