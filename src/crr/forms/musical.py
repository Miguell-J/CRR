"""Musical isomorphisms between vector fields and 1-forms."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from crr.forms.differential_form import DifferentialForm


def flat(metric: "Metric", vector: Sequence[object], simplify: bool = False) -> DifferentialForm:
    """Lower a vector field index with the metric."""

    if len(vector) != metric.dimension:
        raise ValueError("Vector length must match metric dimension.")
    components = []
    for i in range(metric.dimension):
        value = sp.S.Zero
        for j in range(metric.dimension):
            value += metric.components[i, j] * sp.sympify(vector[j])
        components.append(sp.simplify(value) if simplify else value)
    return DifferentialForm.one_form(metric.manifold, components)


def sharp(metric: "Metric", one_form: DifferentialForm, simplify: bool = False) -> list[sp.Expr]:
    """Raise a 1-form index with the inverse metric."""

    if one_form.manifold != metric.manifold or one_form.degree != 1:
        raise ValueError("A 1-form on the metric manifold is required.")
    inverse = metric.inverse()
    vector = []
    for i in range(metric.dimension):
        value = sp.S.Zero
        for j in range(metric.dimension):
            value += inverse[i, j] * one_form[(j,)]
        vector.append(sp.simplify(value) if simplify else value)
    return vector


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
