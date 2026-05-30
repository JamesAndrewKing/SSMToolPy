from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "planar_system"
sys.path.insert(0, str(EXAMPLE_DIR))

from planar import (  # noqa: E402
    evaluate_planar_ssm_graph,
    planar_ssm_graph_coefficients,
)


def _planar_graph_loss(decay: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    coefficients = planar_ssm_graph_coefficients(order=5, decay=decay)
    x = jnp.array([-0.12, -0.04, 0.05, 0.11])
    prediction = evaluate_planar_ssm_graph(x, coefficients)
    return jnp.mean((prediction - target) ** 2)


def test_planar_fixed_structure_parameter_to_loss_gradient_smoke() -> None:
    """Differentiate parameter -> graph coefficients -> prediction -> loss.

    This is the smallest current analogue of the desired parameter-to-loss
    workflow. It keeps all discrete choices fixed: selected mode, graph
    structure, truncation order, and nonresonant denominator pattern.
    """

    parameter = jnp.array(jnp.sqrt(24.0))
    target = jnp.array([0.01, -0.002, 0.003, 0.02])

    loss = _planar_graph_loss(parameter, target)
    gradient = jax.grad(_planar_graph_loss)(parameter, target)

    eps = 1e-6
    finite_difference = (
        _planar_graph_loss(parameter + eps, target)
        - _planar_graph_loss(parameter - eps, target)
    ) / (2.0 * eps)

    assert loss.shape == ()
    assert gradient.shape == ()
    assert np.isfinite(np.asarray(loss))
    assert np.isfinite(np.asarray(gradient))
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(finite_difference), rtol=1e-6)
