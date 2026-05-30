from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy.core.graph import (
    evaluate_univariate_graph,
    linear_reduced_trajectory,
    two_sided_graph_curve,
)


def test_evaluate_univariate_graph_lifts_vector_coefficients() -> None:
    coefficients = jnp.array([[0.0, 0.0], [1.0, -1.0], [2.0, 3.0]])
    reduced = jnp.array([0.0, 0.5])
    values = evaluate_univariate_graph(reduced, coefficients)

    expected = jnp.array([[0.0, 0.0], [1.0, 0.25]])
    np.testing.assert_allclose(np.asarray(values), np.asarray(expected), rtol=1e-12)


def test_linear_reduced_trajectory_values_and_grad() -> None:
    times = jnp.array([0.0, 0.25, 0.5])
    trajectory = linear_reduced_trajectory(2.0, times, 0.5)

    np.testing.assert_allclose(
        np.asarray(trajectory),
        np.asarray(2.0 * jnp.exp(0.5 * times)),
        rtol=1e-12,
    )

    gradient = jax.grad(
        lambda eigenvalue: jnp.sum(linear_reduced_trajectory(2.0, times, eigenvalue))
    )(jnp.array(0.5))
    assert gradient.shape == ()
    assert np.isfinite(np.asarray(gradient))


def test_graph_trajectory_and_two_sided_curve_shape() -> None:
    coefficients = jnp.array([[0.0, 0.0], [1.0, 2.0]])
    times = jnp.linspace(0.0, 0.2, 4)
    reduced = linear_reduced_trajectory(0.1, times, 1.0)
    lifted = evaluate_univariate_graph(reduced, coefficients)
    curve = two_sided_graph_curve(times, 0.1, 1.0, coefficients)

    assert lifted.shape == (4, 2)
    assert curve.shape == (8, 2)
    np.testing.assert_allclose(
        np.asarray(curve[:4]),
        np.asarray(
            evaluate_univariate_graph(
                linear_reduced_trajectory(-0.1, times, 1.0),
                coefficients,
            )[::-1]
        ),
        rtol=1e-12,
    )
