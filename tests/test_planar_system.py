from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "planar_system"
sys.path.insert(0, str(EXAMPLE_DIR))

from planar import (  # noqa: E402
    build_planar_system,
    evaluate_planar_ssm_graph,
    planar_nonlinear_coefficients,
    planar_nonlinear_exponents,
    planar_ssm_graph_coefficients,
    planar_vector_field,
)


def test_build_planar_system_matches_matlab_build_model() -> None:
    a, b = build_planar_system()

    np.testing.assert_allclose(
        np.asarray(a),
        np.array([[-1.0, 0.0], [0.0, -np.sqrt(24.0)]]),
    )
    np.testing.assert_allclose(np.asarray(b), np.eye(2))


def test_planar_vector_field_matches_tensor_terms() -> None:
    z = jnp.array([0.5, -0.25])
    expected = jnp.array(
        [
            -0.5,
            -jnp.sqrt(24.0) * (-0.25) + 0.5**2 + 0.5**3 + 0.5**4 + 0.5**5,
        ]
    )

    np.testing.assert_allclose(np.asarray(planar_vector_field(z)), np.asarray(expected))


def test_planar_sparse_terms_match_matlab_build_model() -> None:
    np.testing.assert_array_equal(
        np.asarray(planar_nonlinear_exponents()),
        np.array([[2, 0], [3, 0], [4, 0], [5, 0]]),
    )
    np.testing.assert_allclose(
        np.asarray(planar_nonlinear_coefficients()),
        np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
    )


def test_planar_graph_coefficients_match_demo_formula() -> None:
    coefficients = planar_ssm_graph_coefficients(order=8)
    expected = np.zeros(9)
    for degree in range(2, 6):
        expected[degree] = 1.0 / (np.sqrt(24.0) - degree)

    np.testing.assert_allclose(np.asarray(coefficients), expected, rtol=1e-12)


def test_planar_graph_coefficients_are_computed_by_solver() -> None:
    coefficients = planar_ssm_graph_coefficients(order=5)
    residual = jnp.array(
        [
            (degree * -1.0 + jnp.sqrt(24.0)) * coefficients[degree] - 1.0
            for degree in range(2, 6)
        ]
    )

    np.testing.assert_allclose(np.asarray(residual), np.zeros(4), atol=1e-14)


def test_planar_graph_coefficients_shape_and_validation() -> None:
    assert planar_ssm_graph_coefficients(order=0).shape == (1,)
    assert planar_ssm_graph_coefficients(order=5).shape == (6,)

    with pytest.raises(ValueError, match="order must be nonnegative"):
        planar_ssm_graph_coefficients(order=-1)


def test_planar_graph_satisfies_invariance_equation_through_order_five() -> None:
    coefficients = planar_ssm_graph_coefficients(order=8)
    degrees = jnp.arange(coefficients.shape[0])

    x = jnp.linspace(-0.15, 0.15, 7)
    h = evaluate_planar_ssm_graph(x, coefficients)
    dh = jnp.sum(degrees[1:] * coefficients[1:] * x[..., None] ** (degrees[1:] - 1), axis=-1)
    residual = -x * dh + jnp.sqrt(24.0) * h - (x**2 + x**3 + x**4 + x**5)

    np.testing.assert_allclose(np.asarray(residual), np.zeros_like(np.asarray(x)), atol=1e-14)


def test_planar_public_functions_support_jax_transforms() -> None:
    jacobian = jax.jacfwd(planar_vector_field)(jnp.array([0.2, -0.1]))
    np.testing.assert_allclose(
        np.asarray(jacobian),
        np.array(
            [
                [-1.0, 0.0],
                [
                    2 * 0.2 + 3 * 0.2**2 + 4 * 0.2**3 + 5 * 0.2**4,
                    -np.sqrt(24.0),
                ],
            ]
        ),
        rtol=1e-12,
    )

    grad_value = jax.grad(
        lambda decay: jnp.sum(planar_ssm_graph_coefficients(5, decay))
    )(jnp.sqrt(24.0))
    expected_grad = -sum(1.0 / (np.sqrt(24.0) - degree) ** 2 for degree in range(2, 6))
    np.testing.assert_allclose(np.asarray(grad_value), expected_grad, rtol=1e-12)

    jitted = jax.jit(lambda xs: evaluate_planar_ssm_graph(xs, planar_ssm_graph_coefficients(5)))
    np.testing.assert_allclose(
        np.asarray(jitted(jnp.array([0.0, 0.1]))),
        np.asarray(evaluate_planar_ssm_graph(jnp.array([0.0, 0.1]), planar_ssm_graph_coefficients(5))),
    )
