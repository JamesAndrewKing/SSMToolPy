from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ssmtoolpy.core.invariance import (
    solve_autonomous_quadratic_graph_coefficients,
    solve_scalar_graph_coefficients,
    univariate_graph_invariance_residual,
)
from ssmtoolpy.core.multiindex import multiindices_of_total_degree
from ssmtoolpy.core.polynomial import (
    collect_univariate_coefficients,
    evaluate_monomial_polynomial,
)


def test_multiindices_of_total_degree_one_and_two_dimensions() -> None:
    np.testing.assert_array_equal(np.asarray(multiindices_of_total_degree(1, 3)), [[3]])
    np.testing.assert_array_equal(
        np.asarray(multiindices_of_total_degree(2, 3)),
        np.array([[3, 0], [2, 1], [1, 2], [0, 3]]),
    )


def test_multiindices_of_total_degree_validation() -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        multiindices_of_total_degree(0, 1)
    with pytest.raises(ValueError, match="degree must be nonnegative"):
        multiindices_of_total_degree(2, -1)


def test_evaluate_monomial_polynomial_source_derived_case() -> None:
    x = jnp.array([2.0, -3.0])
    exponents = jnp.array([[1, 0], [0, 1], [2, 1]], dtype=jnp.int32)
    coefficients = jnp.array(
        [
            [1.0, 2.0, 0.5],
            [-1.0, 0.0, 3.0],
        ]
    )

    expected = jnp.array(
        [
            1.0 * 2.0 + 2.0 * (-3.0) + 0.5 * (2.0**2) * (-3.0),
            -1.0 * 2.0 + 3.0 * (2.0**2) * (-3.0),
        ]
    )
    np.testing.assert_allclose(
        np.asarray(evaluate_monomial_polynomial(x, exponents, coefficients)),
        np.asarray(expected),
    )


def test_collect_univariate_coefficients_and_validation() -> None:
    exponents = jnp.array([[2], [4], [2]], dtype=jnp.int32)
    coefficients = jnp.array([1.5, -2.0, 0.25])

    np.testing.assert_allclose(
        np.asarray(collect_univariate_coefficients(exponents, coefficients, order=5)),
        np.array([0.0, 0.0, 1.75, 0.0, -2.0, 0.0]),
    )

    with pytest.raises(ValueError, match="order must be nonnegative"):
        collect_univariate_coefficients(exponents, coefficients, order=-1)
    with pytest.raises(ValueError, match=r"exponents must have shape"):
        collect_univariate_coefficients(jnp.array([2, 4]), coefficients, order=5)
    with pytest.raises(ValueError, match=r"coefficients must have shape"):
        collect_univariate_coefficients(exponents, coefficients[:, None], order=5)
    with pytest.raises(ValueError, match="same number of terms"):
        collect_univariate_coefficients(exponents, coefficients[:2], order=5)


def test_solve_scalar_graph_coefficients_matches_homological_equation() -> None:
    forcing = jnp.array([0.0, 0.0, 1.0, -2.0, 0.5])
    coefficients = solve_scalar_graph_coefficients(
        master_eigenvalue=-1.0,
        transverse_eigenvalue=-5.0,
        forcing_coefficients=forcing,
    )
    expected = np.array([0.0, 0.0, 1.0 / 3.0, -2.0 / 2.0, 0.5 / 1.0])

    np.testing.assert_allclose(np.asarray(coefficients), expected, rtol=1e-12)


def test_differentiable_core_functions_support_jax_transforms() -> None:
    exponents = jnp.array([[2, 0], [1, 1]], dtype=jnp.int32)
    coefficients = jnp.array([[3.0, -2.0]])

    gradient = jax.grad(
        lambda x: evaluate_monomial_polynomial(x, exponents, coefficients)[0]
    )(jnp.array([0.5, -0.25]))
    np.testing.assert_allclose(np.asarray(gradient), np.array([3.5, -1.0]))

    coeff_grad = jax.jacfwd(
        lambda c: collect_univariate_coefficients(jnp.array([[2], [3]], dtype=jnp.int32), c, 4)
    )(jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(
        np.asarray(coeff_grad),
        np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
    )

    solve_grad = jax.grad(
        lambda mu: jnp.sum(
            solve_scalar_graph_coefficients(-1.0, mu, jnp.array([0.0, 0.0, 1.0]))
        )
    )(-5.0)
    np.testing.assert_allclose(np.asarray(solve_grad), 1.0 / 9.0, rtol=1e-12)


def test_quadratic_vector_graph_solver_matches_hand_derived_case() -> None:
    linear_matrix = jnp.diag(jnp.array([-1.0, -4.0]))
    eigenvalue = jnp.array(-1.0)
    eigenvector = jnp.array([1.0, 0.0])

    def quadratic_term(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0, left[0] * right[0]])

    coefficients = solve_autonomous_quadratic_graph_coefficients(
        linear_matrix,
        eigenvalue,
        eigenvector,
        order=3,
        quadratic_term=quadratic_term,
    )

    expected = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(np.asarray(coefficients), expected, rtol=1e-12)


def test_quadratic_vector_graph_solver_supports_jax_grad() -> None:
    eigenvalue = jnp.array(-1.0)
    eigenvector = jnp.array([1.0, 0.0])

    def quadratic_term(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0, left[0] * right[0]])

    def loss_fn(transverse: jnp.ndarray) -> jnp.ndarray:
        linear_matrix = jnp.diag(jnp.array([-1.0, transverse]))
        coefficients = solve_autonomous_quadratic_graph_coefficients(
            linear_matrix,
            eigenvalue,
            eigenvector,
            order=2,
            quadratic_term=quadratic_term,
        )
        return jnp.sum(coefficients**2)

    gradient = jax.grad(loss_fn)(jnp.array(-3.0))

    assert gradient.shape == ()
    assert np.isfinite(np.asarray(gradient))


def test_univariate_graph_invariance_residual_matches_invariant_graph() -> None:
    coefficients = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
        ]
    )
    reduced = jnp.array([-0.2, 0.0, 0.2])

    def vector_field(state: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([-state[0], -4.0 * state[1] + state[0] ** 2])

    residual = univariate_graph_invariance_residual(
        reduced,
        eigenvalue=-1.0,
        coefficients=coefficients,
        vector_field=vector_field,
    )

    np.testing.assert_allclose(np.asarray(residual), np.zeros((3, 2)), atol=1e-14)


def test_univariate_graph_invariance_residual_supports_jax_grad() -> None:
    reduced = jnp.array([-0.1, 0.1])

    def loss_fn(transverse: jnp.ndarray) -> jnp.ndarray:
        coefficients = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0 / (transverse - 2.0)],
            ]
        )

        def vector_field(state: jnp.ndarray) -> jnp.ndarray:
            return jnp.array([-state[0], -transverse * state[1] + state[0] ** 2])

        residual = univariate_graph_invariance_residual(
            reduced,
            eigenvalue=-1.0,
            coefficients=coefficients,
            vector_field=vector_field,
        )
        return jnp.sum(residual**2)

    gradient = jax.grad(loss_fn)(jnp.array(4.0))

    assert gradient.shape == ()
    assert np.isfinite(np.asarray(gradient))
