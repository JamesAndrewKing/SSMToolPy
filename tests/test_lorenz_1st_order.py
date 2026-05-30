from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy.systems.lorenz import (
    build_lorenz_system,
    lorenz_linear_eigenvalues,
    lorenz_nonlinear_coefficients,
    lorenz_nonlinear_exponents,
    lorenz_rk4_trajectory,
    lorenz_vector_field,
    standard_lorenz_parameters,
)


def test_lorenz_build_model_matches_matlab_source() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    a, b = build_lorenz_system(sigma, rho, beta)

    np.testing.assert_allclose(
        np.asarray(a),
        np.array([[-10.0, 10.0, 0.0], [28.0, -1.0, 0.0], [0.0, 0.0, -8.0 / 3.0]]),
    )
    np.testing.assert_allclose(np.asarray(b), np.eye(3))


def test_lorenz_sparse_nonlinear_terms_match_build_model() -> None:
    np.testing.assert_array_equal(
        np.asarray(lorenz_nonlinear_exponents()),
        np.array([[1, 0, 1], [1, 1, 0]]),
    )
    np.testing.assert_allclose(
        np.asarray(lorenz_nonlinear_coefficients()),
        np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
    )


def test_lorenz_vector_field_matches_matlab_lorenz_m() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    state = jnp.array([1.25, -0.5, 2.0])
    expected = jnp.array(
        [
            sigma * (state[1] - state[0]),
            rho * state[0] - state[1] - state[0] * state[2],
            -beta * state[2] + state[0] * state[1],
        ]
    )

    np.testing.assert_allclose(
        np.asarray(lorenz_vector_field(state, sigma, rho, beta)),
        np.asarray(expected),
        rtol=1e-12,
    )


def test_lorenz_linear_eigenvalues_match_demo_values() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    eigenvalues = jnp.sort(jnp.real(lorenz_linear_eigenvalues(sigma, rho, beta)))

    np.testing.assert_allclose(
        np.asarray(eigenvalues),
        np.array([-22.82772345, -8.0 / 3.0, 11.82772345]),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(eigenvalues),
        np.array([-22.828, -2.667, 11.828]),
        atol=5e-4,
    )


def test_lorenz_vector_field_supports_jax_transforms() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    state = jnp.array([0.2, -0.1, 0.3])
    jacobian = jax.jacfwd(lambda z: lorenz_vector_field(z, sigma, rho, beta))(state)

    np.testing.assert_allclose(
        np.asarray(jacobian),
        np.array(
            [
                [-sigma, sigma, 0.0],
                [rho - state[2], -1.0, -state[0]],
                [state[1], state[0], -beta],
            ]
        ),
        rtol=1e-12,
    )


def test_lorenz_parameter_to_output_loss_gradient_smoke() -> None:
    sigma = jnp.array(10.0)
    beta = jnp.array(8.0 / 3.0)
    states = jnp.array([[0.2, -0.1, 0.3], [1.0, 0.5, -0.25]])
    target = jnp.array([[0.1, 0.2, -0.3], [0.0, -0.5, 0.25]])

    def loss_fn(rho: jnp.ndarray) -> jnp.ndarray:
        predictions = jax.vmap(lambda z: lorenz_vector_field(z, sigma, rho, beta))(states)
        return jnp.mean((predictions - target) ** 2)

    rho = jnp.array(28.0)
    loss = loss_fn(rho)
    gradient = jax.grad(loss_fn)(rho)

    assert loss.shape == ()
    assert gradient.shape == ()
    assert np.isfinite(np.asarray(loss))
    assert np.isfinite(np.asarray(gradient))

    eps = 1e-6
    finite_difference = (loss_fn(rho + eps) - loss_fn(rho - eps)) / (2.0 * eps)
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(finite_difference), rtol=1e-6)


def test_lorenz_rk4_trajectory_shape_and_initial_state() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    initial = jnp.array([1.0, 2.0, 3.0])
    times = jnp.linspace(0.0, 0.02, 5)
    trajectory = lorenz_rk4_trajectory(initial, times, sigma, rho, beta)

    assert trajectory.shape == (5, 3)
    np.testing.assert_allclose(np.asarray(trajectory[0]), np.asarray(initial), rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(np.asarray(trajectory)))


def test_lorenz_rk4_one_small_step_matches_taylor_reference() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    initial = jnp.array([1.0, 2.0, 3.0])
    h = 1e-5
    trajectory = lorenz_rk4_trajectory(initial, jnp.array([0.0, h]), sigma, rho, beta)
    expected = initial + h * lorenz_vector_field(initial, sigma, rho, beta)

    np.testing.assert_allclose(np.asarray(trajectory[-1]), np.asarray(expected), rtol=0.0, atol=2e-8)


def test_lorenz_rk4_trajectory_supports_jax_grad_for_fixed_times() -> None:
    sigma = jnp.array(10.0)
    rho = jnp.array(28.0)
    beta = jnp.array(8.0 / 3.0)
    initial = jnp.array([0.1, 0.2, 0.3])
    times = jnp.linspace(0.0, 0.01, 4)

    def loss_fn(rho_value: jnp.ndarray) -> jnp.ndarray:
        trajectory = lorenz_rk4_trajectory(initial, times, sigma, rho_value, beta)
        return jnp.sum(trajectory[-1] ** 2)

    gradient = jax.grad(loss_fn)(rho)

    assert gradient.shape == ()
    assert np.isfinite(np.asarray(gradient))
