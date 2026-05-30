from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy.core.integrators import fixed_step_rk4
from ssmtoolpy.core.trajectories import integrate_two_sided_branches


def test_fixed_step_rk4_integrates_scalar_linear_system() -> None:
    times = jnp.linspace(0.0, 0.1, 11)
    trajectory = fixed_step_rk4(lambda x: -x, jnp.array([1.0]), times)

    np.testing.assert_allclose(
        np.asarray(trajectory[:, 0]),
        np.asarray(jnp.exp(-times)),
        rtol=1e-10,
    )


def test_fixed_step_rk4_supports_jax_grad_for_fixed_times() -> None:
    times = jnp.linspace(0.0, 0.05, 6)

    def loss_fn(rate: jnp.ndarray) -> jnp.ndarray:
        trajectory = fixed_step_rk4(lambda x: rate * x, jnp.array([1.0]), times)
        return trajectory[-1, 0] ** 2

    gradient = jax.grad(loss_fn)(jnp.array(-1.0))

    assert gradient.shape == ()
    assert np.isfinite(np.asarray(gradient))


def test_integrate_two_sided_branches_concatenates_negative_then_positive() -> None:
    def trajectory(initial_state: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([initial_state, 2.0 * initial_state])

    branches = integrate_two_sided_branches(trajectory, jnp.array([3.0]))

    np.testing.assert_allclose(
        np.asarray(branches[:, 0]),
        np.array([-6.0, -3.0, 3.0, 6.0]),
    )
