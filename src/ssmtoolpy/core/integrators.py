"""Small JAX-compatible time integration kernels."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import lax


Array = jnp.ndarray


def fixed_step_rk4(
    vector_field: Callable[[Array], Array],
    initial_state: Array,
    times: Array,
) -> Array:
    """Integrate a vector field on prescribed times with RK4 steps.

    The first row of the returned array is ``initial_state`` and row ``i``
    corresponds to ``times[i]``. The supplied ``vector_field`` must be a pure
    function of the state; close over fixed parameters in the caller.

    Differentiability: differentiable with respect to the initial state and
    any continuous closed-over parameters for fixed ``times`` and fixed vector
    field structure.
    """

    initial_state = jnp.asarray(initial_state)
    times = jnp.asarray(times)

    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")

    def step(state: Array, time_pair: Array) -> tuple[Array, Array]:
        t0, t1 = time_pair
        h = t1 - t0
        k1 = vector_field(state)
        k2 = vector_field(state + 0.5 * h * k1)
        k3 = vector_field(state + 0.5 * h * k2)
        k4 = vector_field(state + h * k3)
        next_state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_state, next_state

    pairs = jnp.stack([times[:-1], times[1:]], axis=1)
    _, tail = lax.scan(step, initial_state, pairs)
    return jnp.concatenate([initial_state[None, :], tail], axis=0)
