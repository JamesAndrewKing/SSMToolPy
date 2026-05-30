"""Lorenz first-order example from MATLAB SSMTool.

MATLAB reference files:
- ``SSMTool/examples/Lorenz1stOrder/build_model.m``
- ``SSMTool/examples/Lorenz1stOrder/lorenz.m``
- ``SSMTool/examples/Lorenz1stOrder/demo.mlx``
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from ssmtoolpy.core.polynomial import evaluate_monomial_polynomial


Array = jnp.ndarray


def standard_lorenz_parameters() -> tuple[float, float, float]:
    """Return the standard parameters used in ``Lorenz1stOrder/demo.mlx``.

    Differentiability: not differentiable. This returns Python constants.
    """

    return 10.0, 28.0, 8.0 / 3.0


def build_lorenz_system(
    sigma: Array | float, rho: Array | float, beta: Array | float
) -> tuple[Array, Array]:
    """Return the linear matrices for the Lorenz first-order SSMTool model.

    The MATLAB model uses ``B z_dot = A z + F(z)`` with ``B = I`` and
    ``A = [[-sigma, sigma, 0], [rho, -1, 0], [0, 0, -beta]]``.

    Differentiability: differentiable with respect to ``sigma``, ``rho``, and
    ``beta``.
    """

    a = jnp.array(
        [
            [-sigma, sigma, 0.0],
            [rho, -1.0, 0.0],
            [0.0, 0.0, -beta],
        ]
    )
    b = jnp.eye(3, dtype=a.dtype)
    return a, b


def lorenz_nonlinear_exponents() -> Array:
    """Return sparse nonlinear exponents from MATLAB ``build_model.m``.

    The terms are ``xz`` and ``xy`` for state ``[x, y, z]``.

    Differentiability: not differentiable. This returns integer exponents.
    """

    return jnp.array([[1, 0, 1], [1, 1, 0]], dtype=jnp.int32)


def lorenz_nonlinear_coefficients() -> Array:
    """Return sparse nonlinear coefficients from MATLAB ``build_model.m``.

    The shape is ``(outputs, terms)`` and represents
    ``[0, -xz, xy]``.

    Differentiability: differentiable. The returned constants are JAX arrays.
    """

    return jnp.array(
        [
            [0.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ]
    )


def lorenz_vector_field(
    z: Array, sigma: Array | float, rho: Array | float, beta: Array | float
) -> Array:
    """Evaluate the Lorenz vector field from MATLAB ``lorenz.m``.

    For ``z = [x, y, z]`` this returns
    ``[sigma * (y - x), rho*x - y - x*z, -beta*z + x*y]``.

    Differentiability: differentiable with respect to the state and continuous
    parameters.
    """

    z = jnp.asarray(z)
    a, _ = build_lorenz_system(sigma, rho, beta)
    nonlinear = evaluate_monomial_polynomial(
        z, lorenz_nonlinear_exponents(), lorenz_nonlinear_coefficients()
    )
    return a @ z + nonlinear


def lorenz_linear_eigenvalues(
    sigma: Array | float, rho: Array | float, beta: Array | float
) -> Array:
    """Return eigenvalues of the Lorenz linearization at the origin.

    Differentiability: differentiable under nondegeneracy assumptions. The
    eigenvalues must remain simple and separated for stable derivatives.
    """

    a, _ = build_lorenz_system(sigma, rho, beta)
    return jnp.linalg.eigvals(a)


def lorenz_rk4_trajectory(
    initial_state: Array,
    times: Array,
    sigma: Array | float,
    rho: Array | float,
    beta: Array | float,
) -> Array:
    """Integrate the Lorenz vector field on prescribed times with RK4 steps.

    This is a deterministic fixed-step trajectory helper for reproducing the
    simulation portion of ``Lorenz1stOrder/demo.mlx``. The first row of the
    returned array is ``initial_state`` and row ``i`` corresponds to
    ``times[i]``.

    Differentiability: differentiable with respect to the initial state and
    continuous parameters for fixed ``times``.
    """

    initial_state = jnp.asarray(initial_state)
    times = jnp.asarray(times)

    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    if initial_state.shape != (3,):
        raise ValueError("initial_state must have shape (3,)")

    def step(state: Array, time_pair: Array) -> tuple[Array, Array]:
        t0, t1 = time_pair
        h = t1 - t0
        k1 = lorenz_vector_field(state, sigma, rho, beta)
        k2 = lorenz_vector_field(state + 0.5 * h * k1, sigma, rho, beta)
        k3 = lorenz_vector_field(state + 0.5 * h * k2, sigma, rho, beta)
        k4 = lorenz_vector_field(state + h * k3, sigma, rho, beta)
        next_state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_state, next_state

    pairs = jnp.stack([times[:-1], times[1:]], axis=1)
    _, tail = lax.scan(step, initial_state, pairs)
    return jnp.concatenate([initial_state[None, :], tail], axis=0)
