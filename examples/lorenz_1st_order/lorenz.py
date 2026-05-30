"""Lorenz first-order example from MATLAB SSMTool.

MATLAB reference files:
- ``SSMTool/examples/Lorenz1stOrder/build_model.m``
- ``SSMTool/examples/Lorenz1stOrder/lorenz.m``
- ``SSMTool/examples/Lorenz1stOrder/demo.mlx``
"""

from __future__ import annotations

import jax.numpy as jnp

from ssmtoolpy.core.graph import evaluate_univariate_graph
from ssmtoolpy.core.integrators import fixed_step_rk4
from ssmtoolpy.core.invariance import (
    solve_autonomous_quadratic_graph_coefficients,
    univariate_graph_invariance_residual,
)
from ssmtoolpy.core.polynomial import evaluate_monomial_polynomial
from ssmtoolpy.core.trajectories import integrate_two_sided_branches


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


def lorenz_unstable_eigenpair(
    sigma: Array | float, rho: Array | float, beta: Array | float
) -> tuple[Array, Array]:
    """Return a deterministically normalized unstable eigenpair.

    The Lorenz live script selects the one-dimensional unstable modal subspace.
    This helper chooses the eigenvalue with largest real part and normalizes the
    right eigenvector to unit Euclidean norm with positive first component.

    Differentiability: not differentiable. Eigenvalue sorting/selection and
    sign normalization are setup choices.
    """

    a, _ = build_lorenz_system(sigma, rho, beta)
    eigenvalues, eigenvectors = jnp.linalg.eig(a)
    index = jnp.argmax(jnp.real(eigenvalues))
    eigenvalue = jnp.real(eigenvalues[index])
    eigenvector = jnp.real(eigenvectors[:, index])
    eigenvector = eigenvector / jnp.linalg.norm(eigenvector)
    sign = jnp.where(eigenvector[0] < 0.0, -1.0, 1.0)
    return eigenvalue, sign * eigenvector


def _lorenz_quadratic_term(left: Array, right: Array) -> Array:
    return jnp.array(
        [
            0.0,
            -left[0] * right[2],
            left[0] * right[1],
        ]
    )


def solve_lorenz_unstable_graph_coefficients(
    linear_matrix: Array,
    eigenvalue: Array | float,
    eigenvector: Array,
    order: int,
) -> Array:
    """Solve fixed-choice Lorenz unstable SSM graph coefficients.

    The parameterization is ``W(p) = sum_k W[k] p**k`` with fixed reduced
    dynamics ``p_dot = eigenvalue * p``. Coefficients solve the nonresonant
    first-order homological equations
    ``(k * eigenvalue * I - A) W[k] = RHS[k]``.

    The returned array has shape ``(order + 1, 3)`` with zero constant term and
    ``W[1]`` equal to the supplied eigenvector.

    Differentiability: differentiable under nondegeneracy assumptions for fixed
    eigenvalue, eigenvector, order, and nonresonant solve structure.
    """

    return solve_autonomous_quadratic_graph_coefficients(
        linear_matrix, eigenvalue, eigenvector, order, _lorenz_quadratic_term
    )


def lorenz_unstable_ssm_graph_coefficients(
    sigma: Array | float, rho: Array | float, beta: Array | float, order: int
) -> tuple[Array, Array, Array]:
    """Return fixed-choice unstable Lorenz graph coefficients.

    This combines the setup-only unstable eigenpair selection with the
    nonresonant coefficient solve used for the first Lorenz SSM graph
    subproblem.

    Differentiability: not differentiable as a whole because eigenpair
    selection and normalization are setup choices. The underlying fixed-choice
    solve is differentiable under nondegeneracy assumptions.
    """

    a, _ = build_lorenz_system(sigma, rho, beta)
    eigenvalue, eigenvector = lorenz_unstable_eigenpair(sigma, rho, beta)
    coefficients = solve_lorenz_unstable_graph_coefficients(
        a, eigenvalue, eigenvector, order
    )
    return eigenvalue, eigenvector, coefficients


def lorenz_full_unstable_trajectories(
    times: Array,
    amplitude: Array | float,
    eigenvector: Array,
    sigma: Array | float,
    rho: Array | float,
    beta: Array | float,
) -> Array:
    """Return the two-sided full Lorenz trajectories used for validation.

    This mirrors the MATLAB live script's forward integrations from
    ``V(:,1) * amplitude`` and its negative, then reverses the negative branch
    before concatenation.

    Differentiability: differentiable with respect to amplitude, eigenvector,
    and continuous parameters for fixed ``times``.
    """

    initial = jnp.asarray(amplitude) * jnp.asarray(eigenvector)

    def trajectory(initial_state: Array) -> Array:
        return lorenz_rk4_trajectory(initial_state, times, sigma, rho, beta)

    return integrate_two_sided_branches(trajectory, initial)


def lorenz_ssm_invariance_residual(
    reduced_coordinate: Array,
    eigenvalue: Array | float,
    coefficients: Array,
    sigma: Array | float,
    rho: Array | float,
    beta: Array | float,
) -> Array:
    """Evaluate the autonomous invariance residual for the Lorenz graph.

    The residual is ``DW(p) * eigenvalue * p - f(W(p))``.

    Differentiability: differentiable for fixed reduced dynamics and fixed
    coefficient length.
    """

    return univariate_graph_invariance_residual(
        reduced_coordinate,
        eigenvalue,
        coefficients,
        lambda state: lorenz_vector_field(state, sigma, rho, beta),
    )


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
    if initial_state.shape != (3,):
        raise ValueError("initial_state must have shape (3,)")

    def vector_field(state: Array) -> Array:
        return lorenz_vector_field(state, sigma, rho, beta)

    return fixed_step_rk4(vector_field, initial_state, times)
