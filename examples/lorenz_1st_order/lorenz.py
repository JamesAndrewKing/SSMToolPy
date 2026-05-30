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

    if order < 1:
        raise ValueError("order must be at least 1")

    linear_matrix = jnp.asarray(linear_matrix)
    eigenvector = jnp.asarray(eigenvector)
    coefficients = [jnp.zeros(3, dtype=linear_matrix.dtype), eigenvector]
    identity = jnp.eye(3, dtype=linear_matrix.dtype)

    for degree in range(2, order + 1):
        rhs = jnp.zeros(3, dtype=linear_matrix.dtype)
        for left_degree in range(1, degree):
            right_degree = degree - left_degree
            rhs = rhs + _lorenz_quadratic_term(
                coefficients[left_degree], coefficients[right_degree]
            )
        matrix = degree * eigenvalue * identity - linear_matrix
        coefficients.append(jnp.linalg.solve(matrix, rhs))

    return jnp.stack(coefficients)


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


def evaluate_lorenz_ssm_graph(reduced_coordinate: Array, coefficients: Array) -> Array:
    """Evaluate a one-dimensional Lorenz SSM graph parameterization.

    ``coefficients[k]`` stores the full-space coefficient multiplying ``p**k``.
    The result has shape ``(..., 3)`` for reduced coordinates with shape
    ``(...)``.

    Differentiability: differentiable with respect to reduced coordinates and
    coefficients for fixed coefficient length.
    """

    reduced_coordinate = jnp.asarray(reduced_coordinate)
    coefficients = jnp.asarray(coefficients)
    degrees = jnp.arange(coefficients.shape[0], dtype=coefficients.dtype)
    powers = reduced_coordinate[..., None] ** degrees
    return powers @ coefficients


def lorenz_reduced_trajectory(
    initial_reduced_coordinate: Array | float,
    times: Array,
    eigenvalue: Array | float,
) -> Array:
    """Evaluate the linear Lorenz reduced dynamics used in ``demo.mlx``.

    No inner resonances are detected in the MATLAB live script, so the reduced
    coordinate obeys ``p_dot = eigenvalue * p`` and therefore
    ``p(t) = p(0) * exp(eigenvalue * t)``.

    Differentiability: differentiable with respect to the initial reduced
    coordinate and eigenvalue for fixed ``times``.
    """

    times = jnp.asarray(times)
    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    return jnp.asarray(initial_reduced_coordinate) * jnp.exp(eigenvalue * times)


def lorenz_reduced_to_full_trajectory(
    reduced_coordinates: Array,
    coefficients: Array,
) -> Array:
    """Map a reduced Lorenz trajectory to full coordinates.

    This is the one-dimensional autonomous graph-parameterization slice of
    MATLAB ``reduced_to_full_traj.m`` used by ``Lorenz1stOrder/demo.mlx``.

    Differentiability: differentiable with respect to reduced coordinates and
    coefficients for fixed coefficient length.
    """

    return evaluate_lorenz_ssm_graph(reduced_coordinates, coefficients)


def lorenz_unstable_ssm_curve(
    times: Array,
    amplitude: Array | float,
    eigenvalue: Array | float,
    coefficients: Array,
) -> Array:
    """Return the two-sided unstable SSM curve plotted in the MATLAB live script.

    The negative branch is reversed and concatenated with the positive branch,
    matching ``z = [z2(:,end:-1:1) z1]`` in ``demo.mlx``.

    Differentiability: differentiable for fixed ``times`` and fixed coefficient
    length.
    """

    positive_reduced = lorenz_reduced_trajectory(amplitude, times, eigenvalue)
    negative_reduced = lorenz_reduced_trajectory(-amplitude, times, eigenvalue)
    positive = lorenz_reduced_to_full_trajectory(positive_reduced, coefficients)
    negative = lorenz_reduced_to_full_trajectory(negative_reduced, coefficients)
    return jnp.concatenate([negative[::-1], positive], axis=0)


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
    positive = lorenz_rk4_trajectory(initial, times, sigma, rho, beta)
    negative = lorenz_rk4_trajectory(-initial, times, sigma, rho, beta)
    return jnp.concatenate([negative[::-1], positive], axis=0)


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

    reduced_coordinate = jnp.asarray(reduced_coordinate)
    coefficients = jnp.asarray(coefficients)
    degrees = jnp.arange(coefficients.shape[0], dtype=coefficients.dtype)
    derivative_coefficients = degrees[:, None] * coefficients
    derivative = jnp.sum(
        derivative_coefficients[1:] * reduced_coordinate[..., None, None] ** (degrees[1:, None] - 1.0),
        axis=-2,
    )
    state = evaluate_lorenz_ssm_graph(reduced_coordinate, coefficients)
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]
    vector_field = jnp.stack(
        [
            sigma * (y - x),
            rho * x - y - x * z,
            -beta * z + x * y,
        ],
        axis=-1,
    )
    return derivative * (eigenvalue * reduced_coordinate)[..., None] - vector_field


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
