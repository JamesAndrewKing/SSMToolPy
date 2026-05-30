"""Reproduce the first bounded Lorenz1stOrder numerical checks."""

from __future__ import annotations

import jax.numpy as jnp

from ssmtoolpy.systems.lorenz import (
    evaluate_lorenz_ssm_graph,
    lorenz_linear_eigenvalues,
    lorenz_rk4_trajectory,
    lorenz_ssm_invariance_residual,
    lorenz_unstable_ssm_graph_coefficients,
    lorenz_vector_field,
    standard_lorenz_parameters,
)


def main() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    state = jnp.array([1.0, 2.0, 3.0])
    values = lorenz_vector_field(state, sigma, rho, beta)
    eigenvalues = jnp.sort(jnp.real(lorenz_linear_eigenvalues(sigma, rho, beta)))
    eigenvalue, _, coefficients = lorenz_unstable_ssm_graph_coefficients(
        sigma, rho, beta, order=3
    )
    reduced = jnp.linspace(-1e-4, 1e-4, 5)
    ssm_states = evaluate_lorenz_ssm_graph(reduced, coefficients)
    residual = lorenz_ssm_invariance_residual(
        reduced, eigenvalue, coefficients, sigma, rho, beta
    )
    times = jnp.linspace(0.0, 0.1, 51)
    trajectory = lorenz_rk4_trajectory(1e-4 * state, times, sigma, rho, beta)

    print("Lorenz vector field at [1, 2, 3]:")
    print(jnp.asarray(values))
    print("Sorted linear eigenvalues at the origin:")
    print(jnp.asarray(eigenvalues))
    print("Unstable SSM graph coefficients through order 3:")
    print(jnp.asarray(coefficients))
    print("Maximum invariance residual on small amplitudes:")
    print(jnp.max(jnp.abs(residual)))
    print("Sample SSM graph states:")
    print(jnp.asarray(ssm_states))
    print("Small-amplitude RK4 trajectory final state:")
    print(jnp.asarray(trajectory[-1]))


if __name__ == "__main__":
    main()
