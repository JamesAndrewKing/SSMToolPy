"""Reproduce the first bounded Lorenz1stOrder numerical checks."""

from __future__ import annotations

import jax.numpy as jnp

from lorenz import (
    lorenz_full_unstable_trajectories,
    lorenz_linear_eigenvalues,
    lorenz_rk4_trajectory,
    lorenz_ssm_invariance_residual,
    lorenz_unstable_ssm_graph_coefficients,
    lorenz_vector_field,
    standard_lorenz_parameters,
)
from ssmtoolpy.core.graph import (
    evaluate_univariate_graph,
    linear_reduced_trajectory,
    two_sided_graph_curve,
)


def main() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    state = jnp.array([1.0, 2.0, 3.0])
    values = lorenz_vector_field(state, sigma, rho, beta)
    eigenvalues = jnp.sort(jnp.real(lorenz_linear_eigenvalues(sigma, rho, beta)))
    eigenvalue, eigenvector, coefficients = lorenz_unstable_ssm_graph_coefficients(
        sigma, rho, beta, order=5
    )
    reduced = jnp.linspace(-1e-4, 1e-4, 5)
    ssm_states = evaluate_univariate_graph(reduced, coefficients)
    residual = lorenz_ssm_invariance_residual(
        reduced, eigenvalue, coefficients, sigma, rho, beta
    )
    times = jnp.linspace(0.0, 1.0, 501)
    reduced_positive = linear_reduced_trajectory(1e-4, times, eigenvalue)
    lifted_positive = evaluate_univariate_graph(reduced_positive, coefficients)
    ssm_curve = two_sided_graph_curve(times, 1e-4, eigenvalue, coefficients)
    full_trajectories = lorenz_full_unstable_trajectories(
        times, 1e-4, eigenvector, sigma, rho, beta
    )
    comparison_times = jnp.linspace(0.0, 0.05, 101)
    comparison_reduced = linear_reduced_trajectory(1e-5, comparison_times, eigenvalue)
    comparison_lifted = evaluate_univariate_graph(comparison_reduced, coefficients)
    comparison_full = lorenz_rk4_trajectory(
        comparison_lifted[0], comparison_times, sigma, rho, beta
    )
    trajectory = lorenz_rk4_trajectory(1e-4 * state, times, sigma, rho, beta)

    print("Lorenz vector field at [1, 2, 3]:")
    print(jnp.asarray(values))
    print("Sorted linear eigenvalues at the origin:")
    print(jnp.asarray(eigenvalues))
    print("Unstable SSM graph coefficients through order 5:")
    print(jnp.asarray(coefficients))
    print("Maximum invariance residual on small amplitudes:")
    print(jnp.max(jnp.abs(residual)))
    print("Sample SSM graph states:")
    print(jnp.asarray(ssm_states))
    print("Reduced-to-full trajectory final state:")
    print(jnp.asarray(lifted_positive[-1]))
    print("Two-sided SSM curve shape:")
    print(ssm_curve.shape)
    print("Two-sided full trajectory shape:")
    print(full_trajectories.shape)
    print("Short reduced/full maximum difference:")
    print(jnp.max(jnp.abs(comparison_full - comparison_lifted)))
    print("Small-amplitude RK4 trajectory final state:")
    print(jnp.asarray(trajectory[-1]))


if __name__ == "__main__":
    main()
