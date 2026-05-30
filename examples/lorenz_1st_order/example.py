"""Reproduce the first bounded Lorenz1stOrder numerical checks."""

from __future__ import annotations

import jax.numpy as jnp

from ssmtoolpy.systems.lorenz import (
    lorenz_linear_eigenvalues,
    lorenz_rk4_trajectory,
    lorenz_vector_field,
    standard_lorenz_parameters,
)


def main() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    state = jnp.array([1.0, 2.0, 3.0])
    values = lorenz_vector_field(state, sigma, rho, beta)
    eigenvalues = jnp.sort(jnp.real(lorenz_linear_eigenvalues(sigma, rho, beta)))
    times = jnp.linspace(0.0, 0.1, 51)
    trajectory = lorenz_rk4_trajectory(1e-4 * state, times, sigma, rho, beta)

    print("Lorenz vector field at [1, 2, 3]:")
    print(jnp.asarray(values))
    print("Sorted linear eigenvalues at the origin:")
    print(jnp.asarray(eigenvalues))
    print("Small-amplitude RK4 trajectory final state:")
    print(jnp.asarray(trajectory[-1]))


if __name__ == "__main__":
    main()
