"""Reproduce the first bounded Lorenz1stOrder numerical checks."""

from __future__ import annotations

import jax.numpy as jnp

from ssmtoolpy.systems.lorenz import (
    lorenz_linear_eigenvalues,
    lorenz_vector_field,
    standard_lorenz_parameters,
)


def main() -> None:
    sigma, rho, beta = standard_lorenz_parameters()
    state = jnp.array([1.0, 2.0, 3.0])
    values = lorenz_vector_field(state, sigma, rho, beta)
    eigenvalues = jnp.sort(jnp.real(lorenz_linear_eigenvalues(sigma, rho, beta)))

    print("Lorenz vector field at [1, 2, 3]:")
    print(jnp.asarray(values))
    print("Sorted linear eigenvalues at the origin:")
    print(jnp.asarray(eigenvalues))


if __name__ == "__main__":
    main()
