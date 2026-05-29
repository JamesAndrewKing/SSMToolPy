"""Reproduce the BenchmarkSSM1stOrder workflow using the PlanarSystem core."""

from __future__ import annotations

import jax.numpy as jnp

from ssmtoolpy.systems.planar import planar_ssm_graph_coefficients


def main() -> None:
    coefficients = planar_ssm_graph_coefficients(order=8)
    reference = jnp.array([1.0 / (jnp.sqrt(24.0) - degree) for degree in range(2, 6)])
    difference = coefficients[2:6] - reference

    print("BenchamrkSSM1stOrder coefficients a2..a5:")
    print(jnp.asarray(coefficients[2:6]))
    print("Difference from analytical coefficients:")
    print(jnp.asarray(difference))


if __name__ == "__main__":
    main()
