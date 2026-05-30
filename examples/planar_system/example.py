"""Reproduce the PlanarSystem graph coefficients from MATLAB SSMTool."""

from __future__ import annotations

import jax.numpy as jnp

from planar import evaluate_planar_ssm_graph, planar_ssm_graph_coefficients


def main() -> None:
    coefficients = planar_ssm_graph_coefficients(order=8)
    reference = jnp.zeros_like(coefficients)
    reference = reference.at[2:6].set(
        jnp.array([1.0 / (jnp.sqrt(24.0) - degree) for degree in range(2, 6)])
    )
    active = coefficients[2:6]
    sample_x = jnp.array([-0.2, 0.0, 0.2])
    sample_y = evaluate_planar_ssm_graph(sample_x, coefficients)

    print("PlanarSystem active graph coefficients a2..a5:")
    print(jnp.asarray(active))
    print("Maximum absolute difference from demo formula:")
    print(jnp.max(jnp.abs(coefficients - reference)))
    print("Sample graph values:")
    print(jnp.asarray(sample_y))


if __name__ == "__main__":
    main()
