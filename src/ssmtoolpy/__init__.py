"""Reusable numerical kernels for the clean-room SSMTool Python/JAX port."""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from ssmtoolpy.core import (
    collect_univariate_coefficients,
    evaluate_monomial_polynomial,
    multiindices_of_total_degree,
    solve_scalar_graph_coefficients,
)

__all__ = [
    "collect_univariate_coefficients",
    "evaluate_monomial_polynomial",
    "multiindices_of_total_degree",
    "solve_scalar_graph_coefficients",
]
