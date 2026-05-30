"""Small, tested Python/JAX targets ported from MATLAB SSMTool."""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from ssmtoolpy.systems.lorenz import (
    build_lorenz_system,
    lorenz_linear_eigenvalues,
    lorenz_nonlinear_coefficients,
    lorenz_nonlinear_exponents,
    lorenz_vector_field,
    standard_lorenz_parameters,
)
from ssmtoolpy.systems.planar import (
    build_planar_system,
    evaluate_planar_ssm_graph,
    planar_ssm_graph_coefficients,
    planar_vector_field,
)

__all__ = [
    "build_lorenz_system",
    "build_planar_system",
    "evaluate_planar_ssm_graph",
    "lorenz_linear_eigenvalues",
    "lorenz_nonlinear_coefficients",
    "lorenz_nonlinear_exponents",
    "lorenz_vector_field",
    "planar_ssm_graph_coefficients",
    "planar_vector_field",
    "standard_lorenz_parameters",
]
