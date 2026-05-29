"""Small, tested Python/JAX targets ported from MATLAB SSMTool."""

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from ssmtoolpy.systems.planar import (
    build_planar_system,
    evaluate_planar_ssm_graph,
    planar_ssm_graph_coefficients,
    planar_vector_field,
)

__all__ = [
    "build_planar_system",
    "evaluate_planar_ssm_graph",
    "planar_ssm_graph_coefficients",
    "planar_vector_field",
]
