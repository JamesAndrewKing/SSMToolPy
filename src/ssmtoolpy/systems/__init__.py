"""Example-specific system definitions."""

from ssmtoolpy.systems.lorenz import (
    build_lorenz_system,
    evaluate_lorenz_ssm_graph,
    lorenz_linear_eigenvalues,
    lorenz_nonlinear_coefficients,
    lorenz_nonlinear_exponents,
    lorenz_rk4_trajectory,
    lorenz_ssm_invariance_residual,
    lorenz_unstable_eigenpair,
    lorenz_unstable_ssm_graph_coefficients,
    lorenz_vector_field,
    solve_lorenz_unstable_graph_coefficients,
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
    "evaluate_lorenz_ssm_graph",
    "evaluate_planar_ssm_graph",
    "lorenz_linear_eigenvalues",
    "lorenz_nonlinear_coefficients",
    "lorenz_nonlinear_exponents",
    "lorenz_rk4_trajectory",
    "lorenz_ssm_invariance_residual",
    "lorenz_unstable_eigenpair",
    "lorenz_unstable_ssm_graph_coefficients",
    "lorenz_vector_field",
    "planar_ssm_graph_coefficients",
    "planar_vector_field",
    "solve_lorenz_unstable_graph_coefficients",
    "standard_lorenz_parameters",
]
