"""Small numerical kernels used by reproduced SSMTool examples."""

from ssmtoolpy.core.graph import (
    evaluate_graph_trajectory,
    evaluate_univariate_graph,
    linear_reduced_trajectory,
    two_sided_graph_curve,
)
from ssmtoolpy.core.integrators import fixed_step_rk4
from ssmtoolpy.core.invariance import (
    solve_autonomous_quadratic_graph_coefficients,
    solve_scalar_graph_coefficients,
)
from ssmtoolpy.core.multiindex import multiindices_of_total_degree
from ssmtoolpy.core.polynomial import (
    collect_univariate_coefficients,
    evaluate_monomial_polynomial,
)
from ssmtoolpy.core.trajectories import integrate_two_sided_branches

__all__ = [
    "collect_univariate_coefficients",
    "evaluate_graph_trajectory",
    "evaluate_monomial_polynomial",
    "evaluate_univariate_graph",
    "fixed_step_rk4",
    "integrate_two_sided_branches",
    "linear_reduced_trajectory",
    "multiindices_of_total_degree",
    "solve_autonomous_quadratic_graph_coefficients",
    "solve_scalar_graph_coefficients",
    "two_sided_graph_curve",
]
