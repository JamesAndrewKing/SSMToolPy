"""JAX implementation of selected SSMTool numerical kernels.

The MATLAB implementation in ``SSMTool/`` remains the reference. This
package ports low-level kernels first and documents differentiability status
for every public API.
"""

from ssmtoolpy.coefficients import (
    ConjugateOrdering,
    coeffs_conj2full,
    coeffs_conj2lex,
    coeffs_lex2revlex,
    coeffs_output,
    conjugate_flip,
    conjugate_ordering,
    number_of_multis,
)
from ssmtoolpy.frc import (
    check_stability,
    compute_fixed_points_2d,
    compute_gamma,
    frc_ab,
    frc_jacobian,
    frc_psi,
    get_contour_xy,
)
from ssmtoolpy.misc import (
    AutoReducedDynamicsData,
    OutputSummary,
    ProjectionData,
    auto_red_dyn,
    extract_output,
    nonlinear_projection_objective,
    project_to_ssm_linear,
    real_to_conjugate_state,
    reduced_to_full_traj,
    solve_invariance_equation,
    spblkdiag,
    squared_distance_to_point_ssm,
)
from ssmtoolpy.multi_index import (
    MultiIndexPolynomial,
    expand_multiindex,
    expand_multiindex_derivative,
    multi_addition,
    multi_index_2_ordering,
    multi_index_to_tensor,
    multi_nsumk,
    multi_subtraction,
    nsumk,
    sub2multiind,
    tensor_to_multi_index,
)
from ssmtoolpy.reduction import NonAutonomousTerm, reduced_to_full, reduced_to_full_complex
from ssmtoolpy.tensor import expand_tensor, expand_tensor_derivative, khatri_rao_product

__all__ = [
    "MultiIndexPolynomial",
    "AutoReducedDynamicsData",
    "ConjugateOrdering",
    "NonAutonomousTerm",
    "OutputSummary",
    "ProjectionData",
    "auto_red_dyn",
    "coeffs_conj2full",
    "coeffs_conj2lex",
    "coeffs_lex2revlex",
    "coeffs_output",
    "conjugate_flip",
    "conjugate_ordering",
    "check_stability",
    "compute_fixed_points_2d",
    "compute_gamma",
    "extract_output",
    "expand_multiindex",
    "expand_multiindex_derivative",
    "expand_tensor",
    "expand_tensor_derivative",
    "frc_ab",
    "frc_jacobian",
    "frc_psi",
    "get_contour_xy",
    "khatri_rao_product",
    "multi_addition",
    "multi_index_2_ordering",
    "multi_index_to_tensor",
    "multi_nsumk",
    "multi_subtraction",
    "nonlinear_projection_objective",
    "number_of_multis",
    "nsumk",
    "project_to_ssm_linear",
    "real_to_conjugate_state",
    "reduced_to_full",
    "reduced_to_full_complex",
    "reduced_to_full_traj",
    "solve_invariance_equation",
    "spblkdiag",
    "squared_distance_to_point_ssm",
    "sub2multiind",
    "tensor_to_multi_index",
]
