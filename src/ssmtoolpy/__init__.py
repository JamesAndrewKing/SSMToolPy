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
from ssmtoolpy.frc import frc_ab
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
from ssmtoolpy.reduction import reduced_to_full, reduced_to_full_complex
from ssmtoolpy.tensor import expand_tensor, expand_tensor_derivative, khatri_rao_product

__all__ = [
    "MultiIndexPolynomial",
    "ConjugateOrdering",
    "coeffs_conj2full",
    "coeffs_conj2lex",
    "coeffs_lex2revlex",
    "coeffs_output",
    "conjugate_flip",
    "conjugate_ordering",
    "expand_multiindex",
    "expand_multiindex_derivative",
    "expand_tensor",
    "expand_tensor_derivative",
    "frc_ab",
    "khatri_rao_product",
    "multi_addition",
    "multi_index_2_ordering",
    "multi_index_to_tensor",
    "multi_nsumk",
    "multi_subtraction",
    "number_of_multis",
    "nsumk",
    "reduced_to_full",
    "reduced_to_full_complex",
    "sub2multiind",
    "tensor_to_multi_index",
]
