# Migration Report

## Scope

`SSMTool/` is treated as immutable MATLAB reference code. This Python package
starts with low-level kernels from `SSMTool/src/misc`, `SSMTool/src/frc`, and
selected private multi-index helpers used by manifold coefficient assembly.

Bundled third-party MATLAB code in `SSMTool/ext` is external dependency code and
is not a Python migration target for this package.

## MATLAB-to-Python API Mapping

| MATLAB file | Category | Python destination | Status | Differentiability |
| --- | --- | --- | --- | --- |
| `misc/nsumk.m` | core utility | `ssmtoolpy.multi_index.nsumk` | ported | not differentiable |
| `misc/sub2multiind.m` | core utility | `ssmtoolpy.multi_index.sub2multiind` | ported | not differentiable |
| `misc/expand_multiindex.m` | core kernel | `ssmtoolpy.multi_index.expand_multiindex` | ported | differentiable |
| `misc/expand_multiindex_derivative.m` | core kernel | `ssmtoolpy.multi_index.expand_multiindex_derivative` | ported | differentiable |
| `misc/tensor_to_multi_index.m` | conversion utility | `ssmtoolpy.multi_index.tensor_to_multi_index` | ported for dense tensors | not differentiable |
| `misc/multi_index_to_tensor.m` | conversion utility | `ssmtoolpy.multi_index.multi_index_to_tensor` | ported for dense tensors | not differentiable |
| `misc/khatri_rao_product.m` | core kernel | `ssmtoolpy.tensor.khatri_rao_product` | ported | differentiable |
| `misc/expand_tensor.m` | core kernel | `ssmtoolpy.tensor.expand_tensor` | ported for dense tensors | differentiable |
| `misc/expand_tensor_derivative.m` | core kernel | `ssmtoolpy.tensor.expand_tensor_derivative` | ported for dense tensors | differentiable |
| `misc/reduced_to_full.m` | reconstruction kernel | `ssmtoolpy.reduction.reduced_to_full` | partially ported | differentiable for fixed structure |
| `misc/reduced_to_full_complex.m` | reconstruction kernel | `ssmtoolpy.reduction.reduced_to_full_complex` | partially ported | not yet verified |
| `frc/frc_ab.m` and `misc/frc_ab.m` | FRC kernel | `ssmtoolpy.frc.frc_ab` | ported | differentiable |
| `frc/compute_gamma.m` | FRC utility | `ssmtoolpy.frc.compute_gamma` | ported | not differentiable |
| `frc/frc_psi.m` | FRC kernel | `ssmtoolpy.frc.frc_psi` | ported | piecewise differentiable |
| `frc/frc_Jacobian.m` | FRC kernel | `ssmtoolpy.frc.frc_jacobian` | ported | differentiable for `rho != 0` |
| `frc/check_stability.m` | FRC utility | `ssmtoolpy.frc.check_stability` | ported | not differentiable |
| `frc/get_contour_xy.m` | FRC utility | `ssmtoolpy.frc.get_contour_xy` | ported | not differentiable |
| `frc/compute_fixed_points_2D.m` | FRC utility | `ssmtoolpy.frc.compute_fixed_points_2d` | ported with lightweight grid-contour intersection | not differentiable |
| `@Manifold/private/multi_addition.m` | core utility | `ssmtoolpy.multi_index.multi_addition` | ported | not differentiable |
| `@Manifold/private/multi_subtraction.m` | core utility | `ssmtoolpy.multi_index.multi_subtraction` | ported | not differentiable |
| `@Manifold/private/multi_index_2_ordering.m` | core utility | `ssmtoolpy.multi_index.multi_index_2_ordering` | ported | not differentiable |
| `@Manifold/private/multi_nsumk.m` | core utility | `ssmtoolpy.multi_index.multi_nsumk` | ported | not differentiable |
| `@Manifold/private/coeffs_conj2full.m` | coefficient ordering utility | `ssmtoolpy.coefficients.coeffs_conj2full` | ported | differentiable for fixed index arrays |
| `@Manifold/private/coeffs_conj2lex.m` | coefficient ordering utility | `ssmtoolpy.coefficients.coeffs_conj2lex` | ported | differentiable for fixed ordering data |
| `@Manifold/private/coeffs_lex2revlex.m` | coefficient ordering utility | `ssmtoolpy.coefficients.coeffs_lex2revlex` | ported | differentiable for fixed structures |
| `@Manifold/private/coeffs_output.m` | coefficient output utility | `ssmtoolpy.coefficients.coeffs_output` | ported | not differentiable |
| nested `coeffs_setup.m/conjugate_ordering` | coefficient ordering utility | `ssmtoolpy.coefficients.conjugate_ordering` | ported | not differentiable |

## Source Inventory Summary

`SSMTool/src` contains 230 MATLAB files:

- `@DynamicalSystem`: high-level dynamical-system model class, forcing,
  nonlinear force evaluation, and spectral analysis.
- `@Manifold`: autonomous/non-autonomous SSM coefficient assembly, invariance
  equations, multi-index bookkeeping, and output helpers.
- `@SSM`: continuation, FRC/FRS extraction, stability diagrams, COCO wrappers,
  and plotting/output code.
- `frc`: compact FRC algebra and fixed-point helpers.
- `misc`: low-level polynomial/tensor kernels, reconstruction helpers, plotting,
  solution readers, and numerical utilities.
- option classes: `DSOptions.m`, `ManifoldOptions.m`, `FRCOptions.m`,
  `FRSOptions.m`.

The complete per-file inventory is in `docs/migration_inventory.md`.

## Not Yet Ported

Most high-level SSM workflows remain unported: class APIs, spectral subspace
selection, cohomological equation solvers, intrusive/semi-intrusive force
assembly, COCO continuation wrappers, plotting, solution readers, examples, and
large finite-element model builders.

Known blockers and design work:

- MATLAB sparse `sptensor` support needs a Python representation or dense/sparse
  split API.
- Eigenvector sorting, nullspaces, rank decisions, resonant-mode detection, and
  continuation/event routines require explicit nondegeneracy assumptions before
  differentiability can be claimed.
- `compute_fixed_points_2d` uses an internal marching-squares style locator
  instead of MATLAB's `contourc`/`polyxpoly`; tests cover representative grid
  intersections, but dense contour parity is not yet exhaustively validated.
- MATLAB R2024b spot fixtures have been generated for selected low-level
  helpers; broad fixture generation for full examples remains outstanding.
