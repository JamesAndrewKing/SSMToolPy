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
| `misc/reduced_to_full_traj.m` | reconstruction kernel | `ssmtoolpy.misc.reduced_to_full_traj` | ported | differentiable for fixed structure |
| `misc/extract_output.m` | output utility | `ssmtoolpy.misc.extract_output` | ported | piecewise differentiable |
| `misc/spblkdiag.m` | linear algebra utility | `ssmtoolpy.misc.spblkdiag` | ported as dense JAX array | differentiable |
| `misc/solveinveq.m` | linear solve utility | `ssmtoolpy.misc.solve_invariance_equation` | ported with JAX direct, pseudoinverse, and sparse iterative solvers | differentiable under nondegeneracy assumptions |
| `misc/auto_red_dyn.m` | reduced dynamics kernel | `ssmtoolpy.misc.auto_red_dyn` | ported | differentiable |
| `misc/proj2SSM.m` | projection utility | `ssmtoolpy.misc.project_to_ssm_linear` and `nonlinear_projection_objective` | partially ported: linear projection and nonlinear objective only | differentiable |
| `misc/squaDist2pointSSM.m` | projection objective | `ssmtoolpy.misc.squared_distance_to_point_ssm` | ported against autonomous reconstruction API | differentiable |
| `@DynamicalSystem/evaluate_Fnl.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.first_order_nonlinearity` and `first_order_from_second_order_nonlinearity` | functional core ported | differentiable for intrusive terms |
| `@DynamicalSystem/compute_fnl.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_internal_force` | functional core ported | differentiable for intrusive terms |
| `@DynamicalSystem/compute_dfnldx.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_internal_force_jacobian_x` | functional core ported | differentiable for intrusive terms |
| `@DynamicalSystem/compute_dfnldxd.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_internal_force_jacobian_xd` | functional displacement-only behavior ported | differentiable |
| `@DynamicalSystem/residual.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_residual` | functional core ported | piecewise differentiable |
| `@DynamicalSystem/odefun.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.evaluate_first_order_vector_field` | functional core ported | differentiable under nonsingular `B` |
| `@DynamicalSystem/evaluate_Fext.m` and `compute_fext.m` | forcing evaluation | `ssmtoolpy.dynamical_system.evaluate_periodic_forcing` | functional periodic forcing core ported | differentiable for fixed forcing structure |
| `@DynamicalSystem/private/get_BinvA.m` | mechanical first-order conversion | `ssmtoolpy.dynamical_system.mechanical_binv_a` | ported | differentiable under nonsingular mass matrix |
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
| `@Manifold/private/coeffs_composition.m` | manifold coefficient algebra | `ssmtoolpy.manifold.coeffs_composition` | lex/revlex branches ported | differentiable for fixed index structure |
| `@Manifold/private/coeffs_mixed_terms.m` | manifold coefficient algebra | `ssmtoolpy.manifold.coeffs_mixed_terms` | lex/revlex branches ported | differentiable for fixed index structure |

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
- `solve_invariance_equation` uses `jax.scipy.sparse.linalg` for iterative
  solver names. JAX provides `cg`, `bicgstab`, and `gmres`; MATLAB names without
  exact JAX equivalents are mapped conservatively (`bicg`/`cgs` to `bicgstab`,
  `lsqr` to `gmres`).
- `proj2SSM` nonlinear optimization is not ported as an optimizer. The
  differentiable objective is exposed for use with a future JAX optimizer.
- MATLAB `squaDist2pointSSM.m` calls `reduced_to_full(x,W_0)` although the local
  MATLAB `reduced_to_full` signature expects non-autonomous arguments too. The
  Python port uses the intended autonomous reconstruction behavior.
- The full mutable MATLAB `DynamicalSystem` class surface is not ported yet.
  This pass adds functional JAX kernels for nonlinear force, forcing, residual,
  and ODE RHS evaluation.
- `coeffs_composition` and `coeffs_mixed_terms` currently cover lexicographic
  and reverse-lexicographic computation. The conjugate-order branches are tied
  to the full cohomological solver and remain unported.
- MATLAB R2024b spot fixtures have been generated for selected low-level
  helpers; broad fixture generation for full examples remains outstanding.
