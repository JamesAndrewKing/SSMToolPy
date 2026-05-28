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
| `misc/tensor_composition.m` | tensor composition kernel | `ssmtoolpy.tensor.tensor_composition` and `tensor_product` | ported for dense tensors and JAX `BCOO` sparse tensors | differentiable for fixed dense pattern; sparse path supports forward-mode for fixed sparsity |
| `misc/reduced_to_full.m` | reconstruction kernel | `ssmtoolpy.reduction.reduced_to_full` | partially ported | differentiable for fixed structure |
| `misc/reduced_to_full_complex.m` | reconstruction kernel | `ssmtoolpy.reduction.reduced_to_full_complex` | partially ported | not yet verified |
| `misc/reduced_to_full_traj.m` | reconstruction kernel | `ssmtoolpy.misc.reduced_to_full_traj` | ported | differentiable for fixed structure |
| `misc/extract_output.m` | output utility | `ssmtoolpy.misc.extract_output` | ported | piecewise differentiable |
| `misc/spblkdiag.m` | linear algebra utility | `ssmtoolpy.misc.spblkdiag` | ported as dense JAX array | differentiable |
| `misc/solveinveq.m` | linear solve utility | `ssmtoolpy.misc.solve_invariance_equation` | ported with JAX direct, pseudoinverse, and sparse iterative solvers | differentiable under nondegeneracy assumptions |
| `misc/linear_response.m` | linear response utility | `ssmtoolpy.misc.first_order_linear_response` and `second_order_linear_response` | functional first- and second-order branches ported | differentiable under nonsingular frequency-domain operators; amplitudes are piecewise differentiable |
| `misc/StEP.m` | polarization utility | `ssmtoolpy.manifold.step_polynomial` | ported for orders 1-3 | differentiable if supplied callable is differentiable |
| `misc/auto_red_dyn.m` | reduced dynamics kernel | `ssmtoolpy.misc.auto_red_dyn` | ported | differentiable |
| shared `R_0` assembly in `transient_traj_on_auto_ssm.m` and `reduced_dynamics_symbolic.m` | reduced dynamics utility | `ssmtoolpy.misc.assemble_auto_reduced_dynamics` | ported | differentiable for fixed polynomial structure |
| `misc/transient_traj_on_auto_ssm.m` | reduced dynamics trajectory utility | `ssmtoolpy.misc.transient_traj_on_auto_ssm` | functional autonomous branch ported with fixed-step RK4 | differentiable for fixed step count and structures |
| `misc/reduced_dynamics_symbolic.m` | reduced dynamics documentation utility | `ssmtoolpy.misc.reduced_dynamics_symbolic` | autonomous polar symbolic rendering ported | not differentiable |
| `misc/proj2SSM.m` | projection utility | `ssmtoolpy.misc.project_to_ssm_linear` and `nonlinear_projection_objective` | partially ported: linear projection and nonlinear objective only | differentiable |
| `misc/squaDist2pointSSM.m` | projection objective | `ssmtoolpy.misc.squared_distance_to_point_ssm` | ported against autonomous reconstruction API | differentiable |
| `@DynamicalSystem/evaluate_Fnl.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.first_order_nonlinearity` and `first_order_from_second_order_nonlinearity` | functional core ported | differentiable for intrusive terms |
| `@DynamicalSystem/compute_fnl.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_internal_force` | functional core ported | differentiable for intrusive terms |
| `@DynamicalSystem/compute_dfnldx.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_internal_force_jacobian_x` | functional core ported | differentiable for intrusive terms |
| `@DynamicalSystem/compute_dfnldxd.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_internal_force_jacobian_xd` | functional displacement-only behavior ported | differentiable |
| `@DynamicalSystem/residual.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_residual` | functional core ported | piecewise differentiable |
| `@DynamicalSystem/odefun.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.evaluate_first_order_vector_field` | functional core ported | differentiable under nonsingular `B` |
| `@DynamicalSystem/evaluate_Fext.m` and `compute_fext.m` | forcing evaluation | `ssmtoolpy.dynamical_system.evaluate_periodic_forcing` | functional periodic forcing core ported | differentiable for fixed forcing structure |
| `@DynamicalSystem/private/get_kappas.m` | forcing metadata utility | `ssmtoolpy.dynamical_system.forcing_kappas` | ported | not differentiable |
| `@DynamicalSystem/private/set_Fext.m` and `get_Fext.m` | forcing representation conversion | `ssmtoolpy.dynamical_system.first_order_forcing_terms_from_second_order` | functional second-to-first-order padding ported | differentiable for fixed forcing structure |
| `@DynamicalSystem/private/get_BinvA.m` | mechanical first-order conversion | `ssmtoolpy.dynamical_system.mechanical_binv_a` | ported | differentiable under nonsingular mass matrix |
| `@DynamicalSystem/private/get_A.m` | mechanical first-order conversion | `ssmtoolpy.dynamical_system.mechanical_a_matrix` | mechanical matrix branch ported | differentiable |
| `@DynamicalSystem/private/get_B.m` | mechanical first-order conversion | `ssmtoolpy.dynamical_system.mechanical_b_matrix` | mechanical matrix branch ported | differentiable |
| `@DynamicalSystem/private/get_F_input_dim.m` and `get_fnl_input_dim.m` | nonlinear metadata utility | `ssmtoolpy.dynamical_system.polynomial_input_dim` | intrusive polynomial/tensor behavior ported | not differentiable |
| `@DynamicalSystem/private/get_F_non_input_dim.m` and `get_fnl_non_input_dim.m` | callable metadata utility | `ssmtoolpy.dynamical_system.infer_callable_input_dim` | functional probing behavior ported | not differentiable |
| `@DynamicalSystem/private/get_F_semi_input_dim.m`, `get_fnl_semi_input_dim.m`, and `get_nl_input_dim.m` | callable metadata utility | `ssmtoolpy.dynamical_system.infer_semi_intrusive_input_dim` plus dispatch by caller | functional probing behavior ported | not differentiable |
| `@DynamicalSystem/private/get_degree.m` | nonlinear metadata utility | `ssmtoolpy.dynamical_system.polynomial_degree` | functional term-sequence behavior ported | not differentiable |
| `@DynamicalSystem/private/set_Ftens_from_fnlmulti.m` | nonlinear representation conversion | `ssmtoolpy.dynamical_system.first_order_polynomial_terms_from_second_order` | dense/multi-index equivalent ported | differentiable |
| `@DynamicalSystem/private/set_Ftens_from_fnltens.m` | nonlinear representation conversion | `ssmtoolpy.dynamical_system.first_order_tensor_terms_from_second_order` | dense tensor equivalent ported | differentiable |
| `@DynamicalSystem/private/get_F_from_fnl.m` | nonlinear representation conversion | `ssmtoolpy.dynamical_system.first_order_terms_from_second_order` | functional equivalent ported | differentiable |
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
| `@Manifold/choose_E.m` | modal subspace selection | `ssmtoolpy.manifold.choose_master_subspace` and `resonance_analysis` | functional spectral-data branch ported | not differentiable |
| `@Manifold/compuate_invariance_residual.m` | invariance residual utility | `ssmtoolpy.manifold.autonomous_invariance_residual` | autonomous branch ported | piecewise differentiable |
| `@Manifold/compute_auto_invariance_error.m` | invariance residual utility | `ssmtoolpy.manifold.compute_auto_invariance_error` | 2D/4D autonomous sampling behavior ported | piecewise differentiable |
| `@Manifold/private/coeffs_conj2full.m` | coefficient ordering utility | `ssmtoolpy.coefficients.coeffs_conj2full` | ported | differentiable for fixed index arrays |
| `@Manifold/private/coeffs_conj2lex.m` | coefficient ordering utility | `ssmtoolpy.coefficients.coeffs_conj2lex` | ported | differentiable for fixed ordering data |
| `@Manifold/private/coeffs_lex2revlex.m` | coefficient ordering utility | `ssmtoolpy.coefficients.coeffs_lex2revlex` | ported | differentiable for fixed structures |
| `@Manifold/private/coeffs_output.m` | coefficient output utility | `ssmtoolpy.coefficients.coeffs_output` | ported | not differentiable |
| nested `coeffs_setup.m/conjugate_ordering` | coefficient ordering utility | `ssmtoolpy.coefficients.conjugate_ordering` | ported | not differentiable |
| `@Manifold/private/check_DStype.m` | manifold validation utility | `ssmtoolpy.manifold.check_ds_type` | ported as functional classifier | not differentiable |
| `@Manifold/private/check_COMPtype.m` | manifold validation utility | `ssmtoolpy.manifold.check_comp_type` | ported as functional selector | not differentiable |
| `@Manifold/private/Aut_resonant_terms.m` | manifold resonance utility | `ssmtoolpy.manifold.autonomous_resonant_terms` | ported with zero-based Python indices | not differentiable |
| `@Manifold/private/Aut_1stOrder_RedDyn.m` | autonomous cohomological kernel | `ssmtoolpy.manifold.autonomous_first_order_reduced_dynamics` | functional resonant projection ported | differentiable for fixed resonance pattern |
| `@Manifold/private/Aut_1stOrder_SSM.m` | autonomous cohomological kernel | `ssmtoolpy.manifold.autonomous_first_order_ssm` | functional coefficient solve ported | differentiable under fixed resonance/nondegeneracy assumptions |
| `@Manifold/private/Aut_2ndOrder_RedDyn.m` | autonomous cohomological kernel | `ssmtoolpy.manifold.autonomous_second_order_reduced_dynamics` | single-resonance branch ported; 1:1 resonance remains unsupported like MATLAB error path | differentiable under fixed resonance/nondegeneracy assumptions |
| `@Manifold/private/Aut_2ndOrder_SSM.m` | autonomous cohomological kernel | `ssmtoolpy.manifold.autonomous_second_order_ssm` | analytic reduced-dynamics branch ported | differentiable under fixed resonance/nondegeneracy assumptions |
| `@Manifold/private/nonAut_resonant_terms.m` | manifold resonance utility | `ssmtoolpy.manifold.nonautonomous_resonant_terms` | zero/k branches ported with zero-based Python indices | not differentiable |
| `@Manifold/private/nonAut_conj_red.m` | non-autonomous forcing bookkeeping | `ssmtoolpy.manifold.nonautonomous_conjugate_reduction` | ported with zero-based Python index maps | not differentiable |
| `@Manifold/private/nonAut_struct_setup.m` | non-autonomous coefficient setup | `ssmtoolpy.manifold.nonautonomous_struct_setup` | ported as immutable Python containers | not differentiable |
| `@Manifold/private/nonAut_assembleCoefficients.m` | non-autonomous coefficient setup | `ssmtoolpy.manifold.nonautonomous_assemble_coefficients` | ported as immutable one-harmonic update | differentiable for fixed index structure |
| `@Manifold/private/nonAut_W1R0_plus_W0R1.m` | non-autonomous coefficient algebra | `ssmtoolpy.manifold.nonautonomous_w1r0_plus_w0r1` | ported for lex/revlex-style Python polynomial containers | differentiable for fixed index structure |
| `@Manifold/private/fnl_nonIntrusive.m` | manifold force composition | `ssmtoolpy.manifold.fnl_nonintrusive` | revlex branch ported | differentiable for fixed index structure |
| `@Manifold/private/fnl_semiIntrusive.m` | manifold force composition | `ssmtoolpy.manifold.fnl_semi_intrusive` | revlex branch ported | differentiable for fixed index structure |
| `@Manifold/private/dfnl_nonIntrusive.m` | non-autonomous Jacobian force composition | `ssmtoolpy.manifold.dfnl_nonintrusive` | revlex branch ported | differentiable for fixed index structure |
| `@Manifold/private/dfnl_semiIntrusive.m` | non-autonomous Jacobian force composition | `ssmtoolpy.manifold.dfnl_semi_intrusive` | revlex branch ported | differentiable for fixed index structure |
| `@Manifold/private/coeffs_composition.m` | manifold coefficient algebra | `ssmtoolpy.manifold.coeffs_composition` | lex/revlex branches ported | differentiable for fixed index structure |
| `@Manifold/private/coeffs_mixed_terms.m` | manifold coefficient algebra | `ssmtoolpy.manifold.coeffs_mixed_terms` | lex/revlex branches ported | differentiable for fixed index structure |
| `DSOptions.m` | options class | `ssmtoolpy.options.DSOptions` | ported | not differentiable |
| `ManifoldOptions.m` | options class | `ssmtoolpy.options.ManifoldOptions` | ported | not differentiable |
| `FRCOptions.m` | options class | `ssmtoolpy.options.FRCOptions` | ported | not differentiable |
| `FRSOptions.m` | options class | `ssmtoolpy.options.FRSOptions` | ported | not differentiable |

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

- MATLAB sparse `sptensor` support is partially represented with JAX `BCOO` for
  tensor composition. This covers the sparse multilinear contraction used by
  `misc/tensor_composition.m`, but it is not a full Sandia Tensor Toolbox clone.
  Broader sparse-coordinate APIs may still be needed for high-level class
  parity.
- Multi-index notation remains a priority for practical performance. The
  MATLAB implementation is often more efficient in multi-index form than tensor
  form, so future cohomological and intrusive-force ports should prefer the
  existing `MultiIndexPolynomial` path when both representations are available.
- Eigenvector sorting, nullspaces, rank decisions, resonant-mode detection, and
  continuation/event routines require explicit nondegeneracy assumptions before
  differentiability can be claimed.
- `choose_master_subspace` assumes spectral data have already been computed; it
  does not port the mutable call to `System.linear_spectral_analysis`.
- `autonomous_invariance_residual` ports only the autonomous branch of the
  misspelled MATLAB `compuate_invariance_residual.m`. The non-autonomous branch
  includes forcing evaluation and plotting side effects and remains unported.
- `compute_fixed_points_2d` uses an internal marching-squares style locator
  instead of MATLAB's `contourc`/`polyxpoly`; tests cover representative grid
  intersections, but dense contour parity is not yet exhaustively validated.
- `solve_invariance_equation` uses `jax.scipy.sparse.linalg` for iterative
  solver names. JAX provides `cg`, `bicgstab`, and `gmres`; MATLAB names without
  exact JAX equivalents are mapped conservatively (`bicg`/`cgs` to `bicgstab`,
  `lsqr` to `gmres`).
- `linear_response` is ported as functional kernels instead of a method taking
  the mutable MATLAB `DS` object. The second-order helper has an explicit
  `conjugate_symmetric=True` option for MATLAB's common two-harmonic
  conjugate-pair layout.
- `transient_traj_on_auto_ssm` is ported as a functional autonomous trajectory
  helper. MATLAB uses adaptive `ode45`; Python uses fixed-step RK4 on the
  requested sampling grid to keep the core JAX-transformable.
- `reduced_dynamics_symbolic` currently ports the autonomous polar expression
  branch. The optional non-autonomous symbolic forcing branch remains future
  work and should be added alongside the corresponding non-autonomous reduced
  dynamics APIs.
- `proj2SSM` nonlinear optimization is not ported as an optimizer. The
  differentiable objective is exposed for use with a future JAX optimizer.
- MATLAB `squaDist2pointSSM.m` calls `reduced_to_full(x,W_0)` although the local
  MATLAB `reduced_to_full` signature expects non-autonomous arguments too. The
  Python port uses the intended autonomous reconstruction behavior.
- The full mutable MATLAB `DynamicalSystem` class surface is not ported yet.
  This pass adds functional JAX kernels for nonlinear force, forcing, residual,
  and ODE RHS evaluation.
- Some `@DynamicalSystem/private` methods have MATLAB class-specific branches
  for stored properties. The Python ports expose the equivalent functional
  mechanical-matrix and nonlinear-representation conversions.
- Dynamical-system forcing metadata and second-to-first-order forcing padding
  are exposed as functional helpers instead of mutable `Fext`/`fext` property
  getters.
- `coeffs_composition` and `coeffs_mixed_terms` currently cover lexicographic
  and reverse-lexicographic computation. The conjugate-order branches are tied
  to the full cohomological solver and remain unported.
- Autonomous first- and second-order SSM coefficient kernels are ported as
  functional one-order solves. They do not yet assemble full `W_0`/`R_0`
  sequences or replace `cohomological_solution.m`.
- The second-order autonomous reduced-dynamics kernel preserves MATLAB's
  limitation for 1:1 internal resonance in `COMPtype=second` by raising
  `NotImplementedError`.
- `fnl_nonintrusive`, `fnl_semi_intrusive`, `dfnl_nonintrusive`, and
  `dfnl_semi_intrusive` currently cover the reverse-lexicographic branches.
  The conjugate-order branches and the intrusive tensor composition helpers
  remain unported.
- Manifold resonance helpers return zero-based Python indices while preserving
  MATLAB `find` ordering. The MATLAB source returns one-based indices.
- Non-autonomous Manifold setup helpers use immutable tuple/NamedTuple
  containers and dense zero coefficient arrays instead of mutable MATLAB
  structs and sparse empty matrices.
- MATLAB option classes are ported as Python dataclasses with default values,
  validation, and MATLAB-style field-name export. They are configuration
  containers rather than JAX-transformable numerical kernels.
- MATLAB R2024b spot fixtures have been generated for selected low-level
  helpers; broad fixture generation for full examples remains outstanding.
