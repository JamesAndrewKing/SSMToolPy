# Migration Inventory

Generated from `SSMTool/src` after inspecting the MATLAB source tree. `SSMTool/ext` is bundled third-party MATLAB dependency code and is excluded from this migration target inventory.

| MATLAB file | Category | Planned Python destination | Migration status | Differentiability classification |
| --- | --- | --- | --- | --- |
| `@DynamicalSystem/DynamicalSystem.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/add_forcing.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/compute_dfnldx.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_internal_force_jacobian_x` | functional core ported | differentiable for intrusive terms |
| `@DynamicalSystem/compute_dfnldxd.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_internal_force_jacobian_xd` | functional displacement-only behavior ported | differentiable |
| `@DynamicalSystem/compute_fext.m` | forcing evaluation | `ssmtoolpy.dynamical_system.evaluate_periodic_forcing` | functional periodic forcing core ported | differentiable for fixed forcing structure |
| `@DynamicalSystem/compute_fnl.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_internal_force` | functional core ported | differentiable for intrusive terms |
| `@DynamicalSystem/evaluate_Fext.m` | forcing evaluation | `ssmtoolpy.dynamical_system.evaluate_periodic_forcing` | functional periodic forcing core ported | differentiable for fixed forcing structure |
| `@DynamicalSystem/evaluate_Fnl.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.first_order_nonlinearity` and `first_order_from_second_order_nonlinearity` | functional core ported | differentiable for intrusive terms |
| `@DynamicalSystem/linear_spectral_analysis.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/odefun.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.evaluate_first_order_vector_field` | functional core ported | differentiable under nonsingular `B` |
| `@DynamicalSystem/private/get_A.m` | mechanical first-order conversion | `ssmtoolpy.dynamical_system.mechanical_a_matrix` | mechanical matrix branch ported | differentiable |
| `@DynamicalSystem/private/get_B.m` | mechanical first-order conversion | `ssmtoolpy.dynamical_system.mechanical_b_matrix` | mechanical matrix branch ported | differentiable |
| `@DynamicalSystem/private/get_BinvA.m` | mechanical first-order conversion | `ssmtoolpy.dynamical_system.mechanical_binv_a` | ported | differentiable under nonsingular mass matrix |
| `@DynamicalSystem/private/get_F.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/get_F_from_fnl.m` | nonlinear representation conversion | `ssmtoolpy.dynamical_system.first_order_terms_from_second_order` | functional equivalent ported | differentiable |
| `@DynamicalSystem/private/get_F_input_dim.m` | nonlinear metadata utility | `ssmtoolpy.dynamical_system.polynomial_input_dim` | intrusive polynomial/tensor behavior ported | not differentiable |
| `@DynamicalSystem/private/get_F_non.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/get_F_non_input_dim.m` | callable metadata utility | `ssmtoolpy.dynamical_system.infer_callable_input_dim` | ported | not differentiable |
| `@DynamicalSystem/private/get_F_semi.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/get_F_semi_input_dim.m` | callable metadata utility | `ssmtoolpy.dynamical_system.infer_semi_intrusive_input_dim` | ported | not differentiable |
| `@DynamicalSystem/private/get_Fext.m` | forcing representation conversion | `ssmtoolpy.dynamical_system.first_order_forcing_terms_from_second_order` | functional second-to-first-order padding ported | differentiable for fixed forcing structure |
| `@DynamicalSystem/private/get_dF.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/get_dF_non.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/get_dF_semi.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/get_degree.m` | nonlinear metadata utility | `ssmtoolpy.dynamical_system.polynomial_degree` | functional term-sequence behavior ported | not differentiable |
| `@DynamicalSystem/private/get_dfnl.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/get_fnl_input_dim.m` | nonlinear metadata utility | `ssmtoolpy.dynamical_system.polynomial_input_dim` | intrusive polynomial/tensor behavior ported | not differentiable |
| `@DynamicalSystem/private/get_fnl_non_input_dim.m` | callable metadata utility | `ssmtoolpy.dynamical_system.infer_callable_input_dim` | ported | not differentiable |
| `@DynamicalSystem/private/get_fnl_semi_input_dim.m` | callable metadata utility | `ssmtoolpy.dynamical_system.infer_semi_intrusive_input_dim` | ported | not differentiable |
| `@DynamicalSystem/private/get_kappas.m` | forcing metadata utility | `ssmtoolpy.dynamical_system.forcing_kappas` | ported | not differentiable |
| `@DynamicalSystem/private/get_nl_input_dim.m` | nonlinear metadata dispatch | `ssmtoolpy.dynamical_system.polynomial_input_dim`, `infer_callable_input_dim`, `infer_semi_intrusive_input_dim` | functional cases ported; mutable property dispatch not ported | not differentiable |
| `@DynamicalSystem/private/set_F.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_F_non.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_F_semi.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_F_tensor.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_Fext.m` | forcing representation conversion | `ssmtoolpy.dynamical_system.first_order_forcing_terms_from_second_order` | functional second-to-first-order padding ported | differentiable for fixed forcing structure |
| `@DynamicalSystem/private/set_Ftens_from_fnlmulti.m` | nonlinear representation conversion | `ssmtoolpy.dynamical_system.first_order_polynomial_terms_from_second_order` | dense/multi-index equivalent ported | differentiable |
| `@DynamicalSystem/private/set_Ftens_from_fnltens.m` | nonlinear representation conversion | `ssmtoolpy.dynamical_system.first_order_tensor_terms_from_second_order` | dense tensor equivalent ported | differentiable |
| `@DynamicalSystem/private/set_dF.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_dF_from_dfnl.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_dF_non.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_dF_semi.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_dfnl.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_fnl.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/private/set_fnl_tensor.m` | class/model API | `ssmtoolpy.dynamical_system` | not yet ported | not yet verified |
| `@DynamicalSystem/residual.m` | dynamical-system evaluation | `ssmtoolpy.dynamical_system.second_order_residual` | functional core ported | piecewise differentiable |
| `@Manifold/Manifold.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/choose_E.m` | modal subspace selection | `ssmtoolpy.manifold.choose_master_subspace` and `resonance_analysis` | functional spectral-data branch ported | not differentiable |
| `@Manifold/cohomological_solution.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/compuate_invariance_residual.m` | invariance residual utility | `ssmtoolpy.manifold.autonomous_invariance_residual` | autonomous branch ported | piecewise differentiable |
| `@Manifold/compute_analyticity_domain.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/compute_auto_invariance_error.m` | invariance residual utility | `ssmtoolpy.manifold.compute_auto_invariance_error` | 2D/4D autonomous sampling behavior ported | piecewise differentiable |
| `@Manifold/compute_perturbed_whisker.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/compute_sensitivity_coefficients.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/compute_whisker.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/Aut_1stOrder_RedDyn.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/Aut_1stOrder_SSM.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/Aut_2ndOrder_RedDyn.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/Aut_2ndOrder_SSM.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/Aut_resonant_terms.m` | manifold resonance utility | `ssmtoolpy.manifold.autonomous_resonant_terms` | ported with zero-based Python indices | not differentiable |
| `@Manifold/private/check_COMPtype.m` | manifold validation utility | `ssmtoolpy.manifold.check_comp_type` | ported as functional selector | not differentiable |
| `@Manifold/private/check_DStype.m` | manifold validation utility | `ssmtoolpy.manifold.check_ds_type` | ported as functional classifier | not differentiable |
| `@Manifold/private/coeffs_composition.m` | manifold coefficient algebra | `ssmtoolpy.manifold.coeffs_composition` | lex/revlex branches ported | differentiable for fixed index structure |
| `@Manifold/private/coeffs_conj2full.m` | coefficient ordering utility | `ssmtoolpy.coefficients.coeffs_conj2full` | ported | differentiable for fixed index arrays |
| `@Manifold/private/coeffs_conj2lex.m` | coefficient ordering utility | `ssmtoolpy.coefficients.coeffs_conj2lex` | ported | differentiable for fixed ordering data |
| `@Manifold/private/coeffs_lex2revlex.m` | coefficient ordering utility | `ssmtoolpy.coefficients.coeffs_lex2revlex` | ported | differentiable for fixed structures |
| `@Manifold/private/coeffs_mixed_terms.m` | manifold coefficient algebra | `ssmtoolpy.manifold.coeffs_mixed_terms` | lex/revlex branches ported | differentiable for fixed index structure |
| `@Manifold/private/coeffs_output.m` | coefficient output utility | `ssmtoolpy.coefficients.coeffs_output` | ported | not differentiable |
| `@Manifold/private/coeffs_setup.m` | manifold coefficient setup | `ssmtoolpy.coefficients.conjugate_ordering` plus future setup API | partially ported: nested conjugate ordering only | not differentiable |
| `@Manifold/private/dfnl_intrusive.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/dfnl_nonIntrusive.m` | non-autonomous Jacobian force composition | `ssmtoolpy.manifold.dfnl_nonintrusive` | revlex branch ported | differentiable for fixed index structure |
| `@Manifold/private/dfnl_semiIntrusive.m` | non-autonomous Jacobian force composition | `ssmtoolpy.manifold.dfnl_semi_intrusive` | revlex branch ported | differentiable for fixed index structure |
| `@Manifold/private/fnl_intrusive.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/fnl_nonIntrusive.m` | manifold force composition | `ssmtoolpy.manifold.fnl_nonintrusive` | revlex branch ported | differentiable for fixed index structure |
| `@Manifold/private/fnl_semiIntrusive.m` | manifold force composition | `ssmtoolpy.manifold.fnl_semi_intrusive` | revlex branch ported | differentiable for fixed index structure |
| `@Manifold/private/multi_addition.m` | core utility | `ssmtoolpy.multi_index.multi_addition` | ported | not differentiable |
| `@Manifold/private/multi_index_2_ordering.m` | core utility | `ssmtoolpy.multi_index.multi_index_2_ordering` | ported | not differentiable |
| `@Manifold/private/multi_nsumk.m` | core utility | `ssmtoolpy.multi_index.multi_nsumk` | ported | not differentiable |
| `@Manifold/private/multi_subtraction.m` | core utility | `ssmtoolpy.multi_index.multi_subtraction` | ported | not differentiable |
| `@Manifold/private/nonAut_1stOrder_SolveInvEq.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_1stOrder_highTerms.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_1stOrder_leadTerms.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_1stOrder_whisker.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_2ndOrder_RedDyn.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_2ndOrder_SolveInvEq.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_2ndOrder_highTerms.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_2ndOrder_leadTerms.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_2ndOrder_whisker.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_Fext_plus_Fnl.m` | manifold coefficient API | `ssmtoolpy.manifold` | not yet ported | not yet verified |
| `@Manifold/private/nonAut_W1R0_plus_W0R1.m` | non-autonomous coefficient algebra | `ssmtoolpy.manifold.nonautonomous_w1r0_plus_w0r1` | ported for Python polynomial containers | differentiable for fixed index structure |
| `@Manifold/private/nonAut_assembleCoefficients.m` | non-autonomous coefficient setup | `ssmtoolpy.manifold.nonautonomous_assemble_coefficients` | ported as immutable one-harmonic update | differentiable for fixed index structure |
| `@Manifold/private/nonAut_conj_red.m` | non-autonomous forcing bookkeeping | `ssmtoolpy.manifold.nonautonomous_conjugate_reduction` | ported with zero-based Python index maps | not differentiable |
| `@Manifold/private/nonAut_resonant_terms.m` | manifold resonance utility | `ssmtoolpy.manifold.nonautonomous_resonant_terms` | zero/k branches ported with zero-based Python indices | not differentiable |
| `@Manifold/private/nonAut_struct_setup.m` | non-autonomous coefficient setup | `ssmtoolpy.manifold.nonautonomous_struct_setup` | ported as immutable Python containers | not differentiable |
| `@SSM/FRC_cont_ep.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/FRC_cont_po.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/FRC_level_set.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_BP2ep.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_BP2po.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_BP2tor.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_HB2po.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_TR2tor.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_cont_ep.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_cont_po.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_cont_tor.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_ep2HB.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_ep2SN.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_ep2ep.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_epSweeps.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_isol2ep.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_isol2po.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_lvlSweeps.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_po2PD.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_po2SN.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_po2TR.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_po2po.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_poSweeps.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/SSM_tor2tor.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/activate_parallel.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/auto_po_solver.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/deactivate_parallel.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/extract_FRC.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/extract_FRS.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/extract_Stability_Diagram.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/extract_backbone.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/extract_ridges_trenches.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/FRC_reduced_to_full.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/auto_ode_2mDSSM_cartesian.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/cal_FRS_via_ana.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/cal_FRS_via_cont_ep.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/cal_ab_dab.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/cal_rhos.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/check_auto_reduced_dynamics.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/check_spectrum_and_internal_resonance.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/check_stability.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/cocoSet.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/compute_fixed_points_2D.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/compute_full_response_2mD_ReIm.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/compute_full_response_traj.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/compute_gamma.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/compute_output_polar2D.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/compute_reduced_dynamics_2D_polar.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/create_data_for_po_amp.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/create_reduced_dynamics_data.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/damped_backbone_l2.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/damped_backbone_linf.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/detect_resonant_modes.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ep_reduced_results.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/frc_Jacobian.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/frc_ab.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/frc_psi.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/frs_level_set.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/frs_output.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/get_contour_xy.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/initial_fixed_point.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/leading_order_nonauto_SSM.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/monitor_scaled_states.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/monitor_states.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2DSSM_cartesian.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2DSSM_cartesian_DFDP.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2DSSM_cartesian_DFDX.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2DSSM_cartesian_fixROM.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2DSSM_cartesian_fixROM_DFDP.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2DSSM_cartesian_fixROM_DFDX.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2mDSSM_cartesian.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2mDSSM_cartesian_DFDP.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2mDSSM_cartesian_DFDX.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2mDSSM_polar.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2mDSSM_polar_DFDP.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/ode_2mDSSM_polar_DFDX.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/plot3_frc_full.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/plot_frc_full.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/po_amp.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/po_amp_du.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/po_full_results.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/po_reduced_results.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/stab_plot3.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `@SSM/private/tor_reduced_results.m` | SSM workflow/continuation API | `ssmtoolpy.ssm` | not yet ported | not yet verified |
| `DSOptions.m` | options class | `ssmtoolpy.options.DSOptions` | ported | not differentiable |
| `FRCOptions.m` | options class | `ssmtoolpy.options.FRCOptions` | ported | not differentiable |
| `FRSOptions.m` | options class | `ssmtoolpy.options.FRSOptions` | ported | not differentiable |
| `ManifoldOptions.m` | options class | `ssmtoolpy.options.ManifoldOptions` | ported | not differentiable |
| `frc/check_stability.m` | FRC utility | `ssmtoolpy.frc.check_stability` | ported | not differentiable |
| `frc/compute_fixed_points_2D.m` | FRC utility | `ssmtoolpy.frc.compute_fixed_points_2d` | ported with lightweight grid-contour intersection | not differentiable |
| `frc/compute_gamma.m` | FRC utility | `ssmtoolpy.frc.compute_gamma` | ported | not differentiable |
| `frc/frc_Jacobian.m` | FRC kernel | `ssmtoolpy.frc.frc_jacobian` | ported | differentiable for `rho != 0` |
| `frc/frc_ab.m` | FRC kernel | `ssmtoolpy.frc.frc_ab` | ported | differentiable |
| `frc/frc_psi.m` | FRC kernel | `ssmtoolpy.frc.frc_psi` | ported | piecewise differentiable |
| `frc/get_contour_xy.m` | FRC utility | `ssmtoolpy.frc.get_contour_xy` | ported | not differentiable |
| `misc/SSM_ep_read_solution.m` | solution reader | `ssmtoolpy.io` | not yet ported | not differentiable |
| `misc/SSM_plot_torus.m` | numerical/utility function | `ssmtoolpy.misc` | not yet ported | not yet verified |
| `misc/SSM_po_read_solution.m` | solution reader | `ssmtoolpy.io` | not yet ported | not differentiable |
| `misc/SSM_tor_read_solution.m` | solution reader | `ssmtoolpy.io` | not yet ported | not differentiable |
| `misc/StEP.m` | polarization utility | `ssmtoolpy.manifold.step_polynomial` | ported for orders 1-3 | differentiable if supplied callable is differentiable |
| `misc/auto_red_dyn.m` | reduced dynamics kernel | `ssmtoolpy.misc.auto_red_dyn` | ported | differentiable |
| `misc/expand_multiindex.m` | core kernel | `ssmtoolpy.multi_index.expand_multiindex` | ported | differentiable |
| `misc/expand_multiindex_derivative.m` | core kernel | `ssmtoolpy.multi_index.expand_multiindex_derivative` | ported | differentiable |
| `misc/expand_tensor.m` | core kernel | `ssmtoolpy.tensor.expand_tensor` | ported for dense tensors | differentiable |
| `misc/expand_tensor_derivative.m` | core kernel | `ssmtoolpy.tensor.expand_tensor_derivative` | ported for dense tensors | differentiable |
| `misc/extract_output.m` | output utility | `ssmtoolpy.misc.extract_output` | ported | piecewise differentiable |
| `misc/frc_ab.m` | FRC kernel | `ssmtoolpy.frc.frc_ab` | ported | differentiable |
| `misc/isocurve3.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/khatri_rao_product.m` | core kernel | `ssmtoolpy.tensor.khatri_rao_product` | ported | differentiable |
| `misc/linear_response.m` | linear response utility | `ssmtoolpy.misc.first_order_linear_response` and `second_order_linear_response` | functional first- and second-order branches ported | differentiable under nonsingular frequency-domain operators; amplitudes are piecewise differentiable |
| `misc/monitor_memory.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/multi_index_to_tensor.m` | conversion utility | `ssmtoolpy.multi_index.multi_index_to_tensor` | ported for dense tensors | not differentiable |
| `misc/nsumk.m` | core utility | `ssmtoolpy.multi_index.nsumk` | ported | not differentiable |
| `misc/plot3_stab_lines.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/plot_2D_auto_SSM.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/plot_FRC.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/plot_analytical_domain.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/plot_atlas_2d.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/plot_backbone_curves.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/plot_stab_lines.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/proj2SSM.m` | projection utility | `ssmtoolpy.misc.project_to_ssm_linear` and `nonlinear_projection_objective` | partially ported: linear projection and nonlinear objective only | differentiable |
| `misc/read_num_int_sol.m` | solution reader | `ssmtoolpy.io` | not yet ported | not differentiable |
| `misc/read_po_ssm_init.m` | solution reader | `ssmtoolpy.io` | not yet ported | not differentiable |
| `misc/reduced_dynamics_symbolic.m` | reduced dynamics documentation utility | `ssmtoolpy.misc.reduced_dynamics_symbolic` | autonomous polar symbolic rendering ported | not differentiable |
| `misc/reduced_to_full.m` | reconstruction kernel | `ssmtoolpy.reduction.reduced_to_full` | partially ported | differentiable |
| `misc/reduced_to_full_complex.m` | reconstruction kernel | `ssmtoolpy.reduction.reduced_to_full_complex` | partially ported | not yet verified |
| `misc/reduced_to_full_traj.m` | reconstruction kernel | `ssmtoolpy.misc.reduced_to_full_traj` | ported | differentiable |
| `misc/solveinveq.m` | linear solve utility | `ssmtoolpy.misc.solve_invariance_equation` | ported with JAX direct, pseudoinverse, and sparse iterative solvers | differentiable under nondegeneracy assumptions |
| `misc/spblkdiag.m` | linear algebra utility | `ssmtoolpy.misc.spblkdiag` | ported as dense JAX array | differentiable |
| `misc/squaDist2pointSSM.m` | projection objective | `ssmtoolpy.misc.squared_distance_to_point_ssm` | ported against autonomous reconstruction API | differentiable |
| `misc/stab_plot.m` | plotting/diagnostic utility | `not planned for numerical core yet` | not yet ported | not yet verified |
| `misc/sub2multiind.m` | core utility | `ssmtoolpy.multi_index.sub2multiind` | ported | not differentiable |
| `misc/tensor_composition.m` | tensor composition kernel | `ssmtoolpy.tensor.tensor_composition` and `tensor_product` | ported for dense tensors and JAX `BCOO` sparse tensors | differentiable for fixed dense pattern; sparse path supports forward-mode for fixed sparsity |
| `misc/tensor_to_multi_index.m` | conversion utility | `ssmtoolpy.multi_index.tensor_to_multi_index` | ported for dense tensors | not differentiable |
| `misc/transient_traj_on_auto_ssm.m` | reduced dynamics trajectory utility | `ssmtoolpy.misc.transient_traj_on_auto_ssm` | functional autonomous branch ported with fixed-step RK4 | differentiable for fixed step count and structures |
