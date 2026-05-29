# Differentiability Status

| Public API | Status | Notes |
| --- | --- | --- |
| `ConjugateOrdering` | not differentiable | Frozen index-data container; coefficient reconstruction APIs carry value differentiability. |
| `MultiIndexPolynomial` | not differentiable | Frozen data container; evaluation APIs carry differentiability. |
| `number_of_multis` | not differentiable | Discrete combinatorial count. |
| `conjugate_flip` | not differentiable | Discrete coordinate permutation. |
| `conjugate_ordering` | not differentiable | Discrete multi-index ordering construction. |
| `coeffs_conj2full` | differentiable | Linear coefficient reconstruction for fixed index arrays; tested with `jax.grad`. |
| `coeffs_conj2lex` | differentiable | Linear coefficient reconstruction for fixed conjugate-ordering data; tested with `jax.grad`. |
| `coeffs_lex2revlex` | differentiable | Column permutation for fixed structures; tested with `jax.grad`. |
| `coeffs_output` | not differentiable | Performs exact-zero column selection and attaches discrete multi-indices. |
| `nsumk` | not differentiable | Discrete integer tuple enumeration. |
| `sub2multiind` | not differentiable | Discrete one-based subscript counting. |
| `expand_multiindex` | differentiable | Tested with `jax.jit` and `jax.grad`. |
| `expand_multiindex_derivative` | differentiable | Implemented through JAX autodiff; tested indirectly against `jax.jacfwd`. |
| `tensor_to_multi_index` | not differentiable | Discrete grouping/conversion from dense tensor entries. |
| `multi_index_to_tensor` | not differentiable | Discrete canonical tensor placement. |
| `multi_addition` | not differentiable | Discrete index algebra. |
| `multi_subtraction` | not differentiable | Discrete index algebra and filtering. |
| `multi_nsumk` | not differentiable | Discrete multi-index partition enumeration with optional filtering and deduplication. |
| `multi_index_2_ordering` | not differentiable | Discrete combinatorial ordering lookup. |
| `khatri_rao_product` | differentiable | Tested with `jax.jacfwd`. |
| `expand_tensor` | differentiable | Tested with `jax.jit` and `jax.grad`. |
| `expand_tensor_derivative` | differentiable | Tested against `jax.jacfwd`. |
| `sparse_tensor_from_dense` | piecewise differentiable | JAX `BCOO` adapter. Values are differentiable for fixed sparse capacity/indices; index extraction is discrete. |
| `tensor_to_dense` | differentiable for fixed sparse indices | Densifies JAX `BCOO` storage or returns dense input. |
| `tensor_product` | differentiable for dense inputs; differentiable under fixed sparse-structure assumptions for sparse inputs | Dense Tucker-style contraction is tested with `jax.jacfwd`. Sparse `BCOO` contraction is tested with `jax.jit` and forward-mode `jax.jacfwd`; reverse-mode through sparse-sparse `bcoo_spdot_general` is not implemented in JAX 0.10.1. |
| `tensor_composition` | differentiable for fixed dense pattern; differentiable under fixed sparse-structure assumptions for sparse inputs | Dense row-pattern sum is tested with `jax.jit`, `jax.grad`, and `jax.vmap`. Sparse `BCOO` composition is tested with `jax.jit` and forward-mode `jax.jacfwd`; integer patterns and result shapes are static/discrete. |
| `frc_ab` | differentiable | Tested with `jax.grad` and `jax.vmap`. |
| `cal_ab_dab` | differentiable | Autonomous polar normal-form terms and radius derivatives; tested with `jax.jacfwd`. |
| `ReducedDynamicsHarmonic` | not differentiable | Static non-autonomous reduced-dynamics harmonic container. |
| `compute_reduced_dynamics_2d_polar` | differentiable for fixed harmonic structure | Polar-grid 2D reduced dynamics; harmonic sign normalization and coefficient presence are discrete. Tested with `jax.grad`. |
| `ode_2d_ssm_cartesian` | differentiable for fixed polynomial/harmonic structure | Functional Cartesian 2D SSM reduced dynamics; MATLAB object recomputation branch is outside this kernel. Tested with `jax.jit` and `jax.jacfwd`. |
| `ode_2d_ssm_cartesian_jac_x` | differentiable for fixed polynomial/harmonic structure | Jacobian with respect to Cartesian reduced coordinates via `jax.jacfwd`. |
| `ode_2d_ssm_cartesian_jac_params` | differentiable for fixed polynomial/harmonic structure | Jacobian with respect to `[Omega, epsilon]`; frequency-dependence of recomputed coefficients is outside this functional kernel. |
| `ode_2d_ssm_cartesian_fixrom` | differentiable for fixed polynomial/harmonic structure | Fixed-ROM alias for explicit coefficient evaluation. |
| `ode_2d_ssm_cartesian_fixrom_jac_x` | differentiable for fixed polynomial/harmonic structure | Fixed-ROM coordinate Jacobian alias. |
| `ode_2d_ssm_cartesian_fixrom_jac_params` | differentiable for fixed polynomial/harmonic structure | Fixed-ROM parameter Jacobian alias. |
| `ReducedDynamics2mDData` | not differentiable | Static coefficient container for 2m-dimensional reduced-dynamics ODE kernels. |
| `ode_2md_ssm_cartesian` | differentiable for fixed polynomial/harmonic structure | 2mD Cartesian reduced dynamics; tested with `jax.jit`. |
| `ode_2md_ssm_cartesian_jac_x` | differentiable for fixed polynomial/harmonic structure | State Jacobian via `jax.jacfwd`; tested against direct AD. |
| `ode_2md_ssm_cartesian_jac_params` | differentiable for fixed polynomial/harmonic structure | Parameter Jacobian via `jax.jacfwd`; base-force scaling is a static option. |
| `ode_2md_ssm_polar` | differentiable for fixed polynomial/harmonic structure and positive radii | Polar reduced dynamics includes divisions by radii; tested with `jax.jit`. |
| `ode_2md_ssm_polar_jac_x` | differentiable for fixed polynomial/harmonic structure and positive radii | State Jacobian via `jax.jacfwd`; tested against direct AD. |
| `ode_2md_ssm_polar_jac_params` | differentiable for fixed polynomial/harmonic structure and positive radii | Parameter Jacobian via `jax.jacfwd`; base-force scaling is a static option. |
| `ReducedDynamicsData` | not differentiable | Static data container for SSM reduced-dynamics workflows. |
| `POAmplitudeData` | not differentiable | Static data container for periodic-orbit amplitude objectives. |
| `cal_rhos` | differentiable except at zero radius | Cartesian-to-radius conversion; tested with `jax.grad`. |
| `monitor_state_names` | not differentiable | String metadata helper for continuation state labels. |
| `scale_parameters` | differentiable | Elementwise scaling helper from `monitor_scaled_states.m`. |
| `initial_fixed_point_guess` | piecewise differentiable | Polar regularization branches on radius signs and wraps phases modulo `2*pi`; solver refinement is not included. |
| `check_spectrum_and_internal_resonance` | not differentiable | Thresholded spectral validation with exceptions on failure. |
| `check_auto_reduced_dynamics` | not differentiable | Extracts and validates integer multi-index resonance conditions. |
| `create_reduced_dynamics_data` | not differentiable | Metadata/container constructor; MATLAB disk-write side effect omitted. |
| `reduced_data_to_2md` | not differentiable | Container conversion for 2mD ODE kernels. |
| `create_po_amplitude_data` | not differentiable | Metadata/container constructor for periodic-orbit amplitude objectives. |
| `auto_ode_2md_ssm_cartesian` | differentiable for fixed polynomial structure | Autonomous 2mD Cartesian reduced dynamics; tested with `jax.jit`. |
| `detect_resonant_modes` | not differentiable | Rounds frequency ratios and thresholds internal resonance closeness. |
| `compute_gamma` | not differentiable | Discrete multi-index lookup into reduced-dynamics coefficients. |
| `frc_psi` | piecewise differentiable | Uses `atan2`; tested with `jax.grad` and `jax.vmap`, excluding branch-cut/undefined cases. |
| `frc_jacobian` | differentiable | Differentiable for `rho != 0`; tested with `jax.jacfwd`. |
| `check_stability` | not differentiable | Thresholds determinant and trace sign conditions. |
| `get_contour_xy` | not differentiable | Parses MATLAB `contourc` plotting matrices. |
| `compute_fixed_points_2d` | not differentiable | Uses sign tests, marching-squares segments, and geometric intersection selection. |
| `reduced_to_full` | differentiable | For fixed polynomial structure; tested with `jax.jit`. Non-autonomous branch uses fixed Python structure. |
| `reduced_to_full_complex` | differentiable | For fixed polynomial/forcing structure; transform coverage not yet verified. |
| `reduced_to_full_traj` | differentiable | Single-time reconstruction; tested with `jax.jit` and `jax.jacfwd` for autonomous structure. |
| `DynamicalSystem` | not differentiable | Immutable model-data container; methods inherit their functional kernel differentiability. |
| `FirstOrderExampleModel` | not differentiable | Source-derived example data container; stored arrays and `system.odefun` carry value differentiability. |
| `planar_system_model` | not differentiable | Constructor for the MATLAB `PlanarSystem/build_model.m` coefficients. Returned ODE evaluation is differentiable under nonsingular `B`; tested with `jax.jacfwd` through `planar_system_vector_field`. |
| `planar_system_vector_field` | differentiable | Closed-form MATLAB planar example RHS; tested with `jax.jit` and `jax.jacfwd`. |
| `benchmark_ssm_1st_order_model` | not differentiable | Constructor alias for the source-identical MATLAB `BenchamrkSSM1stOrder/build_model.m`. Returned ODE evaluation inherits the planar model status. |
| `lorenz_first_order_model` | not differentiable | Constructor for `Lorenz1stOrder/build_model.m`. Returned ODE evaluation is differentiable under nonsingular `B`; tensor coefficients are fixed. |
| `lorenz_vector_field` | differentiable | Closed-form `Lorenz1stOrder/lorenz.m` RHS; tested with `jax.jit`, `jax.jacfwd`, and parameter `jax.grad`. |
| `LinearSpectrum` | not differentiable | Result container for non-differentiable spectral analysis. |
| `add_forcing` | not differentiable | Constructor/metadata conversion for MATLAB-style force columns; returned forcing evaluation is differentiable. |
| `sort_modes` | not differentiable | Eigenvalue sorting and conjugate-pair ordering are discrete. |
| `normalize_modes` | differentiable under nondegeneracy assumptions | Requires nonzero modal products; normally used after non-differentiable eigensolver steps. |
| `linear_spectral_analysis` | not differentiable | Eigensolver, stiff/zero-mode removal, sorting, and normalization conventions are branch-sensitive. |
| `extract_output` | piecewise differentiable | Norm outputs are differentiable away from zero/ties; amplitudes use infinity norms. |
| `spblkdiag` | differentiable | Dense block diagonal assembly; tested with `jax.jacfwd`. |
| `solve_invariance_equation` | differentiable under nondegeneracy assumptions | Direct solve requires nonsingular matrices; `pinv`/`lsqminnorm` assume stable rank; iterative names use `jax.scipy.sparse.linalg` implicit linear solves. |
| `auto_red_dyn` | differentiable | For fixed integer exponent matrix; tested with `jax.grad`. |
| `assemble_auto_reduced_dynamics` | differentiable for fixed polynomial structure | Concatenates fixed `R_0` coefficient blocks; exponent arrays and tuple structure are discrete. |
| `TransientTrajectory` | not differentiable | Result container. Numeric fields from trajectory integration carry value differentiability. |
| `transient_traj_on_auto_ssm` | differentiable for fixed step count and fixed structures | Uses fixed-step RK4 rather than adaptive `ode45`; tested with `jax.jit` and `jax.jacfwd`. |
| `ReducedDynamicsSymbolicOptions` | not differentiable | String-formatting options. |
| `ReducedDynamicsSymbolicResult` | not differentiable | String result container. |
| `reduced_dynamics_symbolic` | not differentiable | Performs thresholding and string generation for documentation; autonomous branch ported. |
| `LinearResponseResult` | not differentiable | Result container. Numeric fields from response kernels carry value differentiability. |
| `first_order_linear_response` | differentiable under nondegeneracy assumptions | Requires nonsingular `A - i*kappa*Omega*B`; response and norms tested with `jax.jit`/`jax.jacfwd`. Output amplitudes are piecewise differentiable because they use infinity norms. |
| `second_order_linear_response` | differentiable under nondegeneracy assumptions | Requires nonsingular dynamic stiffness matrices; tested with `jax.grad`. The `conjugate_symmetric` convention is static/discrete, and amplitudes are piecewise differentiable. |
| `ProjectionData` | not differentiable | Index-data container. |
| `AutoReducedDynamicsData` | not differentiable | Data container; `auto_red_dyn` carries value differentiability. |
| `OutputSummary` | not differentiable | Result container. |
| `real_to_conjugate_state` | differentiable | For fixed real/complex coordinate index sets. |
| `squared_distance_to_point_ssm` | differentiable | Uses autonomous `reduced_to_full`; tested with `jax.grad` via objective wrapper. |
| `project_to_ssm_linear` | differentiable | Linear projection. |
| `nonlinear_projection_objective` | differentiable | Returns objective only; optimizer driver is not included. |
| `FourierForcingTerm` | not differentiable | Forcing data container; evaluation APIs carry value differentiability. |
| `PeriodicForcing` | not differentiable | Forcing data container; evaluation APIs carry value differentiability. |
| `ResidualResult` | not differentiable | Result container. |
| `evaluate_polynomial_terms` | differentiable | For fixed polynomial/tensor term structure. |
| `evaluate_polynomial_jacobian` | differentiable | For fixed polynomial/tensor term structure. |
| `first_order_nonlinearity` | differentiable | For intrusive polynomial/tensor terms; non-intrusive callables must be JAX-transformable. |
| `second_order_internal_force` | differentiable | For intrusive polynomial/tensor terms; non-intrusive callables must be JAX-transformable. |
| `second_order_internal_force_jacobian_x` | differentiable | For displacement-only intrusive nonlinearities. |
| `second_order_internal_force_jacobian_xd` | differentiable | Constant zero map for displacement-only nonlinearities. |
| `first_order_from_second_order_nonlinearity` | differentiable | Algebraic first-order embedding `[-fnl; 0]`. |
| `evaluate_periodic_forcing` | differentiable | For fixed Fourier/Taylor forcing structure. |
| `forcing_kappas` | not differentiable | Harmonic metadata extraction. |
| `first_order_forcing_terms_from_second_order` | differentiable | Coefficient padding for fixed forcing structure; tested with `jax.grad`. |
| `infer_callable_input_dim` | not differentiable | Probes Python callables using exception-based shape checks. |
| `infer_semi_intrusive_input_dim` | not differentiable | Probes multilinear/semi-intrusive callables using exception-based shape checks. |
| `evaluate_first_order_vector_field` | differentiable under nondegeneracy assumptions | Requires nonsingular `B` and differentiable nonlinear/forcing terms. |
| `second_order_residual` | piecewise differentiable | Residual/tangent terms are differentiable; norm-based `c0` is non-smooth at zero norms. |
| `mechanical_binv_a` | differentiable under nondegeneracy assumptions | Requires nonsingular mass matrix. |
| `mechanical_a_matrix` | differentiable | Algebraic mechanical first-order `A` matrix assembly. |
| `mechanical_b_matrix` | differentiable | Algebraic mechanical first-order `B` matrix assembly. |
| `polynomial_input_dim` | not differentiable | Shape metadata inference. |
| `polynomial_degree` | not differentiable | Shape metadata inference. |
| `first_order_polynomial_terms_from_second_order` | differentiable | Coefficient embedding for fixed term structure. |
| `first_order_tensor_terms_from_second_order` | differentiable | Dense tensor embedding for fixed shapes. |
| `first_order_terms_from_second_order` | differentiable | Dispatches to fixed-structure polynomial/tensor embedding. |
| `NonAutonomousResonanceData` | not differentiable | Metadata container for non-autonomous resonance detection. |
| `NonAutonomousCoefficientSeries` | not differentiable | Metadata container for one harmonic's non-autonomous coefficient series. |
| `NonAutonomousStructure` | not differentiable | Metadata container initialized by `nonautonomous_struct_setup`. |
| `NonAutonomousFirstOrderData` | not differentiable | Static data container for non-autonomous first-order solves. |
| `NonAutonomousLeadResult` | not differentiable | Result container. |
| `NonAutonomousSolveResult` | not differentiable | Result container. |
| `IntrusiveCompositionData` | not differentiable | Static data container for intrusive multi-index composition. |
| `ResonanceSide` | not differentiable | Metadata container for one side of SSM resonance analysis. |
| `ResonanceAnalysis` | not differentiable | Metadata container for inner/outer resonance analysis. |
| `MasterSubspace` | not differentiable | Metadata container for selected modal subspace and resonance results. |
| `check_ds_type` | not differentiable | Discrete dtype/metadata decision matching Manifold DS-type selection. |
| `check_comp_type` | not differentiable | Discrete algorithm-selection rule for first- versus second-order Manifold computation. |
| `resonance_analysis` | not differentiable | Integer combination enumeration, spectral quotient truncation, and thresholded resonance selection. |
| `choose_master_subspace` | not differentiable | Discrete mode selection and resonance analysis on supplied spectral data. |
| `autonomous_resonant_terms` | not differentiable | Thresholded resonance-index selection; returns zero-based index arrays in MATLAB `find` order. |
| `AutonomousFirstOrderData` | not differentiable | Static data container for autonomous first-order cohomological solves. |
| `AutonomousSecondOrderData` | not differentiable | Static data container for autonomous second-order cohomological solves. |
| `AutonomousSSMSolveResult` | not differentiable | Result container; numeric fields inherit solve-kernel differentiability. |
| `autonomous_first_order_reduced_dynamics` | differentiable for fixed resonance pattern | Projects resonant RHS columns into reduced dynamics and subtracts `B*V_M*R`; tested with `jax.grad`. Resonance selection is thresholded/discrete. |
| `autonomous_first_order_ssm` | differentiable under nondegeneracy assumptions | One-order first-order SSM solve; assumes fixed resonance pattern and nonsingular cohomological matrices. |
| `autonomous_second_order_reduced_dynamics` | differentiable under nondegeneracy assumptions | Single-resonance second-order reduced-dynamics projection; 1:1 resonance raises `NotImplementedError`. |
| `autonomous_second_order_ssm` | differentiable under nondegeneracy assumptions | Analytic one-order second-order SSM solve; tested with `jax.jacfwd`, assumes fixed resonance pattern and nonsingular dynamic stiffness matrices. |
| `nonautonomous_resonant_terms` | not differentiable | Enumerates integer multi-indices and threshold-selects resonances; returns zero-based indices. |
| `nonautonomous_conjugate_reduction` | not differentiable | Exact harmonic matching and norm-thresholded conjugacy detection with discrete index-map output. |
| `nonautonomous_struct_setup` | not differentiable | Initializes discrete non-autonomous coefficient containers and harmonic metadata. |
| `nonautonomous_assemble_coefficients` | differentiable for fixed index structure | Immutable insertion of solved non-autonomous coefficients and row-wise multi-index metadata; tested with `jax.grad`. |
| `nonautonomous_zeroth_order_forcing` | not differentiable | Active harmonic selection by exact-zero testing. |
| `nonautonomous_first_order_lead_terms` | differentiable under fixed active/resonance structure | Leading non-autonomous first-order solve; tested with `jax.grad`. Active harmonic selection, conjugate reduction, and resonance detection are discrete. |
| `nonautonomous_first_order_solve_invariance` | differentiable under nondegeneracy assumptions | Per-harmonic/per-order first-order invariance solve; tested with `jax.jacfwd`, assumes fixed resonance pattern and nonsingular coefficient matrices. |
| `NonAutonomousSecondOrderData` | not differentiable | Static data container for non-autonomous second-order solves. |
| `nonautonomous_second_order_reduced_dynamics` | differentiable under fixed resonance pattern | Second-order resonant projection; tested with source-derived cases. Resonance grouping is discrete. |
| `nonautonomous_second_order_solve_invariance` | differentiable under nondegeneracy assumptions | Per-harmonic/per-order second-order invariance solve; tested with `jax.jacfwd` on real force perturbations, assumes fixed resonance pattern and nonsingular dynamic stiffness matrices. |
| `nonautonomous_forcing_plus_nonlinearity` | differentiable for fixed harmonic structure | Summation core for non-autonomous forcing plus nonlinear/Jacobian composition; tested with `jax.grad`. |
| `nonautonomous_w1r0_plus_w0r1` | differentiable for fixed index structure | Non-autonomous mixed product `W1 R0 + W0 R1`; tested with `jax.grad`. |
| `step_polynomial` | differentiable for fixed order | MATLAB `StEP` polarization for orders 1-3; differentiability depends on the supplied JAX-transformable callable. Tested with `jax.grad`. |
| `fnl_intrusive` | differentiable for fixed index structure | Intrusive multi-index composition core; tested with `jax.grad`. Conjugate-order optimized branch is not ported. |
| `fnl_nonintrusive` | differentiable for fixed index structure | Revlex non-intrusive Manifold force composition; uses complex polarization internally. Tested with `jax.grad`. |
| `fnl_semi_intrusive` | differentiable for fixed index structure | Revlex semi-intrusive Manifold multilinear force composition; tested with `jax.jacfwd`. |
| `dfnl_intrusive` | differentiable for fixed index structure | Intrusive Jacobian-action multi-index composition; tested with `jax.jacfwd`. Conjugate-order optimized branch is not ported. |
| `dfnl_nonintrusive` | differentiable for fixed index structure | Revlex non-autonomous Jacobian composition; uses `step_polynomial` on the Jacobian callable. Tested with `jax.grad`. |
| `dfnl_semi_intrusive` | differentiable for fixed index structure | Revlex non-autonomous semi-intrusive Jacobian action; tested with `jax.jacfwd`. |
| `autonomous_invariance_residual` | piecewise differentiable | Autonomous invariance residual norm; supports complex reduced coordinates via explicit polynomial derivatives. Non-smooth at zero residual. Tested with `jax.grad` away from zero residual. |
| `compute_auto_invariance_error` | piecewise differentiable | 2D/4D autonomous residual sampling average for fixed sampling grids; inherits norm non-smoothness. |
| `coeffs_composition` | differentiable for fixed index structure | Lex/revlex coefficient composition; conjugate branch not yet ported. Tested with `jax.grad`. |
| `coeffs_mixed_terms` | differentiable for fixed index structure | Lex/revlex mixed coefficient products; conjugate branch not yet ported. |
| `SSMContinuationInfo` | not differentiable | Metadata container parsed from MATLAB `.mat` solution payloads. |
| `NumericalIntegrationSummary` | not differentiable | File-reader result container. |
| `PeriodicOrbitInitialConditions` | not differentiable | File-reader result container. |
| `coco_fname` | not differentiable | Filesystem lookup helper matching COCO run/data directory conventions. |
| `read_ssm_ep_solution` | not differentiable | MATLAB `.mat` file reader; converts external persisted data to Python/JAX arrays. |
| `read_ssm_po_solution` | not differentiable | MATLAB `.mat` file reader; supports SSM FRC payloads, not external COCO `po_read_solution` data. |
| `read_ssm_tor_solution` | not differentiable | MATLAB `.mat` file reader; supports SSM FRC payloads, not external COCO `tor_read_solution` data. |
| `read_num_int_sol` | not differentiable | File aggregation and norm/amplitude extraction over saved numerical-integration results. |
| `read_po_ssm_init` | not differentiable | File aggregation for saved SSM periodic-orbit initial conditions. |
| `DSOptions` | not differentiable | Configuration container. |
| `ManifoldOptions` | not differentiable | Configuration container. |
| `FRCOptions` | not differentiable | Configuration container. |
| `FRSOptions` | not differentiable | Configuration container. |
| `matlab_style_options` | not differentiable | Metadata conversion to MATLAB-style field names. |
