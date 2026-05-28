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
| `frc_ab` | differentiable | Tested with `jax.grad` and `jax.vmap`. |
| `compute_gamma` | not differentiable | Discrete multi-index lookup into reduced-dynamics coefficients. |
| `frc_psi` | piecewise differentiable | Uses `atan2`; tested with `jax.grad` and `jax.vmap`, excluding branch-cut/undefined cases. |
| `frc_jacobian` | differentiable | Differentiable for `rho != 0`; tested with `jax.jacfwd`. |
| `check_stability` | not differentiable | Thresholds determinant and trace sign conditions. |
| `get_contour_xy` | not differentiable | Parses MATLAB `contourc` plotting matrices. |
| `compute_fixed_points_2d` | not differentiable | Uses sign tests, marching-squares segments, and geometric intersection selection. |
| `reduced_to_full` | differentiable | For fixed polynomial structure; tested with `jax.jit`. Non-autonomous branch uses fixed Python structure. |
| `reduced_to_full_complex` | differentiable | For fixed polynomial/forcing structure; transform coverage not yet verified. |
| `reduced_to_full_traj` | differentiable | Single-time reconstruction; tested with `jax.jit` and `jax.jacfwd` for autonomous structure. |
| `extract_output` | piecewise differentiable | Norm outputs are differentiable away from zero/ties; amplitudes use infinity norms. |
| `spblkdiag` | differentiable | Dense block diagonal assembly; tested with `jax.jacfwd`. |
| `solve_invariance_equation` | differentiable under nondegeneracy assumptions | Direct solve requires nonsingular matrices; `pinv`/`lsqminnorm` assume stable rank; iterative names use `jax.scipy.sparse.linalg` implicit linear solves. |
| `auto_red_dyn` | differentiable | For fixed integer exponent matrix; tested with `jax.grad`. |
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
| `coeffs_composition` | differentiable for fixed index structure | Lex/revlex coefficient composition; conjugate branch not yet ported. Tested with `jax.grad`. |
| `coeffs_mixed_terms` | differentiable for fixed index structure | Lex/revlex mixed coefficient products; conjugate branch not yet ported. |
| `DSOptions` | not differentiable | Configuration container. |
| `ManifoldOptions` | not differentiable | Configuration container. |
| `FRCOptions` | not differentiable | Configuration container. |
| `FRSOptions` | not differentiable | Configuration container. |
| `matlab_style_options` | not differentiable | Metadata conversion to MATLAB-style field names. |
