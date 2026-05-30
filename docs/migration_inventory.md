# Migration Inventory

## MATLAB Core

| Path | Role | Dependency status | Planned Python destination | Migration status | Test status | Differentiability |
| --- | --- | --- | --- | --- | --- | --- |
| `SSMTool/src/@DynamicalSystem/*.m` | core | Needed by full examples; bypassed for first subproblem | future reusable modules only if required by an accepted workflow | deferred | not tested | not yet verified |
| `SSMTool/src/@Manifold/*.m` | core | Needed by full SSM coefficient computation; scalar graph form inspected for PlanarSystem | `src/ssmtoolpy/core/invariance.py` and related modules | minimal scalar graph slice implemented | tested for PlanarSystem slice | differentiable under nonresonance for implemented slice |
| `SSMTool/src/@SSM/*.m` | core/continuation | Needed by full SSM workflows and FRC/FRS | later algorithms and continuation modules | deferred | not tested | mixed; not yet verified |
| `SSMTool/src/frc/*.m` | core/FRS utility | Needed by forced response workflows | later `core/` or `algorithms/` | deferred | not tested | not yet verified |
| `SSMTool/src/misc/*.m` | utility/plotting/core mix | Needed selectively by examples | case-by-case under `src/ssmtoolpy/core/` or example-local presentation | one-dimensional reduced-to-full and trajectory assembly slices implemented | tested for core slices and Lorenz | mixed; fixed graph/lifting helpers differentiable |
| `SSMTool/ext/**` | third-party/external | Deferred unless a target requires it | none planned yet | deferred | not tested | external |

## Example and Workflow Inventory

| Workflow | MATLAB files | Estimated complexity | Usefulness | Planned Python destination | Planned notebook | Status |
| --- | --- | --- | --- | --- | --- | --- |
| PlanarSystem | `build_model.m`, `demo.mlx` | very low | high first regression | `examples/planar_system/planar.py`, `examples/planar_system/example.py` plus reusable `src/ssmtoolpy/core/` kernels | `examples/planar_system/planar_system.ipynb` | solver-derived coefficient subproblem implemented |
| BenchamrkSSM1stOrder | `build_model.m`, `demo.mlx` | very low | confirms PlanarSystem duplicate workflow | `examples/benchmark_ssm_1st_order/benchmark.py`, `examples/benchmark_ssm_1st_order/example.py` plus reusable `src/ssmtoolpy/core/` kernels | `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb` | reproduced as source-confirmed duplicate |
| Lorenz1stOrder | `build_model.m`, `lorenz.m`, `demo.mlx` | low-medium | canonical first-order nonlinear system | `examples/lorenz_1st_order/lorenz.py`, `examples/lorenz_1st_order/example.py` plus reusable `src/ssmtoolpy/core/` kernels | `examples/lorenz_1st_order/lorenz_1st_order.ipynb` | fixed-choice live-script workflow reproduced and tested |
| TwoOscillators | `build_model.m`, `demo.mlx`, `demoSymbolicExpression.mlx` | medium | small second-order oscillator with forcing | example-local module first; reusable kernels only when shared | later notebooks | deferred |
| ThreeOscillators | `build_model.m`, `ThreeOscillators.m`, `ThreeOscillatorsBook.mlx` | medium | low-dimensional oscillator regression | example-local module first; reusable kernels only when shared | later notebook | deferred |
| TwoToOneIRs | `build_model.m`, `demo_torus_cart.mlx` | medium-high | internal resonance coverage | later | later notebook | deferred |
| OscillatorChain | `build_model.m`, `fnl.m`, `dfnl_dx.m`, `forcing.m`, workbooks | medium-high | chain model, derivatives, forcing | later | later notebooks | deferred |
| Lorenz/Charney first-order family | `CharneyDeVore1stOrder/*`, `Lorenz1stOrder/*` | medium | first-order systems beyond PlanarSystem | later | later notebooks | deferred |
| Beam examples | `BernoulliBeam*`, `TimoshenkoBeamIRs`, `PrismaticBeamStretching`, `AxialMovingBeam` | high | mechanical benchmark coverage | later | later notebooks | deferred |
| von Karman examples | `vonKarmanBeam*`, `vonKarmanPlate*`, `vonKarmanShell*` | high | nonlinear structural benchmarks | later | later notebooks | deferred |
| FRS examples | `FRS/**` | high | forced response surface workflows | later | later notebooks | deferred |
| PipeConveyingFluid | `build_model*.m`, demos, utilities | high | nonintrusive/static response coverage | later | later notebooks | deferred |
| NACAWing | `build_model*.m`, workbooks, mesh/PDF data | high | FE/nonintrusive coverage | later | later notebook | deferred |
| COMSOL examples | `COMSOL/**` | high | external-model workflows | later | later notebook | deferred |
| ParametricResonance | `ParametricResonance/**` | high | parametric forcing and stability diagrams | later | later notebooks | deferred |
| DAE examples | `DAEs/**` | high | constrained dynamics workflows | later | later notebooks | deferred |
| ComplexDyn | `build_model.m`, `compute_rho_grid.m`, `demo.mlx` | medium-high | complex dynamics visualization | later | later notebook | deferred |

## Migrated Example Fidelity Audit

| Workflow | MATLAB source | Python example path | Notebook path | Numerical fidelity status | Plot fidelity status | Missing MATLAB workflow steps | Missing plots | Tests covering numerical core |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PlanarSystem | `SSMTool/examples/PlanarSystem/build_model.m`, `SSMTool/examples/PlanarSystem/demo.mlx` | `examples/planar_system/example.py` | `examples/planar_system/planar_system.ipynb` | `partial` | not applicable; MATLAB source has no plot | full `DynamicalSystem`, `SSM`, `choose_E`, `compute_whisker`, exact `W0/R0` MATLAB object workflow | none | `tests/test_planar_system.py`, `tests/test_core_graph_solver.py`, `tests/test_parameter_to_loss.py` |
| BenchamrkSSM1stOrder | `SSMTool/examples/BenchamrkSSM1stOrder/build_model.m`, `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx` | `examples/benchmark_ssm_1st_order/example.py` | `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb` | `partial` | not applicable; MATLAB source has no plot | full `DynamicalSystem`, `SSM`, `choose_E`, `compute_whisker`, exact `W0/R0` MATLAB object workflow | none | `tests/test_benchmark_ssm_1st_order.py`, `tests/test_core_graph_solver.py`, `tests/test_core_graph.py` |
| Lorenz1stOrder | `SSMTool/examples/Lorenz1stOrder/build_model.m`, `SSMTool/examples/Lorenz1stOrder/lorenz.m`, `SSMTool/examples/Lorenz1stOrder/demo.mlx` | `examples/lorenz_1st_order/example.py` | `examples/lorenz_1st_order/lorenz_1st_order.ipynb` | `plot-incomplete` | mostly matched; 3D SSM/full plot exists with matching variables, branches, labels, legend, view, and time range; exact `ode45` sampling and MATLAB styling not exact | full adaptive `DynamicalSystem`/`SSM`/`compute_whisker` object workflow, exact `W0/R0` layout, adaptive `ode45` trajectory sampling | none missing; existing plot still has styling/sampling discrepancies | `tests/test_lorenz_1st_order.py`, `tests/test_core_graph.py`, `tests/test_core_integrators.py`, `tests/test_core_graph_solver.py` |

## Active Target Details

| MATLAB file | Role | Public behavior used | Python destination | Status | Test status | Differentiability |
| --- | --- | --- | --- | --- | --- | --- |
| `SSMTool/examples/PlanarSystem/build_model.m` | example model | `A`, `B`, and polynomial terms `x^2..x^5` in second equation | `examples/planar_system/planar.py` | implemented as example-local helper | tested | differentiable through JAX/core kernels |
| `SSMTool/examples/PlanarSystem/demo.mlx` | `.mlx` workflow | closed-form graph coefficients and graph parameterization | `examples/planar_system/planar.py` | implemented as subproblem | tested | differentiable under nondegeneracy / differentiable |
| `SSMTool/examples/BenchamrkSSM1stOrder/build_model.m` | example model | same `A`, `B`, and polynomial terms as PlanarSystem | `examples/benchmark_ssm_1st_order/benchmark.py` | reproduced as duplicate | tested | differentiable through core kernels |
| `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx` | `.mlx` workflow | analytical coefficient comparison `coeffs(2,2:5)` | `examples/benchmark_ssm_1st_order/benchmark.py`, `examples/benchmark_ssm_1st_order/example.py`, `examples/benchmark_ssm_1st_order/benchmark_ssm_1st_order.ipynb` | reproduced as duplicate | tested | differentiable through core kernels |
| `SSMTool/examples/Lorenz1stOrder/build_model.m` | example model | parameterized `A`, `B`, and quadratic terms `-xz`, `xy` | `examples/lorenz_1st_order/lorenz.py` | first subproblem implemented as example-local helper | tested | differentiable with respect to continuous parameters except integer term setup |
| `SSMTool/examples/Lorenz1stOrder/lorenz.m` | example vector field | `sigma*(y-x)`, `rho*x-y-x*z`, `-beta*z+x*y` | `examples/lorenz_1st_order/lorenz.py` | implemented as example-local helper | tested | differentiable |
| `SSMTool/examples/Lorenz1stOrder/demo.mlx` | `.mlx` workflow | standard parameters, eigenvalues, SSM graph, reduced trajectory mapping, full ODE trajectory, and 3D plot | `examples/lorenz_1st_order/example.py`, `examples/lorenz_1st_order/lorenz_1st_order.ipynb` plus reusable core kernels | fixed-choice workflow reproduced: setup/eigenvalues, graph coefficients, linear reduced trajectory, reduced-to-full lifting, direct RK4 trajectories, reduced/full comparison, and 3D visualization | tested | vector field, RK4 trajectory, fixed-choice graph solve, reduced trajectory, and lifting differentiable; eigenpair selection setup-only |
| `SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m` | core | nonresonant solve `(B*K_Lambda - A) W_k = RHS` | `solve_scalar_graph_coefficients` | scalar graph slice implemented | tested | differentiable under nonresonance |
| `SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m` | core | fixed-choice vector graph solve for autonomous first-order quadratic systems | `solve_autonomous_quadratic_graph_coefficients` | quadratic one-master vector slice implemented | tested | differentiable under fixed eigenpair/order and nonsingular homological systems |
| `SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m` | core | autonomous graph invariance equation residual for fixed choices | `univariate_graph_invariance_residual` | one-dimensional vector graph residual implemented | tested | differentiable for fixed graph structure and pure vector field |
| `SSMTool/src/@Manifold/private/Aut_1stOrder_RedDyn.m` | core | resonance detection before coefficient solve | no public API yet | nonresonant PlanarSystem case documented only | not directly tested | not yet verified |
| `SSMTool/src/@Manifold/private/coeffs_setup.m` | core | master eigenvalue and multi-index setup | `multiindices_of_total_degree` | tiny index slice implemented | tested | not differentiable |
| `SSMTool/src/@Manifold/private/multi_nsumk.m` | core utility | small nonnegative multi-index combinations | `multiindices_of_total_degree` | tiny total-degree slice implemented | tested | not differentiable |
| `SSMTool/src/misc/reduced_to_full.m` | core utility | evaluates SSM parameterization at reduced coordinates | `src/ssmtoolpy/core/graph.py::evaluate_univariate_graph` | one-dimensional graph slice implemented | tested | differentiable for fixed coefficient shape |
| `SSMTool/src/misc/reduced_to_full_traj.m` | core utility | maps reduced SSM trajectory to full coordinates in Lorenz live script | `src/ssmtoolpy/core/graph.py::evaluate_univariate_graph` | one-dimensional autonomous graph slice implemented | tested | differentiable for fixed graph coefficients |
| `SSMTool/src/misc/transient_traj_on_auto_ssm.m` | core utility | evaluates autonomous reduced trajectories then lifts them to full coordinates | `src/ssmtoolpy/core/graph.py::linear_reduced_trajectory` and `evaluate_univariate_graph` | linear one-dimensional autonomous slice implemented | tested | differentiable for fixed times and graph structure |
| `SSMTool/src/@DynamicalSystem/odefun.m` | core utility | evaluates vector field during trajectory integration | `src/ssmtoolpy/core/integrators.py::fixed_step_rk4` for fixed-step examples | fixed-step integration kernel implemented | tested | differentiable for fixed times and pure vector fields |
| `SSMTool/src/@SSM/SSM.m` and `SSMTool/src/@Manifold/compute_whisker.m` | core | object workflow used by Lorenz SSM graph computation | no object API planned; port smallest required numerical behavior | fixed-choice Lorenz coefficient slice implemented; full object workflow missing | partial tests | mixed; fixed-choice solve differentiable, adaptive choices setup-only |
| MATLAB plotting cells in `Lorenz1stOrder/demo.mlx` | presentation | `plot3` SSM curve and full trajectory comparison | notebook plotting from tested arrays | implemented in Lorenz notebook | notebook executed; numerical arrays tested | not part of differentiable core |

## Parameter-to-Loss Path Classification

| Python module/function | Path role | Classification | Current test coverage |
| --- | --- | --- | --- |
| `examples/planar_system/planar.py::planar_ssm_graph_coefficients` | fixed SSM reduction phase for PlanarSystem scalar graph | differentiable under nonresonance assumptions | coefficient tests, JAX grad tests, parameter-to-loss smoke |
| `src/ssmtoolpy/core/graph.py::evaluate_univariate_graph` used by PlanarSystem | reduced prediction phase for PlanarSystem scalar graph | differentiable | graph evaluation tests, JAX jit tests, parameter-to-loss smoke |
| `examples/benchmark_ssm_1st_order/benchmark.py::benchmark_ssm_graph_coefficients` | fixed SSM reduction phase for duplicate benchmark graph | differentiable under nonresonance assumptions | coefficient tests, JAX grad tests |
| `src/ssmtoolpy/core/invariance.py::solve_scalar_graph_coefficients` | differentiable homological-equation solve for fixed one-master/one-transverse structure | differentiable under nonresonance assumptions | solver tests, JAX grad tests |
| `src/ssmtoolpy/core/invariance.py::solve_autonomous_quadratic_graph_coefficients` | fixed-choice SSM reduction phase for one-master autonomous quadratic vector graph | differentiable under fixed choices and nonsingular homological systems | solver value tests, JAX grad tests, Lorenz residual tests |
| `src/ssmtoolpy/core/invariance.py::univariate_graph_invariance_residual` | fixed-choice graph diagnostic for autonomous one-dimensional reductions | differentiable for fixed graph structure and pure vector field | core residual tests, JAX grad tests, Lorenz residual tests |
| `src/ssmtoolpy/core/integrators.py::fixed_step_rk4` | prediction/simulation phase for pure fixed-step vector fields | differentiable for fixed time grid and pure vector-field structure | value/shape tests and JAX grad tests |
| `src/ssmtoolpy/core/graph.py::evaluate_univariate_graph` | graph evaluation and lifting phase | differentiable for fixed coefficient shape | value/shape tests |
| `src/ssmtoolpy/core/graph.py::linear_reduced_trajectory` | reduced dynamics prediction phase | differentiable for fixed times | value tests and JAX grad tests |
| `src/ssmtoolpy/core/graph.py::two_sided_graph_curve` | visualization-data assembly from reduced graph branches | differentiable for fixed times and coefficients | shape/ordering tests |
| `src/ssmtoolpy/core/trajectories.py::integrate_two_sided_branches` | visualization/validation trajectory assembly | differentiable for fixed trajectory function structure | shape/ordering tests |
| `src/ssmtoolpy/core/polynomial.py::evaluate_monomial_polynomial` | differentiable vector-field/polynomial evaluation | differentiable for fixed exponents | polynomial tests, JAX grad tests |
| `src/ssmtoolpy/core/polynomial.py::collect_univariate_coefficients` | coefficient assembly for fixed sparse terms/order | differentiable with respect to coefficients; setup choices fixed | coefficient collection tests, JAX jacobian tests |
| `src/ssmtoolpy/core/multiindex.py::multiindices_of_total_degree` | setup/classification only | not differentiable | index-generation tests |
| `examples/lorenz_1st_order/lorenz.py::build_lorenz_system` | parameterized full-system definition stage | differentiable with respect to continuous parameters | Lorenz model tests |
| `examples/lorenz_1st_order/lorenz.py::lorenz_vector_field` | parameter-to-output prediction stage before SSM reduction | differentiable | vector-field tests, JAX jacobian test, parameter-to-output smoke |
| `examples/lorenz_1st_order/lorenz.py::lorenz_rk4_trajectory` | Lorenz wrapper over reusable direct full-system trajectory computation | differentiable for fixed times | shape/reference/gradient tests plus core RK4 tests |
| `src/ssmtoolpy/core/graph.py::linear_reduced_trajectory` used by Lorenz | reusable linear reduced dynamics prediction | differentiable for fixed times | deterministic value/shape tests plus core graph tests |
| `src/ssmtoolpy/core/graph.py::evaluate_univariate_graph` used by Lorenz | reusable reduced-to-full lifting | differentiable for fixed graph coefficients | lifting shape/correctness tests plus core graph tests |
| `src/ssmtoolpy/core/graph.py::two_sided_graph_curve` used by Lorenz | reusable two-sided SSM curve assembly | differentiable for fixed times and coefficients | concatenation/shape tests plus core graph tests |
| `examples/lorenz_1st_order/lorenz.py::lorenz_full_unstable_trajectories` | Lorenz wrapper over reusable full trajectory validation assembly | differentiable for fixed times | shape/initial-condition tests plus core trajectory tests |
| `examples/lorenz_1st_order/lorenz.py::lorenz_unstable_eigenpair` | setup/classification for one-dimensional unstable SSM | not differentiable due eigenvalue selection and sign normalization | eigenpair regression tests |
| `examples/lorenz_1st_order/lorenz.py::solve_lorenz_unstable_graph_coefficients` | fixed SSM coefficient computation stage for Lorenz unstable graph | differentiable under fixed eigenpair, fixed order, and nonresonance assumptions | coefficient shape, invariance residual, and JAX grad tests |
| `examples/lorenz_1st_order/lorenz.py::lorenz_unstable_ssm_graph_coefficients` | combined setup plus fixed coefficient solve | not differentiable as a whole because setup choices are included | coefficient and residual tests |
| `src/ssmtoolpy/core/graph.py::evaluate_univariate_graph` used by Lorenz graph tests | reduced-to-full graph evaluation stage | differentiable for fixed coefficients | shape and linear-term tests |
| `examples/lorenz_1st_order/lorenz.py::lorenz_ssm_invariance_residual` | Lorenz-specific closure over reusable graph residual and Lorenz vector field | differentiable for fixed coefficient length and reduced dynamics | invariance residual tests plus core residual tests |
| `examples/lorenz_1st_order/lorenz.py::lorenz_linear_eigenvalues` | setup/diagnostic stage for modal analysis | differentiable under eigenvalue nondegeneracy but not used in differentiable workflow yet | eigenvalue regression only |
| `examples/<example_name>/example.py` | presentation and smoke execution | not part of differentiable core | example execution checks |
| `examples/<example_name>/*.ipynb` | presentation | not part of differentiable core | numerical core covered by pytest |
| `SSMTool/src/@SSM/*`, `SSMTool/src/@Manifold/*` not yet ported | future full SSM construction | mixed; expected split between setup-only and differentiable fixed-choice solves | not yet verified |
