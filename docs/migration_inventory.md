# Migration Inventory

## MATLAB Core

| Path | Role | Dependency status | Planned Python destination | Migration status | Test status | Differentiability |
| --- | --- | --- | --- | --- | --- | --- |
| `SSMTool/src/@DynamicalSystem/*.m` | core | Needed by full examples; bypassed for first subproblem | `src/ssmtoolpy/systems/`, later `core/` | deferred | not tested | not yet verified |
| `SSMTool/src/@Manifold/*.m` | core | Needed by full SSM coefficient computation; scalar graph form inspected for PlanarSystem | `src/ssmtoolpy/core/invariance.py` and related modules | minimal scalar graph slice implemented | tested for PlanarSystem slice | differentiable under nonresonance for implemented slice |
| `SSMTool/src/@SSM/*.m` | core/continuation | Needed by full SSM workflows and FRC/FRS | later algorithms and continuation modules | deferred | not tested | mixed; not yet verified |
| `SSMTool/src/frc/*.m` | core/FRS utility | Needed by forced response workflows | later `core/` or `algorithms/` | deferred | not tested | not yet verified |
| `SSMTool/src/misc/*.m` | utility/plotting/core mix | Needed selectively by examples | case-by-case | deferred | not tested | mixed; not yet verified |
| `SSMTool/ext/**` | third-party/external | Deferred unless a target requires it | none planned yet | deferred | not tested | external |

## Example and Workflow Inventory

| Workflow | MATLAB files | Estimated complexity | Usefulness | Planned Python destination | Planned notebook | Status |
| --- | --- | --- | --- | --- | --- | --- |
| PlanarSystem | `build_model.m`, `demo.mlx` | very low | high first regression | `src/ssmtoolpy/systems/planar.py`, `examples/planar_system.py` | `notebooks/planar_system.ipynb` | solver-derived coefficient subproblem implemented |
| BenchamrkSSM1stOrder | `build_model.m`, `demo.mlx` | very low | confirms PlanarSystem duplicate workflow | reuses PlanarSystem module, `examples/benchmark_ssm_1st_order.py` | `notebooks/benchmark_ssm_1st_order.ipynb` | reproduced as source-confirmed duplicate |
| Lorenz1stOrder | `build_model.m`, `lorenz.m`, `demo.mlx` | low-medium | canonical first-order nonlinear system | `systems/lorenz.py` | later notebook | deferred |
| TwoOscillators | `build_model.m`, `demo.mlx`, `demoSymbolicExpression.mlx` | medium | small second-order oscillator with forcing | later systems module | later notebooks | deferred |
| ThreeOscillators | `build_model.m`, `ThreeOscillators.m`, `ThreeOscillatorsBook.mlx` | medium | low-dimensional oscillator regression | later systems module | later notebook | deferred |
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

## Active Target Details

| MATLAB file | Role | Public behavior used | Python destination | Status | Test status | Differentiability |
| --- | --- | --- | --- | --- | --- | --- |
| `SSMTool/examples/PlanarSystem/build_model.m` | example model | `A`, `B`, and polynomial terms `x^2..x^5` in second equation | `build_planar_system`, `planar_vector_field` | implemented | tested | differentiable |
| `SSMTool/examples/PlanarSystem/demo.mlx` | `.mlx` workflow | closed-form graph coefficients and graph parameterization | `planar_ssm_graph_coefficients`, `evaluate_planar_ssm_graph` | implemented as subproblem | tested | differentiable under nondegeneracy / differentiable |
| `SSMTool/examples/BenchamrkSSM1stOrder/build_model.m` | example model | same `A`, `B`, and polynomial terms as PlanarSystem | reuse `src/ssmtoolpy/systems/planar.py` | reproduced as duplicate | tested | differentiable through reused PlanarSystem functions |
| `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx` | `.mlx` workflow | analytical coefficient comparison `coeffs(2,2:5)` | `examples/benchmark_ssm_1st_order.py`, `notebooks/benchmark_ssm_1st_order.ipynb` | reproduced as duplicate | tested | differentiable through reused PlanarSystem functions |
| `SSMTool/src/@Manifold/private/Aut_1stOrder_SSM.m` | core | nonresonant solve `(B*K_Lambda - A) W_k = RHS` | `solve_scalar_graph_coefficients` | scalar graph slice implemented | tested | differentiable under nonresonance |
| `SSMTool/src/@Manifold/private/Aut_1stOrder_RedDyn.m` | core | resonance detection before coefficient solve | no public API yet | nonresonant PlanarSystem case documented only | not directly tested | not yet verified |
| `SSMTool/src/@Manifold/private/coeffs_setup.m` | core | master eigenvalue and multi-index setup | `multiindices_of_total_degree` | tiny index slice implemented | tested | not differentiable |
| `SSMTool/src/@Manifold/private/multi_nsumk.m` | core utility | small nonnegative multi-index combinations | `multiindices_of_total_degree` | tiny total-degree slice implemented | tested | not differentiable |

## Parameter-to-Loss Path Classification

| Python module/function | Path role | Classification | Current test coverage |
| --- | --- | --- | --- |
| `src/ssmtoolpy/systems/planar.py::planar_ssm_graph_coefficients` | fixed SSM reduction phase for PlanarSystem scalar graph | differentiable under nonresonance assumptions | coefficient tests, JAX grad tests, parameter-to-loss smoke |
| `src/ssmtoolpy/systems/planar.py::evaluate_planar_ssm_graph` | reduced prediction phase for PlanarSystem scalar graph | differentiable | graph evaluation tests, JAX jit tests, parameter-to-loss smoke |
| `src/ssmtoolpy/core/invariance.py::solve_scalar_graph_coefficients` | differentiable homological-equation solve for fixed one-master/one-transverse structure | differentiable under nonresonance assumptions | solver tests, JAX grad tests |
| `src/ssmtoolpy/core/polynomial.py::evaluate_monomial_polynomial` | differentiable vector-field/polynomial evaluation | differentiable for fixed exponents | polynomial tests, JAX grad tests |
| `src/ssmtoolpy/core/polynomial.py::collect_univariate_coefficients` | coefficient assembly for fixed sparse terms/order | differentiable with respect to coefficients; setup choices fixed | coefficient collection tests, JAX jacobian tests |
| `src/ssmtoolpy/core/multiindex.py::multiindices_of_total_degree` | setup/classification only | not differentiable | index-generation tests |
| `examples/*.py` | presentation and smoke execution | not part of differentiable core | example execution checks |
| `notebooks/*.ipynb` | presentation | not part of differentiable core | numerical core covered by pytest |
| `SSMTool/src/@SSM/*`, `SSMTool/src/@Manifold/*` not yet ported | future full SSM construction | mixed; expected split between setup-only and differentiable fixed-choice solves | not yet verified |
