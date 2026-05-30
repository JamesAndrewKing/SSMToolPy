# Lorenz1stOrder

Source workflow:

- `SSMTool/examples/Lorenz1stOrder/build_model.m`
- `SSMTool/examples/Lorenz1stOrder/lorenz.m`
- `SSMTool/examples/Lorenz1stOrder/demo.mlx`

Fidelity status: `plot-incomplete`

This example reproduces the tested fixed-choice Python/JAX workflow
corresponding to the MATLAB live script: source model, Lorenz vector field,
standard-parameter eigenvalues, unstable SSM graph coefficients through order
5, linear reduced dynamics, reduced-to-full lifting, direct full trajectory
simulation, reduced/full comparison, and the notebook SSM/full trajectory
visualization.

The numerical and visual workflow is close to the MATLAB intent, but it is not
marked `complete` because the full MATLAB `DynamicalSystem`/`SSM` object
workflow and `ode45` adaptive trajectory sampling are not reproduced exactly.
The notebook now uses the same `t = linspace(0,1,100)` SSM visualization grid
as the MATLAB live script.

Run:

```bash
python examples/lorenz_1st_order/example.py
```

The notebook `lorenz_1st_order.ipynb` uses the same tested numerical API.

## Fidelity checklist

| MATLAB section/cell | MATLAB operation | expected numerical output | expected plot or visual output | Python implementation status | Python test coverage | plot/visualization status | discrepancies | next fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intro equation and parameters | Defines Lorenz system with `sigma=10`, `rho=28`, `beta=8/3`; states eigenvalues near `[-22.828, -2.667, 11.828]` | Standard parameter tuple and eigenvalue regression | None | Implemented in `standard_lorenz_parameters`, `build_lorenz_system`, `lorenz_linear_eigenvalues` | Model, sparse term, vector-field, and eigenvalue tests | Not applicable | None for source-derived values | None |
| `build_model` and `lorenz.m` | Builds `B=I`, `A`, nonlinear terms `[0,-xz,xy]`; evaluates vector field | Vector field values matching MATLAB formula | None | Implemented as example-local Lorenz model/vector field | Model and vector-field regression plus JAX transform tests | Not applicable | MATLAB tensor object not reproduced | Defer full tensor object until needed |
| Dynamical system setup | Creates `DynamicalSystem`, sets `A`, `B`, `fnl` | Configured MATLAB object | None | Not ported; bypassed by fixed-choice numerical core | Not covered directly | Not applicable | Full class workflow missing | Implement class-free reusable system setup only when required by a future example |
| Linear modal analysis | Calls `DS.linear_spectral_analysis()` and selects `masterModes=1` | Unstable eigenvalue/eigenvector for one-dimensional SSM | None | Implemented as deterministic setup helper `lorenz_unstable_eigenpair` | Eigenpair regression tests | Not applicable | Not differentiable through mode selection/sign normalization | Keep setup outside differentiable path; add generic modal helper when reused |
| SSM computation | `SSM(DS)`, graph style, `reltol=0.5`, `order=5`, `compute_whisker` | Unstable graph coefficients `W0` and linear reduced dynamics | None | Fixed-choice quadratic graph coefficient solve through order 5 | Coefficient shape, invariance residual, JAX grad tests | Not applicable | Full adaptive `compute_whisker` and `R0` object layout not reproduced | Largest numerical gap: general first-order autonomous SSM coefficient workflow |
| Reduced dynamics | States no inner resonances; `pdot=lambda_1 p` | `p(t)=p0 exp(lambda_1 t)` | None | Implemented by reusable `linear_reduced_trajectory` | Core graph tests and Lorenz trajectory tests | Not applicable | Nonlinear/reduced `R0` object not represented because no inner resonances are used | None for this live-script path |
| SSM visualization | Uses `t=linspace(0,1,100)`, `p1=1e-4 exp(lambda t)`, `p2=-p1`, `reduced_to_full_traj`, concatenates negative branch reversed, plots blue 3D curve | Two-sided SSM curve in `(x,y,z)` coordinates | `plot3(...,'b-','LineWidth',1.5)`, labels `x,y,z`, grid, box, view `[15,35]`, axis tight | Implemented in notebook using tested arrays and matching 100-point time grid | Curve shape/order tests; notebook execution | Mostly matched | Exact MATLAB `axis tight`, box line width/font size not exactly reproduced | Add reusable/example plotting helper if more 3D SSM plots appear |
| Full trajectory validation | Integrates Lorenz from `±V(:,1)*1e-4` over `[0,1]` with `ode45`; concatenates negative branch reversed | Full trajectory overlaid on SSM | Red circle markers, legend `SSM`, `Full` | Implemented with fixed-step RK4 branches and red markers | RK4, branch assembly, shape, and short reduced/full comparison tests | Mostly matched | Uses fixed-step RK4 rather than adaptive `ode45`, so sampling is not identical | Add optional SciPy/ODE solver comparison outside differentiable core if fixture fidelity requires it |
