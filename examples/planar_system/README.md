# PlanarSystem

Source workflow:

- `SSMTool/examples/PlanarSystem/build_model.m`
- `SSMTool/examples/PlanarSystem/demo.mlx`

Fidelity status: `partial`

This example reproduces the tested graph SSM coefficient subproblem for the
two-dimensional polynomial system. The MATLAB live script does not include a
figure. Full fidelity is blocked by the unported MATLAB object workflow
(`DynamicalSystem`, `SSM`, `choose_E`, and `compute_whisker`), but the
source-stated analytical graph coefficients are computed by the tested
homological-equation core.

Run:

```bash
python examples/planar_system/example.py
```

The notebook `planar_system.ipynb` uses the same tested numerical core.

## Fidelity checklist

| MATLAB section/cell | MATLAB operation | expected numerical output | expected plot or visual output | Python implementation status | Python test coverage | plot/visualization status | discrepancies | next fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intro equation | Defines `xdot=-x`, `ydot=-sqrt(24)y+x^2+x^3+x^4+x^5` and graph `y=h(x)` | Analytical coefficients `a_k=1/(sqrt(24)-k)` for `k=2..5`, zero for `k>=6` | None | Implemented via source-derived model and graph coefficient computation | `tests/test_planar_system.py`, `tests/test_core_graph_solver.py` | Not applicable | None for the coefficient formula | None |
| `clear all; [A,B,F] = build_model();` | Builds first-order model matrices/tensors | `A=diag(-1,-sqrt(24))`, `B=I`, nonlinear powers `x^2..x^5` in second equation | None | Implemented in `examples/planar_system/planar.py` | `test_build_planar_system_matches_matlab_build_model`, sparse-term and vector-field tests | Not applicable | MATLAB tensor object not reproduced; source-derived sparse representation used | Port only if a future example requires full tensor object compatibility |
| Dynamical system setup | Creates `DynamicalSystem`, sets `A`, `B`, `fnl`, and options | Configured MATLAB object | None | Not ported; bypassed by minimal numerical core | Not covered | Not applicable | Full class workflow missing | Implement reusable system/config object only when needed by a substantive example |
| Linear modal analysis | Calls `DS.linear_spectral_analysis()` | Modal data for first master mode | None | Bypassed; eigenstructure is trivial for the selected scalar subproblem | Coefficient tests indirectly cover selected eigenvalues | Not applicable | Full modal API missing | Add fixed-choice modal helper when needed by next examples |
| SSM graph computation | Creates `SSM`, selects `masterModes=1`, `order=8`, computes whisker | `W_0` coefficients matching analytical `a_2..a_5` | None | Implemented as fixed-choice scalar homological solve | Coefficient regression, invariance residual, JAX transform tests | Not applicable | Full `compute_whisker` object output not reproduced | Largest remaining gap: full MATLAB class-stack SSM workflow |
