# BenchamrkSSM1stOrder

Source workflow:

- `SSMTool/examples/BenchamrkSSM1stOrder/build_model.m`
- `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`

Fidelity status: `partial`

The MATLAB directory name is intentionally misspelled upstream. This workflow
is reproduced as a source-confirmed duplicate of `PlanarSystem`; its
example-local coefficient helper calls reusable core kernels from
`src/ssmtoolpy/core/`.

Run:

```bash
python examples/benchmark_ssm_1st_order/example.py
```

The notebook `benchmark_ssm_1st_order.ipynb` mirrors the coefficient
comparison from the MATLAB live script.

## Fidelity checklist

| MATLAB section/cell | MATLAB operation | expected numerical output | expected plot or visual output | Python implementation status | Python test coverage | plot/visualization status | discrepancies | next fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Intro equation | Defines same 2D polynomial system as PlanarSystem | Analytical coefficients `a_k=1/(sqrt(24)-k)` for `k=2..5`, zero for `k>=6` | None | Implemented as named benchmark coefficient computation | `tests/test_benchmark_ssm_1st_order.py`, shared core tests | Not applicable | None for analytical coefficient check | None |
| `clear all; [A,B,F] = build_model();` | Builds first-order model | Same source as PlanarSystem after whitespace normalization | None | Implemented through duplicate helper and source-equivalence tests | `test_benchmark_build_model_matches_planar_system_source` | Not applicable | Full MATLAB tensor object not reproduced | Defer until full class workflow exists |
| Dynamical system setup | Creates `DynamicalSystem`, sets `A`, `B`, `fnl` | Configured MATLAB object | None | Not ported; bypassed by fixed-choice numerical core | Not covered | Not applicable | Full class workflow missing | Implement reusable class-free setup only when required by a future example |
| Linear modal analysis and SSM computation | Calls `linear_spectral_analysis`, `SSM`, `choose_E`, `compute_whisker(order=8)` | Coefficient matrix `coeffs` with active second-row entries at orders 2..5 | None | Active coefficients reproduced by scalar graph solver | Coefficient and JAX transform tests | Not applicable | Full `W0/R0` object layout not reproduced | Largest remaining gap: full MATLAB class-stack SSM workflow |
| Coefficient comparison | Computes `coeffs(2,2:5)' - analytical` | Zero vector | None | Implemented in script and notebook as zero difference | `test_benchmark_coefficients_match_planar_solver_regression` | Not applicable | None for tested comparison | None |
