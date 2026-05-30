# Lorenz1stOrder

Source workflow:

- `SSMTool/examples/Lorenz1stOrder/build_model.m`
- `SSMTool/examples/Lorenz1stOrder/lorenz.m`
- `SSMTool/examples/Lorenz1stOrder/demo.mlx`

This example reproduces the tested fixed-choice Python/JAX workflow
corresponding to the MATLAB live script: source model, Lorenz vector field,
standard-parameter eigenvalues, unstable SSM graph coefficients through order
5, linear reduced dynamics, reduced-to-full lifting, direct full trajectory
simulation, reduced/full comparison, and the notebook SSM/full trajectory
visualization.

Run:

```bash
python examples/lorenz_1st_order/example.py
```

The notebook `lorenz_1st_order.ipynb` uses the same tested numerical API.
