# PlanarSystem

Source workflow:

- `SSMTool/examples/PlanarSystem/build_model.m`
- `SSMTool/examples/PlanarSystem/demo.mlx`

This example reproduces the tested graph SSM coefficient subproblem for the
two-dimensional polynomial system. The PlanarSystem-specific model and graph
helpers live in `examples/planar_system/planar.py`; reusable polynomial and
homological-solve kernels live under `src/ssmtoolpy/core/`.

Run:

```bash
python examples/planar_system/example.py
```

The notebook `planar_system.ipynb` uses the same tested example-local helper.
