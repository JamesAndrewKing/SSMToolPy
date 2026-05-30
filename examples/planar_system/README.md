# PlanarSystem

Source workflow:

- `SSMTool/examples/PlanarSystem/build_model.m`
- `SSMTool/examples/PlanarSystem/demo.mlx`

This example reproduces the tested graph SSM coefficient subproblem for the
two-dimensional polynomial system. The reusable numerical code lives in
`src/ssmtoolpy/systems/planar.py`; this directory contains only the executable
example and colocated notebook.

Run:

```bash
python examples/planar_system/example.py
```

The notebook `planar_system.ipynb` uses the same tested numerical API.
