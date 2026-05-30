# BenchamrkSSM1stOrder

Source workflow:

- `SSMTool/examples/BenchamrkSSM1stOrder/build_model.m`
- `SSMTool/examples/BenchamrkSSM1stOrder/demo.mlx`

The MATLAB directory name is intentionally misspelled upstream. This workflow
is reproduced as a source-confirmed duplicate of `PlanarSystem`; its
example-local coefficient helper lives in
`examples/benchmark_ssm_1st_order/benchmark.py` and calls reusable core kernels
from `src/ssmtoolpy/core/`.

Run:

```bash
python examples/benchmark_ssm_1st_order/example.py
```

The notebook `benchmark_ssm_1st_order.ipynb` mirrors the same coefficient
comparison.
