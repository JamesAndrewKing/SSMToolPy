# Project instructions for Codex

This repository contains a MATLAB implementation in `SSMTool/`. The goal is to build a Python/JAX implementation that is numerically faithful, fully differentiable where mathematically possible, tested, documented, and maintainable.

## General working mode

Work autonomously. Do not stop to ask for confirmation unless:
1. A required input is genuinely unavailable from the repository.
2. Continuing would require deleting user data, changing external services, or making an irreversible action.
3. There are two or more incompatible interpretations that would materially change the public API.

Otherwise, make a reasonable engineering decision, record it in documentation, and continue.

Always inspect the repository before implementing. Prefer incremental commits or clearly separated change sets if the environment supports that.

## Target Python stack

Use:
- Python 3.11+
- JAX and `jax.numpy` for differentiable numerical code
- `pytest` for tests
- `numpy` only for non-differentiated compatibility utilities or test fixtures
- `scipy` only where JAX has no suitable equivalent, and never inside functions that are required to be differentiable
- `jaxtyping` or type annotations where useful
- `ruff` or similar formatting/linting if already present, otherwise keep style simple and PEP 8 compliant

## Differentiability requirements

The Python implementation must be differentiable with JAX wherever the underlying mathematical operation is differentiable.

Avoid:
- In-place mutation
- Python control flow depending on traced array values
- NumPy operations inside differentiable functions
- Object arrays
- Side effects inside JAX-transformable functions
- Non-JAX linear algebra inside differentiable paths

Support, where feasible:
- `jax.jit`
- `jax.grad`
- `jax.jacfwd`
- `jax.jacrev`
- `jax.vmap`

Where exact differentiability is impossible or mathematically ill-defined, document the limitation and add tests that verify the expected behavior.

Differentiability means JAX-transformable where the mathematical map is differentiable and where the algorithm admits stable automatic differentiation. For algorithms involving branch selection, sorting, rank decisions, event detection, convergence failure, eigenspace degeneracy, root selection, or discontinuous normalization conventions, classify the differentiability status explicitly:
- differentiable,
- piecewise differentiable,
- differentiable only under nondegeneracy assumptions,
- not differentiable,
- not yet verified.

Do not claim full differentiability without tests using `jax.grad`, `jax.jacfwd`, or `jax.jacrev` on representative inputs.

Every public Python function/class must have a documented differentiability status:
- differentiable,
- piecewise differentiable,
- differentiable only under stated nondegeneracy assumptions,
- not differentiable,
- not yet verified.

A function may only be marked differentiable if at least one representative test exercises it with `jax.grad`, `jax.jacfwd`, or `jax.jacrev`, as appropriate for its input/output shape.

## MATLAB migration rules

Treat `SSMTool/` as the immutable MATLAB reference implementation. Do not rewrite, reorganize, delete, or translate files in place inside `SSMTool/`. Build the Python/JAX implementation outside it, preferably under `src/ssmtoolpy/`. Put Python tests under `tests/`, documentation under `docs/`, and package configuration at the repository root.

For each MATLAB file in `SSMTool/`:
1. Identify the public API, inputs, outputs, shapes, and numerical assumptions.
2. Port the behavior to Python/JAX.
3. Preserve mathematical semantics over line-by-line syntax.
4. Add docstrings explaining the mathematical operation.
5. Add or transfer tests.
6. Compare against MATLAB fixtures if available.
7. If MATLAB execution is unavailable, derive deterministic reference cases from the source and document the limitation.

Separate core implementation from examples, demos, tutorials, and large regression scripts. First port the dependency-free and low-level numerical kernels, then higher-level APIs, then examples. Use examples as regression fixtures only when they clarify expected behavior or cover important workflows.

Before porting substantial functionality, create or update `docs/migration_inventory.md` listing:
- each MATLAB file,
- whether it is core code, test, example, demo, utility, or generated/irrelevant,
- dependencies inferred from calls/imports,
- planned Python destination,
- migration status,
- differentiability classification if it exposes public numerical behavior.

## Testing requirements

Transfer all relevant MATLAB tests into `pytest`.

Tests must cover:
- Basic correctness
- Shape behavior
- Edge cases
- Numerical tolerances
- JAX transformations: at least `jit` and one derivative test for differentiable functions
- Regression fixtures for representative SSMTool examples

Use strict tolerances where reasonable, but adapt tolerances for floating-point differences between MATLAB and JAX.

Do not mark tests as passing by weakening assertions without justification. If a test cannot be transferred, document why.

## Documentation requirements

Create or update:
- `README.md` with installation, quickstart, examples, and testing instructions
- API documentation in docstrings
- A migration report, e.g. `docs/migration_report.md`, mapping MATLAB files to Python modules
- A known limitations section
- A differentiability section explaining which APIs support JAX transforms

## Completion criteria

The task is complete only when:
1. MATLAB source structure has been inspected.
2. A Python package exists.
3. Core MATLAB functionality has corresponding Python/JAX implementation.
4. Tests have been transferred or accounted for.
5. The test suite runs.
6. Failures have been debugged as far as possible.
7. Documentation is written.
8. A final summary lists implemented modules, skipped items, known limitations, and exact commands run.

Continue iterating until the completion criteria are met or until blocked by missing information that cannot be inferred from the repository.

If MATLAB or Octave is available, use it only to generate reference fixtures and behavioral comparisons. If neither is available, do not block. Instead derive small deterministic reference cases from the MATLAB source, document that they are source-derived, and continue.