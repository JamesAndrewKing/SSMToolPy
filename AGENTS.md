# AGENT.md — SSMTool Python/JAX Port

This repository contains the original MATLAB implementation of SSMTool under `SSMTool/`.

The goal is to build a clean Python/JAX implementation that is:

1. numerically faithful to the MATLAB implementation,
2. differentiable with JAX wherever mathematically valid,
3. simple to read and maintain,
4. well documented,
5. tested,
6. able to reproduce all relevant MATLAB examples,
7. able to translate MATLAB `.mlx` example workflows into executable Jupyter notebooks.

The primary success metric is:

> Number of MATLAB SSMTool examples reproduced end-to-end in Python/JAX with tested numerical outputs and, where applicable, corresponding Jupyter notebooks.

Do not measure progress by number of files translated, lines of code written, APIs stubbed, or percentage of MATLAB code ported.

The Python port should be small, explicit, and example-driven. Prefer a limited, well-tested API over broad incomplete coverage.

---

## 1. Global operating principle

Work example-first and plan before coding.

For every substantial batch:

1. Inspect the repository.
2. Update or create the migration plan.
3. Select one bounded acceptance target.
4. Implement only what is needed for that target.
5. Add tests.
6. Run the tests.
7. Update documentation and the next-batch plan.

A batch should usually target one of the following:

1. Reproduce one MATLAB example end-to-end in Python.
2. Translate one MATLAB `.mlx` workflow into a Jupyter notebook and make its numerical core testable.
3. Complete one coherent numerical subsystem required by a target example, with correctness tests and JAX transform tests.
4. Fix a blocker that prevents an example, notebook, or required test from running.

Do not port code speculatively.

Do not create stubs, placeholder APIs, broad abstractions, or compatibility wrappers for future functionality.

Do not broaden the public API unless the new API is exercised by a test, example, or notebook.

If a MATLAB example has a large dependency closure, first define the smallest meaningful numerical subproblem from that example and make that subproblem pass before expanding.

Nothing counts as progress until at least one example, notebook, subproblem, or regression test moves from missing/failing to passing.

---

## 2. Planning-first workflow

Before implementing new functionality, perform a planning pass.

The planning pass must inspect `SSMTool/` and create or update:

```text
docs/migration_plan.md
docs/migration_inventory.md
docs/current_status.md
docs/migration_report.md
```

The planning pass must answer:

1. What MATLAB examples and `.mlx` files exist?
2. Which examples are smallest and best suited as initial acceptance targets?
3. What is the dependency closure of the next target?
4. Which MATLAB files are core numerical code, examples, demos, utilities, plotting, tests, or generated/irrelevant files?
5. Which functions are likely to be differentiable, piecewise differentiable, differentiable only under nondegeneracy assumptions, non-differentiable, or not yet verified?
6. What minimal Python package skeleton is needed?
7. What tests will prove the next target works?
8. What Jupyter notebook should eventually correspond to each `.mlx` example?

Planning is not a substitute for implementation. After the initial planning pass, implement the first bounded target unless genuinely blocked.

The plan should be updated as new information is discovered.

---

## 3. General working mode

Work autonomously. Do not stop to ask for confirmation unless:

1. A required input is genuinely unavailable from the repository.
2. Continuing would require deleting user data, changing external services, or making an irreversible action.
3. There are two or more incompatible interpretations that would materially change the public API.

Otherwise, make a reasonable engineering decision, document it, and continue.

Always inspect the repository before implementing.

At the beginning of every batch:

1. Read this `AGENT.md`.

2. Read `docs/current_status.md` if it exists.

3. Read `docs/migration_plan.md` if it exists.

4. Read `docs/migration_inventory.md` if it exists.

5. Read the `Next recommended batch` section of `docs/migration_report.md` if it exists.

6. Inspect the repository state:

   ```bash
   git status
   ```

7. If `src/`, `tests/`, and `examples/` already exist, run the current smoke checks before editing code:

   ```bash
   python -m compileall src tests examples
   python -m pytest
   ```

8. Record the current pass/fail state in `docs/current_status.md`.

At the end of every batch:

1. Run the relevant test commands.
2. Update `docs/current_status.md`.
3. Update `docs/migration_plan.md`.
4. Update `docs/migration_inventory.md`.
5. Update `docs/migration_report.md`.
6. Include a `Next recommended batch` section in `docs/migration_report.md`.

---

## 4. Target Python stack

Use:

* Python 3.11+
* JAX and `jax.numpy` for differentiable numerical code
* `pytest` for tests
* Jupyter notebooks for translated `.mlx` workflows
* `numpy` only for non-differentiated compatibility utilities, fixtures, and test reference data
* `scipy` only where JAX has no suitable equivalent, and never inside functions required to be differentiable
* `matplotlib` only for examples, notebooks, and plotting utilities, not differentiable core code
* `jaxtyping` or standard type annotations where useful
* `ruff` or similar formatting/linting if already present; otherwise use simple PEP 8 style

Prefer reusable library implementation under:

```text
src/ssmtoolpy/
```

All reusable numerical algorithms, differentiable kernels, generic ODE/trajectory utilities, SSM graph utilities, reduced-dynamics utilities, polynomial utilities, linear-algebra utilities, and parameter-to-loss utilities belong under `src/ssmtoolpy/`.

Example directories under `examples/<example_name>/` should contain only:

* model-specific definitions,
* model-specific parameters,
* model-specific vector fields or forcing definitions,
* notebook narrative,
* example scripts,
* local fixtures or reference data,
* plotting choices specific to that example,
* thin wrappers that assemble inputs and call `ssmtoolpy`.

Do not duplicate reusable numerical kernels across examples. If an operation is likely to be needed by more than one example, implement it once in `src/ssmtoolpy/` and call it from the examples.

Before adding or modifying example-local helper code, check whether the helper is actually a reusable numerical kernel. If it is reusable, implement or extend the corresponding `src/ssmtoolpy/` function first, then call it from the example. Do not implement reusable algorithms inside examples as a temporary shortcut unless there is a documented blocker.

The dependency direction is:

```text
examples/<example_name>/ -> src/ssmtoolpy/
```

Never the reverse. The `ssmtoolpy` package must not import from `examples/`.

Tests should live under:

```text
tests/
```

Examples should live under one directory per reproduced MATLAB example or workflow:

```text
examples/<example_name>/
  README.md
  example.py
  <example_name>.ipynb   # if a notebook is applicable
  fixtures/              # optional small local fixtures or assets
  helpers.py             # optional, only for one-off example-local helpers
```

Jupyter notebooks translated from `.mlx` workflows should be colocated with their corresponding example:

```text
examples/<example_name>/<example_name>.ipynb
```

Use a top-level `notebooks/` directory only with a documented reason. Do not put one-off scripts directly under `examples/`.

Documentation should live under:

```text
docs/
```

---

## 5. Architecture requirements

Keep the differentiable core functional, explicit, and array-oriented.

Prefer this separation, unless the repository has a clearly better structure:

```text
src/ssmtoolpy/
  core/
    multiindex.py
    polynomial.py
    tensors.py
    linalg.py
    invariance.py
    normal_forms.py
  systems/
    definitions.py
    vector_fields.py
  algorithms/
    solve_coefficients.py
  io/
    matlab_fixtures.py
  plotting/
```

Differentiable code must be isolated from:

* file I/O
* plotting
* logging
* printing
* MATLAB compatibility shims
* notebook display code
* object mutation
* global state
* interactive scripts
* non-JAX numerical libraries

Public convenience wrappers may be object-oriented, but the JAX-transformable numerical core should be pure functions with JAX arrays in and JAX arrays out.

Prefer simple data structures:

* small dataclasses for configuration or system definitions,
* PyTrees where useful,
* explicit arrays and shapes,
* pure functions for numerical maps.

Avoid large class hierarchies unless a reproduced example proves they are necessary.

## Core functionality versus example-specific code

The repository must avoid duplicated numerical implementations across examples.

Reusable numerical functionality belongs in `src/ssmtoolpy/`.

Examples should define only example-specific data and call reusable library functionality.

### Code that belongs in `src/ssmtoolpy/`

Put code under `src/ssmtoolpy/` when it is a reusable numerical operation, algorithm, data structure, or differentiable kernel, including:

* ODE integration methods,
* trajectory generation utilities,
* graph parameterization evaluation,
* SSM coefficient solvers,
* invariance equation solvers,
* reduced dynamics evaluation,
* lifting/reconstruction maps,
* polynomial and multi-index utilities,
* tensor utilities,
* linear algebra helpers,
* continuation or parameter-sweep utilities,
* generic plotting helpers that are useful across examples,
* parameter-to-loss workflow utilities,
* any function likely to be reused by two or more examples.

These functions should be model-agnostic. They should accept explicit arrays, callables, coefficients, modes, graph maps, time grids, and configuration objects rather than hard-coding a specific example such as Lorenz, planar systems, beams, oscillators, or forced systems.

### Code that belongs in `examples/<example_name>/`

Put code under `examples/<example_name>/` only when it is specific to that MATLAB example or `.mlx` workflow, including:

* model-specific parameters,
* model-specific vector fields,
* model-specific forcing definitions,
* model-specific initial conditions,
* model-specific plotting choices,
* notebook narrative,
* example-local fixture data,
* thin wrappers that make the example readable.

Example-local code should be small. It should mostly assemble inputs and call `ssmtoolpy`.

### Required import direction

The dependency direction is:

```text
examples/<example_name>/ -> src/ssmtoolpy/
```

Never the reverse.

`src/ssmtoolpy/` must not import from `examples/`.

### No duplicated kernels

Do not reimplement the same numerical pattern separately for each example.

Before adding a new function under `examples/<example_name>/`, check whether it is actually a reusable operation. If it is reusable, implement it once in `src/ssmtoolpy/` and call it from the example.

Examples of functions that should usually be generic:

* fixed-step RK4 trajectory integration,
* linear reduced trajectory `p(t) = p0 exp(lambda t)`,
* evaluation of a graph parameterization along reduced coordinates,
* assembling two-sided SSM graph curves,
* integrating positive and negative branches of a trajectory,
* lifting reduced trajectories to full coordinates,
* comparing full and reduced trajectories,
* computing scalar losses from predictions,
* differentiating a parameter-to-loss workflow.

Examples of functions that should usually remain example-local:

* `standard_lorenz_parameters`,
* `lorenz_vector_field`,
* `build_lorenz_system`,
* `planar_system_parameters`,
* any model-specific polynomial coefficients,
* any model-specific notebook plotting style.

### Refactoring rule

If a function starts in an example and later becomes useful for another example, move it into `src/ssmtoolpy/` before duplicating it.

When moving a function into `src/ssmtoolpy/`:

1. Rename it to a model-agnostic name.
2. Remove example-specific assumptions.
3. Add a docstring.
4. Add a differentiability classification.
5. Add or move tests to `tests/`.
6. Keep the example code as a thin caller.
7. Update documentation and migration inventory.

### Batch requirement

At the beginning of each implementation batch, inspect any newly added example-local functions. If a function is a reusable numerical kernel, move it into `src/ssmtoolpy/` before continuing to implement new example-specific functionality.

A batch should not leave reusable numerical algorithms hidden inside example directories unless there is a documented reason.


---

## 6. MATLAB migration rules

Treat `SSMTool/` as the immutable MATLAB reference implementation.

Do not rewrite, reorganize, delete, or translate files in place inside `SSMTool/`.

For each MATLAB file touched in the current batch:

1. Identify public API, inputs, outputs, shapes, and numerical assumptions.
2. Identify whether it is required by the current target example.
3. Port mathematical behavior, not line-by-line syntax.
4. Add docstrings explaining the mathematical operation.
5. Add tests.
6. Compare against MATLAB/Octave fixtures if available.
7. If MATLAB/Octave is unavailable, derive small deterministic reference cases from the MATLAB source and document that limitation.

Separate:

* core numerical code
* examples
* `.mlx` workflows
* demos
* tutorials
* plotting
* tests
* large regression scripts
* generated or irrelevant files

Do not port examples merely as scripts. Convert their numerical outputs into regression tests where possible.

Do not treat plotting equivalence as the primary acceptance criterion. Numerical reproducibility is the primary criterion; plots and notebooks should consume tested numerical outputs.

---

## 7. `.mlx` to Jupyter notebook migration

All relevant MATLAB `.mlx` example workflows should eventually have
corresponding Jupyter notebooks colocated with their example under:

```text
examples/<example_name>/
```

For each `.mlx` file:

1. Identify the mathematical/numerical workflow.
2. Identify inputs, parameters, outputs, and expected figures.
3. Separate numerical computation from presentation.
4. Implement the numerical core in `src/ssmtoolpy/`.
5. Add tests for the numerical core.
6. Create a notebook in the example directory that calls the tested Python API.
7. Keep notebooks readable and concise.
8. Avoid hiding core numerical logic inside notebooks.

Notebook migration means reproducing the MATLAB live-script workflow as closely
as reasonably possible in Python/JAX. A setup-only notebook is incomplete. For
SSMTool `.mlx` workflows, the notebook should include the meaningful sequence
present in the MATLAB source whenever applicable:

1. source model definition,
2. parameter setup,
3. SSM reduction or SSM graph computation,
4. reduced dynamics or reduced prediction,
5. trajectory computation or simulation,
6. visualization corresponding to the MATLAB output,
7. numerical checks backed by pytest tests for the computational core.

Do not skip SSM graph computation, reduced dynamics, trajectory computation, or
visualization unless there is a documented hard blocker. If a workflow is only
partially migrated, label the notebook and docs as incomplete, list the missing
MATLAB steps, and make the next batch move one missing step forward.

Notebook acceptance requires:

1. The notebook imports the package normally.
2. The notebook runs from top to bottom.
3. The meaningful numerical and visual workflow of the MATLAB `.mlx` is reproduced, or specific blockers are documented.
4. The important numerical outputs are covered by pytest tests outside the notebook.
5. Any plots are generated from tested numerical outputs.

If a notebook cannot be executed in the current environment, document the reason and ensure its numerical core is still tested.

### Valid stopping points for example and `.mlx` migration batches

For any MATLAB example, demo, or `.mlx` workflow migration, setup-only work is not a valid stopping point.

The following are not acceptable final batch endpoints unless there is a genuine, documented hard blocker:

* imports only,
* parameter definitions only,
* vector field evaluation only,
* linearization only,
* eigenvalue computation only,
* one smoke trajectory only,
* notebook scaffolding only,
* markdown placeholders only,
* documentation updates only,
* tests that cover only preliminary setup.

A migration batch should stop only after one of the following has happened:

1. A substantive MATLAB workflow step has been implemented and tested.
2. The example or notebook now executes a previously missing SSM-related computation.
3. The example or notebook now reproduces a previously missing MATLAB figure or trajectory visualization from tested numerical outputs.
4. A real technical blocker is reached and documented precisely.

For SSMTool examples, substantive workflow steps include:

* SSM graph computation,
* autonomous or non-autonomous SSM coefficient computation,
* reduced dynamics computation,
* reduced prediction,
* lifting or reconstruction from reduced coordinates,
* full-system trajectory simulation,
* reduced/full trajectory comparison,
* continuation or parameter sweep used by the original example,
* forced-response or frequency-response computation if present in the MATLAB example,
* visualization of SSM graphs, manifolds, trajectories, predictions, sweeps, or response curves,
* parameter-to-loss gradient computation.

If the current batch reaches only setup, eigenvalues, or a smoke trajectory, continue implementing the next substantive workflow step before stopping.

This rule applies to all examples, demos, and `.mlx` workflows, not only the
currently active target. Every status file and next-batch plan must classify
each reproduced example against this same substantive-workflow bar.


---

## 8. Example-first migration process

For each target example:

1. Locate the MATLAB example or `.mlx` workflow.
2. Determine its minimal MATLAB dependency closure.
3. Record the closure in `docs/migration_inventory.md`.
4. Implement the smallest Python/JAX subset needed to run the example or its first meaningful numerical subproblem.
5. Create a Python example under `examples/<example_name>/example.py`.
6. Create a Jupyter notebook under `examples/<example_name>/` if the source is a `.mlx` workflow or if an interactive workflow is useful.
7. Create a pytest regression test under `tests/`.
8. Verify deterministic numerical output.
9. Add JAX transform tests for differentiable functions introduced by the example.
10. Update documentation and status files.

A batch is successful only if at least one of the following happens:

1. One target example moves from missing/failing to passing.
2. One target notebook moves from missing/failing to runnable.
3. One minimal numerical subproblem required by a target example moves from missing/failing to passing.
4. A previously blocking failure is diagnosed and documented with a precise next step.

---

## 9. Public API discipline

Do not add a new public function, class, or module unless at least one of the following is true:

1. It is required by the current target example.
2. It is required by a test added in the current batch.
3. It is required by a target notebook.
4. It is already part of a documented API needed for a near-term example.

Every new public function/class must have:

1. A docstring.
2. A differentiability classification.
3. A correctness test.
4. A shape behavior test where applicable.
5. A JAX transform test if marked differentiable.

If a function is not yet tested, keep it private or explicitly mark it as experimental or `not yet verified`.

Avoid speculative compatibility layers.

Avoid broad wrappers around MATLAB APIs unless a reproduced example proves they are necessary.

Prefer small, composable, pure functions.

---

## 10. Differentiability requirements

The Python implementation must be differentiable with JAX wherever the underlying mathematical operation is differentiable and the algorithm admits stable automatic differentiation.

Avoid inside differentiable paths:

* in-place mutation
* Python control flow depending on traced array values
* NumPy operations
* SciPy operations
* object arrays
* side effects
* non-JAX linear algebra
* hidden global state
* data-dependent shape changes

Support where mathematically and practically feasible:

* `jax.jit`
* `jax.grad`
* `jax.jacfwd`
* `jax.jacrev`
* `jax.vmap`

Every public numerical API must document one of these differentiability statuses:

* `differentiable`
* `piecewise differentiable`
* `differentiable under nondegeneracy assumptions`
* `not differentiable`
* `not yet verified`

A function may be marked `differentiable` only if at least one representative test exercises it with `jax.grad`, `jax.jacfwd`, or `jax.jacrev`, as appropriate.

A function involving any of the following must not be marked simply `differentiable` without qualification:

* branch selection
* sorting
* rank decisions
* event detection
* convergence failure
* eigenvalue/eigenvector degeneracy
* root selection
* discontinuous normalization conventions
* index set truncation
* adaptive tolerances
* data-dependent iteration counts

Use qualified statuses such as `piecewise differentiable` or `differentiable under nondegeneracy assumptions`.

When exact differentiability is impossible or mathematically ill-defined, document the limitation and add tests verifying the expected behavior.

### Differentiable parameter-to-loss objective

The main differentiability use case is parameter optimization through the SSM reduction.

The target workflow is:

```text
system parameter p
    -> full system definition at p
    -> SSM reduction at p
    -> reduced-order prediction at p
    -> scalar loss
    -> gradient d(loss)/d p
```

The implementation should therefore prioritize differentiability of the full parameter-to-loss pipeline, not merely isolated helper functions.

A representative public workflow should eventually support:

```python
loss = loss_fn(parameter, data_or_target)
grad = jax.grad(loss_fn)(parameter, data_or_target)
```

where `loss_fn` internally constructs or updates the system at `parameter`, computes the SSM reduction, evaluates the reduced dynamics or prediction, and returns a scalar loss.

Differentiability must be tested at the workflow level whenever possible. For each representative differentiable example, add a test that verifies:

1. `loss_fn(parameter, target)` returns a finite scalar.
2. `jax.grad(loss_fn)(parameter, target)` runs without tracer errors.
3. the gradient has the expected shape.
4. the gradient is finite.
5. where feasible, the gradient agrees with a finite-difference check on a small deterministic problem.

The parameter-to-loss pipeline may be classified as `differentiable under nondegeneracy assumptions` when it depends on operations such as eigendecompositions, mode selection, linear solves, normalization conventions, truncation choices, or iterative convergence.

For such workflows, document the assumptions explicitly, for example:

* selected eigenvalues remain simple and separated,
* the chosen modal subspace does not change dimension,
* resonance or near-resonance classifications remain fixed,
* linear systems remain nonsingular or well-conditioned,
* solver convergence behavior is stable,
* normalization conventions do not switch discontinuously,
* truncation order and index sets are fixed.

Do not claim that the SSM reduction pipeline is fully differentiable unless at least one representative test differentiates through the entire parameter-to-loss workflow using `jax.grad`, `jax.jacfwd`, or `jax.jacrev`.

Where a full MATLAB-faithful algorithm contains non-differentiable choices, separate the implementation into:

1. a non-differentiable setup or classification phase, and
2. a differentiable numerical phase with fixed choices.

For optimization workflows, prefer APIs that allow the user to freeze discrete choices such as selected modes, truncation order, multi-index sets, resonance structure, normalization convention, and solver configuration, then differentiate through the continuous numerical computation.

The documentation should clearly distinguish:

* differentiating through the continuous SSM coefficient computation for fixed algorithmic choices,
* differentiating through reduced prediction for fixed reduction structure,
* differentiating through the entire adaptive SSM construction including mode selection or branch decisions.

The first two are the priority. The last one may be only piecewise differentiable or not supported.


---

## 11. Testing requirements

Use `pytest`.

Tests must cover:

* imports
* basic correctness
* shape behavior
* edge cases
* numerical tolerances
* representative examples
* JAX transformations
* notebook numerical cores

At minimum, each completed batch should run:

```bash
python -m compileall src tests examples
python -m pytest
```

If colocated example notebooks exist and notebook execution tooling is configured, run notebook checks as appropriate. If notebook execution is not configured, make sure each notebook's numerical core is covered by pytest.

When differentiable functions are added or changed, also run targeted differentiability tests, for example:

```bash
python -m pytest tests/test_differentiability.py
```

If examples exist, run relevant examples directly:

```bash
python examples/<target_example>/example.py
```

Do not weaken assertions merely to make tests pass.

If a tolerance is relaxed, document the numerical reason.

Do not mark unimplemented or failing behavior as passing by replacing assertions with smoke checks.

Prefer deterministic fixtures.

If MATLAB or Octave is available, use it only to generate reference fixtures and behavioral comparisons.

If neither MATLAB nor Octave is available, do not block. Instead derive small deterministic reference cases from the MATLAB source, document that they are source-derived, and continue.

---

## 12. Linear algebra and solver requirements

For sparse or structured systems, prefer matrix-free JAX operators plus JAX-compatible iterative solvers where mathematically appropriate.

Before choosing a solver, classify the system:

1. Dense or sparse?
2. Explicit matrix or matrix-free operator?
3. Symmetric positive definite, general nonsymmetric, least-squares, eigenvalue, or nonlinear?
4. Is differentiation through the solve required?
5. Is batching required?
6. Is the solve inside a JIT-compiled path?

Prefer:

* `jax.numpy.linalg` for dense differentiable operations
* `jax.scipy.linalg` where suitable
* `jax.scipy.sparse.linalg.cg`, `gmres`, or `bicgstab` for matrix-free iterative solves where appropriate

Avoid relying on experimental sparse direct solvers in differentiable core code unless there is a documented reason and tests cover the behavior.

Document solver choices and test them under `jax.jit` and, where appropriate, `jax.grad`, `jax.jacfwd`, or `jax.jacrev`.

---

## 13. Documentation requirements

Maintain:

```text
README.md
docs/current_status.md
docs/migration_plan.md
docs/migration_inventory.md
docs/migration_report.md
```

`README.md` should include:

* installation
* quickstart
* currently reproduced examples
* currently reproduced notebooks
* testing instructions
* known limitations
* differentiability summary

`docs/current_status.md` should be concise and machine-actionable:

```md
# Current Status

## Must-pass commands

- python -m compileall src tests examples
- python -m pytest

## Passing

- ...

## Failing

- command:
  error:
  suspected cause:
  next fix:

## Active target

- MATLAB example or `.mlx` workflow:
- Python example:
- Jupyter notebook:
- Required MATLAB files:
- Required Python modules:
- Acceptance criteria:

## Notes for next run

- ...
```

`docs/migration_plan.md` should include:

* overall migration strategy
* ordering of examples
* ordering of `.mlx` to notebook translations
* rationale for the first target
* dependency-closure strategy
* testing strategy
* differentiability strategy
* known risks

`docs/migration_inventory.md` should list:

* MATLAB file
* role: core, test, example, `.mlx`, demo, utility, plotting, generated, irrelevant
* dependency status
* planned Python destination
* planned notebook destination if applicable
* migration status
* test status
* differentiability classification if public numerical behavior is exposed

`docs/migration_report.md` should include:

* implemented modules
* reproduced examples
* reproduced notebooks
* skipped or deferred items
* known limitations
* exact commands run
* failures and debugging notes
* `Next recommended batch`

---

## 14. Batch continuation protocol

At the end of every Codex run, update `docs/migration_report.md` with a section named exactly:

```md
## Next recommended batch
```

That section must include:

```md
## Next recommended batch

### Target

- ...

### MATLAB files involved

- ...

### MATLAB examples or `.mlx` workflows involved

- ...

### Planned Python modules

- ...

### Planned examples or notebooks

- ...

### Expected tests

- ...

### Known risks

- ...

### Differentiability concerns

- ...

### Acceptance criteria

- ...
```

When the user says “continue”, treat that section as the task definition.

Complete the whole batch rather than one isolated function.

---

## 15. Clean-room restart mode

This branch is a clean restart of the Python/JAX port.

Treat `SSMTool/` as the only implementation reference.

Do not copy code from previous partial Python ports.

Do not create compatibility wrappers for APIs that are not required by the active target.

If old generated Python-port artifacts are present on this branch, ignore them unless needed to remove or quarantine them.

New implementation should be example-driven, notebook-aware, and test-gated.

If old generated artifacts obstruct the clean-room implementation, move them under `attic/` or remove them from this branch after documenting the decision in `docs/migration_report.md`.

---

## 16. Git and change management

Prefer coherent batches over tiny disconnected edits.

Each batch should correspond to one acceptance target.

Before making broad changes, inspect current repository status:

```bash
git status
```

Do not overwrite user changes.

Do not modify `SSMTool/`.

Do not commit generated artifacts, caches, or large binary files unless they are intentional fixtures.

Recommended batch pattern:

1. Baseline inspection.
2. Planning update.
3. Minimal implementation.
4. Add or update tests.
5. Run tests.
6. Update docs.
7. Summarize exact commands and results.

---

## 17. Completion criteria

The port is complete only when:

1. MATLAB source structure has been inspected.
2. A Python package exists.
3. Core MATLAB functionality needed by representative examples has corresponding Python/JAX implementation.
4. Relevant MATLAB examples run end-to-end in Python.
5. Relevant MATLAB `.mlx` workflows have corresponding Jupyter notebooks.
6. Tests cover correctness, shape behavior, edge cases, and numerical tolerances.
7. Differentiable public APIs have JAX transform tests.
8. Non-differentiable or partially differentiable APIs are explicitly classified.
9. The test suite runs.
10. Failures have been debugged as far as possible.
11. Documentation is written.
12. A final summary lists implemented modules, reproduced examples, reproduced notebooks, skipped items, known limitations, and exact commands run.

Until examples are reproduced and tested, the project is not complete, even if many MATLAB files have been translated.

Until `.mlx` workflows have notebook equivalents or documented exclusions, the example migration is not complete.

### `.mlx` to Jupyter notebook migration standard

A translated notebook is not a summary of the MATLAB `.mlx` file. It is an executable Python/JAX reproduction of the MATLAB live-script workflow.

For each relevant `.mlx` file, the corresponding notebook should reproduce, as closely as reasonably possible:

1. model definition,
2. parameter setup,
3. linear analysis or modal setup,
4. SSM reduction / SSM graph computation,
5. reduced dynamics or prediction,
6. trajectory computation or simulation,
7. visualization corresponding to the MATLAB output,
8. explanatory text sufficient for a reader to follow the workflow.

The notebook must not stop at preliminary setup such as vector field definition, eigenvalues, or parameter declarations unless there is a documented hard blocker.

SSM graph computation, reduced dynamics, trajectory computation, and visualization are required parts of the notebook migration when they appear in the MATLAB `.mlx` workflow.

A notebook may be marked `complete` only when:

1. it runs from top to bottom,
2. it reproduces the meaningful numerical workflow of the MATLAB `.mlx`,
3. its core numerical computations are implemented in `src/ssmtoolpy/` or clearly isolated helper code,
4. important numerical outputs are covered by pytest tests,
5. plots are generated from tested numerical outputs.

A notebook that only reproduces setup, model definition, eigenvalues, or preliminary diagnostics must be marked `incomplete`.

If full reproduction is not yet possible, document the exact blocker, such as:

* missing SSM coefficient solver,
* missing graph parameterization,
* missing reduced dynamics integration,
* missing trajectory lifting,
* missing plotting utility,
* missing MATLAB fixture or reference data,
* unresolved numerical discrepancy.

Do not use vague language such as “deferred” without naming the missing dependency and adding it to the next-batch plan.


---

## 18. Immediate next task if no status files exist

If `docs/current_status.md`, `docs/migration_plan.md`, `docs/migration_inventory.md`, or `docs/migration_report.md` do not exist, create them before broad implementation.

Then:

1. Inspect `SSMTool/`.
2. Identify all MATLAB examples, demos, and `.mlx` workflows.
3. Create an initial migration plan.
4. Rank examples by likely dependency complexity and usefulness.
5. Select the smallest representative example or `.mlx` workflow as the first target.
6. If that target has a large dependency closure, define the smallest meaningful numerical subproblem from it as the first acceptance target.
7. Build the dependency closure for the selected target.
8. Record that closure in `docs/migration_inventory.md`.
9. Create a minimal Python/JAX package skeleton.
10. Make the selected example, notebook workflow, or subproblem run.
11. Add a regression test.
12. Add JAX transform tests for introduced differentiable functions.
13. Update `docs/migration_report.md` with the next recommended batch.

Do not attempt to port a large fraction of the remaining MATLAB code in the first batch.