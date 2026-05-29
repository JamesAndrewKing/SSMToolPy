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

Prefer implementation under:

```text
src/ssmtoolpy/
```

Tests should live under:

```text
tests/
```

Examples should live under:

```text
examples/
```

Jupyter notebooks translated from `.mlx` should live under:

```text
notebooks/
```

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

All relevant MATLAB `.mlx` example workflows should eventually have corresponding Jupyter notebooks under:

```text
notebooks/
```

For each `.mlx` file:

1. Identify the mathematical/numerical workflow.
2. Identify inputs, parameters, outputs, and expected figures.
3. Separate numerical computation from presentation.
4. Implement the numerical core in `src/ssmtoolpy/`.
5. Add tests for the numerical core.
6. Create a notebook that calls the tested Python API.
7. Keep notebooks readable and concise.
8. Avoid hiding core numerical logic inside notebooks.

Notebook acceptance requires:

1. The notebook imports the package normally.
2. The notebook runs from top to bottom.
3. The important numerical outputs are covered by pytest tests outside the notebook.
4. Any plots are generated from tested numerical outputs.

If a notebook cannot be executed in the current environment, document the reason and ensure its numerical core is still tested.

---

## 8. Example-first migration process

For each target example:

1. Locate the MATLAB example or `.mlx` workflow.
2. Determine its minimal MATLAB dependency closure.
3. Record the closure in `docs/migration_inventory.md`.
4. Implement the smallest Python/JAX subset needed to run the example or its first meaningful numerical subproblem.
5. Create a Python example under `examples/`.
6. Create a Jupyter notebook under `notebooks/` if the source is a `.mlx` workflow or if an interactive workflow is useful.
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

If `notebooks/` exists and notebook execution tooling is configured, run notebook checks as appropriate. If notebook execution is not configured, make sure the notebook numerical core is covered by pytest.

When differentiable functions are added or changed, also run targeted differentiability tests, for example:

```bash
python -m pytest tests/test_differentiability.py
```

If examples exist, run relevant examples directly:

```bash
python examples/<target_example>.py
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
