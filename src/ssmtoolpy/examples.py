"""Source-derived ports of the simplest MATLAB ``SSMTool/examples`` models."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from ssmtoolpy.dynamical_system import DynamicalSystem
from ssmtoolpy.tensor import sparse_tensor_from_dense


Array = jnp.ndarray


class FirstOrderExampleModel(NamedTuple):
    """First-order example model data.

    ``terms`` follows the MATLAB convention used by ``DynamicalSystem.F``:
    entry ``terms[k - 2]`` stores the order-``k`` tensor with the output mode
    first. ``system`` is an immutable Python/JAX wrapper around the same data.

    Differentiability
    -----------------
    Not differentiable as a container. The stored arrays and ``system.odefun``
    are differentiable with respect to state and coefficient values for fixed
    tensor structure.
    """

    system: DynamicalSystem
    a_matrix: Array
    b_matrix: Array
    terms: tuple[Array, ...]


def _maybe_sparse_terms(terms: tuple[Array, ...], sparse: bool) -> tuple[Array, ...]:
    if not sparse:
        return terms
    return tuple(sparse_tensor_from_dense(term, nse=int(jnp.count_nonzero(term))) for term in terms)


def planar_system_vector_field(state: Array) -> Array:
    """Evaluate the planar example vector field from ``PlanarSystem/build_model.m``.

    The MATLAB model is

    ``x1' = -x1``

    ``x2' = -sqrt(24) x2 + x1^2 + x1^3 + x1^4 + x1^5``.

    Differentiability
    -----------------
    Differentiable with respect to ``state``. Representative tests exercise
    ``jax.jit`` and ``jax.jacfwd``.
    """

    state = jnp.asarray(state)
    x1 = state[0]
    x2 = state[1]
    return jnp.array(
        [
            -x1,
            -jnp.sqrt(jnp.asarray(24.0, dtype=state.dtype)) * x2 + x1**2 + x1**3 + x1**4 + x1**5,
        ],
        dtype=state.dtype,
    )


def planar_system_model(*, sparse: bool = False) -> FirstOrderExampleModel:
    """Build the MATLAB ``examples/PlanarSystem`` first-order model.

    The tensor coefficients are source-derived from
    ``SSMTool/examples/PlanarSystem/build_model.m``. MATLAB stores them as
    ``sptensor`` objects with one nonzero entry in each order 2 through 5:
    ``F_k(2,1,...,1) = 1`` in one-based notation.

    Parameters
    ----------
    sparse:
        If true, store nonlinear tensors as JAX ``BCOO`` sparse arrays. The
        returned ``DynamicalSystem`` currently uses dense terms because the ODE
        evaluation kernel contracts dense tensors directly; sparse terms are
        provided in ``terms`` for sparse tensor workflows.

    Differentiability
    -----------------
    Not differentiable as a constructor. The returned model's ODE evaluation is
    differentiable under the usual nonsingular-``B`` assumption.
    """

    dtype = jnp.result_type(1.0)
    a_matrix = jnp.array([[-1.0, 0.0], [0.0, -jnp.sqrt(24.0)]], dtype=dtype)
    b_matrix = jnp.eye(2, dtype=dtype)
    terms = []
    for degree in range(2, 6):
        term = jnp.zeros((2,) + (2,) * degree, dtype=dtype)
        term = term.at[(1,) + (0,) * degree].set(1.0)
        terms.append(term)
    dense_terms = tuple(terms)
    system = DynamicalSystem(order=1, a_matrix_value=a_matrix, b_matrix_value=b_matrix, first_order_terms=dense_terms)
    return FirstOrderExampleModel(system=system, a_matrix=a_matrix, b_matrix=b_matrix, terms=_maybe_sparse_terms(dense_terms, sparse))


def benchmark_ssm_1st_order_model(*, sparse: bool = False) -> FirstOrderExampleModel:
    """Build the MATLAB ``examples/BenchamrkSSM1stOrder`` model.

    The upstream directory name contains the MATLAB typo ``Benchamrk``. Its
    ``build_model.m`` is coefficient-for-coefficient identical to
    ``PlanarSystem/build_model.m``, so this function intentionally delegates to
    :func:`planar_system_model`.

    Differentiability
    -----------------
    Not differentiable as a constructor. The returned model's ODE evaluation is
    differentiable under the usual nonsingular-``B`` assumption.
    """

    return planar_system_model(sparse=sparse)


def lorenz_vector_field(
    state: Array,
    *,
    sigma: float | Array = 10.0,
    rho: float | Array = 28.0,
    beta: float | Array = 8.0 / 3.0,
) -> Array:
    """Evaluate ``examples/Lorenz1stOrder/lorenz.m``.

    Differentiability
    -----------------
    Differentiable with respect to ``state`` and the numeric parameters away
    from no special singularities. Representative tests exercise ``jax.jit``
    and ``jax.jacfwd``.
    """

    state = jnp.asarray(state)
    sigma = jnp.asarray(sigma, dtype=state.dtype)
    rho = jnp.asarray(rho, dtype=state.dtype)
    beta = jnp.asarray(beta, dtype=state.dtype)
    x, y, z = state
    return jnp.array(
        [
            sigma * (y - x),
            rho * x - y - x * z,
            -beta * z + x * y,
        ],
        dtype=state.dtype,
    )


def lorenz_first_order_model(
    *,
    sigma: float | Array = 10.0,
    rho: float | Array = 28.0,
    beta: float | Array = 8.0 / 3.0,
    sparse: bool = False,
) -> FirstOrderExampleModel:
    """Build the MATLAB ``examples/Lorenz1stOrder`` first-order model.

    The source model uses
    ``A = [[-sigma, sigma, 0], [rho, -1, 0], [0, 0, -beta]]`` and a single
    quadratic sparse tensor with one-based nonzeros ``F2(2,1,3)=-1`` and
    ``F2(3,1,2)=1``.

    Parameters
    ----------
    sigma, rho, beta:
        Lorenz parameters from the MATLAB example.
    sparse:
        If true, return the source tensor terms as JAX ``BCOO`` arrays while
        keeping the ``DynamicalSystem`` dense for direct ODE evaluation.

    Differentiability
    -----------------
    Not differentiable as a constructor. The returned model's ODE evaluation is
    differentiable with respect to state and parameters under the usual
    nonsingular-``B`` assumption.
    """

    dtype = jnp.result_type(sigma, rho, beta, 1.0)
    sigma = jnp.asarray(sigma, dtype=dtype)
    rho = jnp.asarray(rho, dtype=dtype)
    beta = jnp.asarray(beta, dtype=dtype)
    a_matrix = jnp.array(
        [
            [-sigma, sigma, 0.0],
            [rho, -1.0, 0.0],
            [0.0, 0.0, -beta],
        ],
        dtype=dtype,
    )
    b_matrix = jnp.eye(3, dtype=dtype)
    f2 = jnp.zeros((3, 3, 3), dtype=dtype)
    f2 = f2.at[1, 0, 2].set(-1.0)
    f2 = f2.at[2, 0, 1].set(1.0)
    dense_terms = (f2,)
    system = DynamicalSystem(order=1, a_matrix_value=a_matrix, b_matrix_value=b_matrix, first_order_terms=dense_terms)
    return FirstOrderExampleModel(system=system, a_matrix=a_matrix, b_matrix=b_matrix, terms=_maybe_sparse_terms(dense_terms, sparse))
