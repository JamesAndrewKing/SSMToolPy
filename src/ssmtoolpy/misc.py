"""Miscellaneous numerical helpers ported from SSMTool."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ssmtoolpy.multi_index import MultiIndexPolynomial, expand_multiindex
from ssmtoolpy.reduction import NonAutonomousTerm, reduced_to_full


Array = jnp.ndarray


class OutputSummary(NamedTuple):
    """Output extraction result.

    Differentiability
    -----------------
    Not differentiable as a container. Numeric fields are differentiable with
    respect to the input trajectory for fixed output selection.
    """

    z_out: Array
    a_out: Array
    z_norm: Array
    z_out_norm: Array | None


class ProjectionData(NamedTuple):
    """Real/complex coordinate mapping data used by SSM projection helpers.

    Indices are zero-based.

    Differentiability
    -----------------
    Not differentiable as a container. Numeric mapping helpers carry their own
    differentiability status.
    """

    realx: Array
    compx: Array
    dim: int


class AutoReducedDynamicsData(NamedTuple):
    """Autonomous reduced dynamics data for ``auto_red_dyn``.

    Differentiability
    -----------------
    Not differentiable as a container. ``auto_red_dyn`` is differentiable with
    respect to state and coefficients for fixed exponents.
    """

    lamd: Array
    beta: Array
    kappa: Array


def _as_column(points: Array) -> Array:
    points = jnp.asarray(points)
    if points.ndim == 1:
        return points[:, None]
    return points


def reduced_to_full_traj(
    time: Array,
    point: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    nonautonomous: tuple[NonAutonomousTerm, ...] = (),
    epsilon: float | Array = 0.0,
    omega: float | Array = 1.0,
) -> Array:
    """Evaluate full-space trajectory reconstruction at a single time.

    This ports ``reduced_to_full_traj.m``. The return shape is ``(output_dim,)``
    for vector input and ``(output_dim, n_points)`` for matrix input.

    Differentiability
    -----------------
    Differentiable with respect to ``time``, ``point``, coefficients, ``epsilon``
    and ``omega`` for fixed polynomial/non-autonomous structure.
    """

    point_matrix = _as_column(point)
    output_dim = autonomous[0].coeffs.shape[0]
    z = jnp.zeros((output_dim, point_matrix.shape[1]), dtype=jnp.result_type(point_matrix, autonomous[0].coeffs))
    if nonautonomous and not bool(jnp.asarray(epsilon == 0)):
        phi = omega * time
        for item in nonautonomous:
            phase = jnp.exp(1j * item.kappa * phi)
            zeroth = item.terms[0]
            z = z + epsilon * jnp.real(zeroth.coeffs * phase)
            for poly in item.terms[1:]:
                z = z + epsilon * jnp.real(expand_multiindex(poly, point_matrix) * phase)
    for poly in autonomous:
        z = z + jnp.real(expand_multiindex(poly, point_matrix))
    return z[:, 0] if jnp.asarray(point).ndim == 1 else z


def extract_output(z: Array, outdof: Array | Callable[[Array], Array] | None = None) -> OutputSummary:
    """Extract output DOFs, amplitudes, and L2-like norms from a trajectory.

    Numeric ``outdof`` indices are zero-based in Python.

    Differentiability
    -----------------
    Piecewise differentiable for fixed output selection. The infinity norm used
    for amplitudes is non-smooth at ties and sign changes.
    """

    z = jnp.asarray(z)
    nt = z.shape[1]
    z_norm = jnp.linalg.norm(z, ord="fro") / jnp.sqrt(nt - 1)
    if outdof is None:
        return OutputSummary(jnp.asarray([]), jnp.asarray([]), z_norm, None)
    if callable(outdof):
        z_out = jnp.asarray(outdof(z))
    else:
        z_out = z[jnp.asarray(outdof, dtype=jnp.int32), :]
    a_out = jnp.max(jnp.abs(z_out), axis=1)
    z_out_norm = jnp.linalg.norm(z_out, ord="fro") / jnp.sqrt(nt - 1)
    return OutputSummary(z_out, a_out, z_norm, z_out_norm)


def spblkdiag(*blocks: Array) -> Array:
    """Construct a dense block diagonal matrix.

    A single 3D input is interpreted as MATLAB ``spblkdiag(X3d)``: each
    ``[:, :, k]`` slice becomes one block.

    Differentiability
    -----------------
    Differentiable with respect to block values.
    """

    if len(blocks) == 1 and jnp.asarray(blocks[0]).ndim == 3:
        array = jnp.asarray(blocks[0])
        blocks = tuple(array[:, :, idx] for idx in range(array.shape[2]))
    arrays = tuple(jnp.asarray(block) for block in blocks)
    total_rows = sum(block.shape[0] for block in arrays)
    total_cols = sum(block.shape[1] for block in arrays)
    dtype = jnp.result_type(*arrays)
    result = jnp.zeros((total_rows, total_cols), dtype=dtype)
    row = 0
    col = 0
    for block in arrays:
        result = result.at[row : row + block.shape[0], col : col + block.shape[1]].set(block)
        row += block.shape[0]
        col += block.shape[1]
    return result


def solve_invariance_equation(matrix: Array, rhs: Array, solver: str = "backslash") -> Array:
    """Solve ``matrix @ x = rhs`` using an SSMTool-compatible solver name.

    Direct solvers map to ``jax.numpy.linalg.solve`` where possible. ``pinv``
    and ``lsqminnorm`` use the Moore-Penrose pseudoinverse. Iterative MATLAB
    solver names currently fall back to the same least-squares solve because
    JAX has no drop-in Krylov equivalent in ``jax.numpy``.

    Differentiability
    -----------------
    Differentiable under the usual full-rank/non-singularity assumptions for the
    selected linear solve or pseudoinverse.
    """

    matrix = jnp.asarray(matrix)
    rhs = jnp.asarray(rhs)
    if solver in {"linsolve", "backslash", "inv"}:
        return jnp.linalg.solve(matrix, rhs)
    if solver in {"pinv", "lsqminnorm", "bicg", "bicgstab", "cgs", "gmres", "lsqr"}:
        return jnp.linalg.pinv(matrix) @ rhs
    raise ValueError("unknown solver")


def auto_red_dyn(state: Array, data: AutoReducedDynamicsData) -> Array:
    """Evaluate autonomous reduced dynamics ``dot(z)=R(z)``.

    Differentiability
    -----------------
    Differentiable with respect to ``state``, ``lamd`` and ``beta`` for fixed
    integer exponent matrix ``kappa``. Tested with ``jax.grad``/``jax.jit``.
    """

    state = jnp.asarray(state)
    lamd = jnp.asarray(data.lamd)
    beta = jnp.asarray(data.beta)
    kappa = jnp.asarray(data.kappa)
    y = lamd[:, None] * state
    monomials = jnp.prod(state.T[:, None, :] ** kappa[None, :, :], axis=-1).T
    return y + beta @ monomials


def real_to_conjugate_state(u: Array, data: ProjectionData) -> Array:
    """Map real optimization coordinates to conjugate-symmetric modal state.

    Differentiability
    -----------------
    Differentiable with respect to ``u`` for fixed index data.
    """

    u = jnp.asarray(u)
    realx = jnp.asarray(data.realx, dtype=jnp.int32)
    compx = jnp.asarray(data.compx, dtype=jnp.int32)
    state = jnp.zeros((data.dim,), dtype=jnp.result_type(u, 1j))
    state = state.at[realx].set(u[realx])
    real_parts = u[compx[0::2]]
    imag_parts = u[compx[1::2]]
    complex_values = real_parts + 1j * imag_parts
    state = state.at[compx[0::2]].set(complex_values)
    state = state.at[compx[1::2]].set(jnp.conj(complex_values))
    return state


def squared_distance_to_point_ssm(z0: Array, u: Array, autonomous: tuple[MultiIndexPolynomial, ...], data: ProjectionData) -> Array:
    """Squared distance from a point to the SSM parametrization at ``u``.

    The MATLAB source calls ``reduced_to_full(x,W_0)`` although its
    ``reduced_to_full`` signature expects non-autonomous arguments. This port
    uses the intended autonomous reconstruction path.

    Differentiability
    -----------------
    Differentiable with respect to ``z0``, ``u`` and polynomial coefficients
    under the differentiability assumptions of ``reduced_to_full``.
    """

    state = real_to_conjugate_state(u, data)
    z = reduced_to_full(state[:, None], autonomous)[:, 0]
    residual = z - jnp.asarray(z0)
    return jnp.sum(residual**2)


def project_to_ssm_linear(z0: Array, left_eigenvectors: Array, b_matrix: Array) -> Array:
    """Linear projection ``q = Wm' * B * z0`` from ``proj2SSM.m``.

    Differentiability
    -----------------
    Differentiable with respect to all numeric inputs.
    """

    return jnp.conj(jnp.asarray(left_eigenvectors)).T @ jnp.asarray(b_matrix) @ jnp.asarray(z0)


def nonlinear_projection_objective(
    z0: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    data: ProjectionData,
) -> Callable[[Array], Array]:
    """Return the nonlinear projection objective used by ``proj2SSM``.

    This exposes the differentiable objective only. The MATLAB optimization
    driver ``fminunc`` is intentionally not ported in this layer.

    Differentiability
    -----------------
    Differentiable with respect to the optimization variable and coefficients.
    """

    return lambda u: squared_distance_to_point_ssm(z0, u, autonomous, data)
