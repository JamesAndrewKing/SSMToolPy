"""Miscellaneous numerical helpers ported from SSMTool."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse import linalg as jsp_sparse_linalg

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


class LinearResponseResult(NamedTuple):
    """Periodic linear-response samples and amplitudes.

    ``response`` has shape ``(n_omega, n_state, nt)``. ``z_norm`` is the
    MATLAB Frobenius/L2-like norm for each sampled frequency, and ``a_out`` is
    the infinity-norm amplitude at selected output coordinates.

    Differentiability
    -----------------
    Not differentiable as a container. Numeric fields from the linear-response
    kernels are differentiable under nonsingular frequency-domain operators and
    away from infinity-norm ties.
    """

    response: Array
    z_norm: Array
    a_out: Array
    phi: Array


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


def solve_invariance_equation(
    matrix: Array | Callable[[Array], Array],
    rhs: Array,
    solver: str = "backslash",
    *,
    tol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    x0: Array | None = None,
    restart: int = 20,
) -> Array:
    """Solve ``matrix @ x = rhs`` using an SSMTool-compatible solver name.

    Direct solvers map to dense ``jax.numpy`` linear algebra. Iterative solver
    names use ``jax.scipy.sparse.linalg`` and accept either a matrix-like object
    supporting ``@`` or a linear-operator callable ``A(x)``. JAX currently
    exposes ``cg``, ``bicgstab``, and ``gmres``; MATLAB names without a direct
    JAX equivalent are mapped to the closest sparse iterative method.

    Differentiability
    -----------------
    Differentiable under the usual full-rank/non-singularity assumptions for the
    selected linear solve. Iterative sparse solvers are differentiable through
    JAX's implicit linear-solve rules when their assumptions hold.
    """

    rhs = jnp.asarray(rhs)
    direct_solvers = {"linsolve", "backslash", "inv"}
    pseudo_inverse_solvers = {"pinv", "lsqminnorm"}
    iterative_solvers = {"bicg", "bicgstab", "cgs", "gmres", "lsqr", "cg"}

    if solver in direct_solvers | pseudo_inverse_solvers:
        if callable(matrix):
            raise TypeError(f"solver '{solver}' requires an explicit matrix, not a callable linear operator")
        matrix_array = jnp.asarray(matrix)
        if solver in direct_solvers:
            return jnp.linalg.solve(matrix_array, rhs)
        return jnp.linalg.pinv(matrix_array) @ rhs

    if solver not in iterative_solvers:
        raise ValueError("unknown solver")

    if callable(matrix):
        operator = matrix
    else:
        matrix_array = jnp.asarray(matrix)
        operator = lambda vector: matrix_array @ vector

    def solve_vector(vector: Array, x0_vector: Array | None) -> Array:
        if solver == "cg":
            solution, _ = jsp_sparse_linalg.cg(operator, vector, x0=x0_vector, tol=tol, atol=atol, maxiter=maxiter)
        elif solver in {"bicg", "bicgstab", "cgs"}:
            solution, _ = jsp_sparse_linalg.bicgstab(operator, vector, x0=x0_vector, tol=tol, atol=atol, maxiter=maxiter)
        else:
            solution, _ = jsp_sparse_linalg.gmres(
                operator,
                vector,
                x0=x0_vector,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
            )
        return solution

    if rhs.ndim == 1:
        return solve_vector(rhs, None if x0 is None else jnp.asarray(x0))

    x0_array = None if x0 is None else jnp.asarray(x0)
    columns = []
    for col in range(rhs.shape[1]):
        x0_col = None if x0_array is None else x0_array[:, col]
        columns.append(solve_vector(rhs[:, col], x0_col))
    return jnp.stack(columns, axis=1)


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


def _linear_response_summary(response: Array, outdof: Array | None) -> tuple[Array, Array]:
    nt = response.shape[-1]
    z_norm = jnp.linalg.norm(response, axis=(-2, -1)) / jnp.sqrt(nt - 1)
    if outdof is None:
        return z_norm, jnp.zeros((response.shape[0],), dtype=response.real.dtype)
    z_out = response[:, jnp.asarray(outdof, dtype=jnp.int32), :]
    a_out = jnp.max(jnp.abs(z_out), axis=(-2, -1))
    return z_norm, a_out


def first_order_linear_response(
    a_matrix: Array,
    b_matrix: Array,
    kappas: Array,
    coefficients: Array,
    omegas: Array,
    *,
    epsilon: float | Array = 1.0,
    outdof: Array | None = None,
    nt: int = 128,
) -> LinearResponseResult:
    """Compute linear periodic response for first-order systems.

    This ports the ``DS.order ~= 2`` branch of MATLAB ``linear_response.m``:
    for each sampled forcing frequency ``Omega`` and harmonic ``kappa``, solve
    ``(A - i*kappa*Omega*B) z_kappa = F_kappa`` and reconstruct
    ``epsilon * real(sum_k z_kappa exp(i*kappa*phi))``.

    Python shapes are ``coefficients.shape == (n_state, n_kappa)`` and
    ``response.shape == (n_omega, n_state, nt)``.

    Differentiability
    -----------------
    Differentiable with respect to matrices, coefficients, frequencies and
    ``epsilon`` under nonsingular frequency-domain operators. Amplitudes are
    piecewise differentiable because they use an infinity norm.
    """

    a_matrix = jnp.asarray(a_matrix)
    b_matrix = jnp.asarray(b_matrix)
    kappas = jnp.asarray(kappas)
    coefficients = jnp.asarray(coefficients)
    omegas = jnp.asarray(omegas)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nt)

    def solve_at_omega(omega: Array) -> Array:
        operators = a_matrix[None, :, :] - 1j * (kappas * omega)[:, None, None] * b_matrix[None, :, :]
        rhs = coefficients.T
        return jax.vmap(jnp.linalg.solve)(operators, rhs).T

    modal = jax.vmap(solve_at_omega)(omegas)
    phases = jnp.exp(1j * kappas[:, None] * phi[None, :])
    response = jnp.asarray(epsilon) * jnp.real(jnp.einsum("wsk,kt->wst", modal, phases))
    z_norm, a_out = _linear_response_summary(response, outdof)
    return LinearResponseResult(response, z_norm, a_out, phi)


def second_order_linear_response(
    mass: Array,
    damping: Array,
    stiffness: Array,
    kappas: Array,
    coefficients: Array,
    omegas: Array,
    *,
    epsilon: float | Array = 1.0,
    outdof: Array | None = None,
    nt: int = 128,
    conjugate_symmetric: bool = False,
) -> LinearResponseResult:
    """Compute linear periodic response for second-order mechanical systems.

    This ports the ``DS.order == 2`` branch of MATLAB ``linear_response.m``.
    The direct path solves
    ``(-kappa^2 Omega^2 M + i*kappa*Omega C + K) x_kappa = f_kappa`` for every
    supplied harmonic. With ``conjugate_symmetric=True``, only the first
    harmonic is solved and later harmonic columns are filled by MATLAB's
    recursive conjugation convention; this matches the common two-harmonic
    ``(+kappa, -kappa)`` forcing layout used by SSMTool.

    Differentiability
    -----------------
    Differentiable with respect to matrices, coefficients, frequencies and
    ``epsilon`` under nonsingular dynamic stiffness matrices. Amplitudes are
    piecewise differentiable because they use an infinity norm. The
    ``conjugate_symmetric`` branch is a static discrete convention.
    """

    mass = jnp.asarray(mass)
    damping = jnp.asarray(damping)
    stiffness = jnp.asarray(stiffness)
    kappas = jnp.asarray(kappas)
    coefficients = jnp.asarray(coefficients)
    omegas = jnp.asarray(omegas)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, nt)

    def solve_all_at_omega(omega: Array) -> Array:
        k_omega = kappas * omega
        operators = (
            stiffness[None, :, :]
            - (k_omega**2)[:, None, None] * mass[None, :, :]
            + 1j * k_omega[:, None, None] * damping[None, :, :]
        )
        rhs = coefficients.T
        return jax.vmap(jnp.linalg.solve)(operators, rhs).T

    def solve_conjugate_at_omega(omega: Array) -> Array:
        k_omega = kappas[0] * omega
        operator = stiffness - k_omega**2 * mass + 1j * k_omega * damping
        first = jnp.linalg.solve(operator, coefficients[:, 0])
        columns = [first]
        for _ in range(1, kappas.shape[0]):
            columns.append(jnp.conj(columns[-1]))
        return jnp.stack(columns, axis=1)

    solver = solve_conjugate_at_omega if conjugate_symmetric else solve_all_at_omega
    modal = jax.vmap(solver)(omegas)
    phases = jnp.exp(1j * kappas[:, None] * phi[None, :])
    response = jnp.asarray(epsilon) * jnp.real(jnp.einsum("wsk,kt->wst", modal, phases))
    z_norm, a_out = _linear_response_summary(response, outdof)
    return LinearResponseResult(response, z_norm, a_out, phi)


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
