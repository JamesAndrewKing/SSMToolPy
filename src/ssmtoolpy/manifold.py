"""Manifold coefficient algebra kernels."""

from __future__ import annotations

from math import comb
from typing import Callable, Literal, NamedTuple

import jax.numpy as jnp

from ssmtoolpy.dynamical_system import evaluate_polynomial_terms
from ssmtoolpy.multi_index import (
    MultiIndexPolynomial,
    expand_multiindex,
    multi_addition,
    multi_index_2_ordering,
    multi_nsumk,
    multi_subtraction,
    nsumk,
)
from ssmtoolpy.misc import solve_invariance_equation


Array = jnp.ndarray
Ordering = Literal["lex", "revlex"]
ResonanceOrder = Literal["zero", "k"]


class ResonanceSide(NamedTuple):
    """One side of SSM non-resonance analysis.

    Differentiability
    -----------------
    Not differentiable. Resonance analysis enumerates integer combinations and
    threshold-selects near resonances.
    """

    occurs: bool
    sigma: int
    combinations: Array
    eigs: Array


class ResonanceAnalysis(NamedTuple):
    """Inner and outer resonance analysis result.

    Differentiability
    -----------------
    Not differentiable. This is discrete spectral metadata.
    """

    inner: ResonanceSide
    outer: ResonanceSide


class MasterSubspace(NamedTuple):
    """Selected master modal subspace and resonance metadata.

    Differentiability
    -----------------
    Not differentiable. This is spectral selection and discrete resonance
    metadata; eigensolver differentiability is outside this helper.
    """

    spectrum: Array
    basis: Array
    adjoint_basis: Array
    normal_modes: Array
    resonance: ResonanceAnalysis


class NonAutonomousResonanceData(NamedTuple):
    """Data required by ``nonautonomous_resonant_terms``.

    Differentiability
    -----------------
    Not differentiable. This is a metadata container for thresholded
    resonance detection.
    """

    omega: Array
    lambda_master: Array
    dim: int
    reltol: float


class NonAutonomousCoefficientSeries(NamedTuple):
    """One harmonic's non-autonomous coefficient series.

    Differentiability
    -----------------
    Not differentiable as a container. Downstream coefficient algebra carries
    value differentiability for fixed structures.
    """

    kappa: Array
    terms: tuple[MultiIndexPolynomial, ...]


class NonAutonomousStructure(NamedTuple):
    """Initialized non-autonomous SSM/reduced-dynamics coefficient structures.

    Differentiability
    -----------------
    Not differentiable. This mirrors MATLAB struct setup and stores discrete
    harmonic and polynomial-shape metadata.
    """

    w1: tuple[NonAutonomousCoefficientSeries, ...]
    r1: tuple[NonAutonomousCoefficientSeries, ...]
    kappas: Array
    forcing_orders: Array


class AutonomousFirstOrderData(NamedTuple):
    """Data for autonomous first-order SSM coefficient solves.

    Differentiability
    -----------------
    Not differentiable as a container. The functional kernels carry value
    differentiability under their stated nondegeneracy assumptions.
    """

    k_multi: Array
    lambda_master: Array
    left_basis: Array
    right_basis: Array
    reltol: float = 1e-6
    solver: str = "backslash"


class AutonomousSecondOrderData(NamedTuple):
    """Data for autonomous second-order SSM coefficient solves.

    Differentiability
    -----------------
    Not differentiable as a container. The functional kernels carry value
    differentiability under their stated nondegeneracy assumptions.
    """

    k_multi: Array
    lambda_master: Array
    left_displacement_basis: Array
    right_displacement_basis: Array
    reltol: float = 1e-6
    solver: str = "backslash"


class AutonomousSSMSolveResult(NamedTuple):
    """Autonomous SSM coefficient solve result.

    Differentiability
    -----------------
    Not differentiable as a container. Numeric fields inherit the
    differentiability of the solve kernel that produced them.
    """

    reduced_dynamics: Array
    parametrization: Array
    rhs: Array


class IntrusiveCompositionData(NamedTuple):
    """Data for intrusive multi-index force composition.

    ``w`` is the autonomous SSM parametrization series, one
    ``MultiIndexPolynomial`` per order. It is kept in multi-index form because
    this is generally the efficient MATLAB representation for these kernels.

    Differentiability
    -----------------
    Not differentiable as a container. Intrusive composition kernels are
    differentiable with respect to coefficient values for fixed index sets.
    """

    w: tuple[MultiIndexPolynomial, ...]
    ordering: Ordering = "revlex"


class NonAutonomousFirstOrderData(NamedTuple):
    """Data for non-autonomous first-order coefficient solves.

    Differentiability
    -----------------
    Not differentiable as a container. Numeric helper outputs are
    differentiable under fixed resonance pattern and nonsingular solves.
    """

    a_matrix: Array
    b_matrix: Array
    omega: Array
    kappas: Array
    left_basis: Array
    right_basis: Array
    lambda_master: Array
    reltol: float = 1e-6
    solver: str = "backslash"


class NonAutonomousLeadResult(NamedTuple):
    """Leading-order non-autonomous first-order solve result.

    Differentiability
    -----------------
    Not differentiable as a container. Numeric fields inherit differentiability
    from ``nonautonomous_first_order_lead_terms``.
    """

    parametrization: Array
    reduced_dynamics: Array
    rhs: Array
    active_harmonics: Array


class NonAutonomousSolveResult(NamedTuple):
    """High-order non-autonomous first-order solve result.

    Differentiability
    -----------------
    Not differentiable as a container. Numeric fields inherit differentiability
    from ``nonautonomous_first_order_solve_invariance``.
    """

    parametrization: Array
    reduced_dynamics: Array
    rhs: Array


def _multi_indices(dim: int, order: int, ordering: Ordering) -> Array:
    if dim == 1:
        return jnp.asarray([[order]], dtype=jnp.int32)
    indices = nsumk(dim, order, "nonnegative")
    rows = sorted([tuple(int(value) for value in row) for row in indices.tolist()])
    matrix = jnp.asarray(rows, dtype=jnp.int32).T
    if ordering == "revlex":
        matrix = jnp.flip(matrix, axis=1)
    return matrix


def _matlab_find(mask: Array) -> tuple[Array, Array]:
    """Return zero-based row/column hits in MATLAB column-major order."""

    cols, rows = jnp.nonzero(jnp.asarray(mask).T)
    return rows.astype(jnp.int32), cols.astype(jnp.int32)


def _kappa_omega(kappa: Array, omega: Array) -> Array:
    kappa_arr = jnp.asarray(kappa)
    omega_arr = jnp.ravel(jnp.asarray(omega))
    if kappa_arr.ndim == 0:
        return kappa_arr * omega_arr[0]
    if kappa_arr.ndim == 1:
        if kappa_arr.shape[0] == omega_arr.shape[0]:
            return jnp.dot(kappa_arr, omega_arr)
        if omega_arr.shape[0] == 1:
            return kappa_arr * omega_arr[0]
        raise ValueError("1D kappa must have length matching omega, or omega must be scalar")
    if kappa_arr.ndim == 2:
        if kappa_arr.shape[0] == omega_arr.shape[0]:
            return omega_arr @ kappa_arr
        if kappa_arr.shape[1] == omega_arr.shape[0]:
            return kappa_arr @ omega_arr
    raise ValueError("kappa must be scalar, a frequency vector, or a frequency-by-term matrix")


def _has_complex_dtype(value: object) -> bool:
    return jnp.iscomplexobj(jnp.asarray(value))


def _forcing_terms(item: object) -> tuple[MultiIndexPolynomial, ...]:
    terms = getattr(item, "terms", None)
    if terms is None and isinstance(item, dict):
        terms = item.get("terms", item.get("F_n_k"))
    if terms is None:
        terms = getattr(item, "F_n_k", None)
    if terms is None:
        raise ValueError("forcing items must expose terms or F_n_k")
    return tuple(terms)


def _forcing_kappa(item: object) -> Array:
    kappa = getattr(item, "kappa", None)
    if kappa is None and isinstance(item, dict):
        kappa = item.get("kappa")
    if kappa is None:
        raise ValueError("forcing items must expose kappa")
    return jnp.asarray(kappa)


def _series_terms(series: NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...]) -> tuple[MultiIndexPolynomial, ...]:
    return series.terms if isinstance(series, NonAutonomousCoefficientSeries) else tuple(series)


def _polynomial_vector(terms: tuple[MultiIndexPolynomial, ...], order: int, idx: int, input_dim: int) -> Array | None:
    if order <= 0 or order > len(terms):
        return None
    coeffs = terms[order - 1].coeffs
    if coeffs.size == 0 or idx < 0 or idx >= coeffs.shape[1]:
        return None
    return coeffs[:input_dim, idx]


def _expand_multiindex_derivative_explicit(poly: MultiIndexPolynomial, points: Array) -> Array:
    coeffs = jnp.asarray(poly.coeffs)
    ind = jnp.asarray(poly.ind)
    points = jnp.asarray(points)
    if ind.size == 0:
        return jnp.zeros((coeffs.shape[0], points.shape[0], points.shape[1]), dtype=jnp.result_type(coeffs, points))
    derivatives = []
    for dim_idx in range(points.shape[0]):
        powers = ind.at[:, dim_idx].set(jnp.maximum(ind[:, dim_idx] - 1, 0))
        monomials = jnp.prod(points.T[:, None, :] ** powers[None, :, :], axis=-1)
        weighted = ind[:, dim_idx][None, :] * monomials
        derivatives.append(coeffs @ weighted.T)
    return jnp.stack(derivatives, axis=1)


def check_ds_type(
    a_matrix: Array,
    b_matrix: Array,
    force_terms: tuple[object, ...] = (),
    *,
    dim_manifold: int | None = None,
    choose_complex_comp: bool = False,
) -> Literal["real", "complex"]:
    """Classify whether Manifold computations should use real or complex DS mode.

    This ports the decision logic from ``@Manifold/private/check_DStype.m``:
    complex system matrices, complex force coefficients, one-dimensional SSMs,
    or an explicit complex-computation request force ``"complex"`` mode.

    ``force_terms`` may contain objects with a ``coeffs`` attribute, mappings
    with a ``"coeffs"`` key, or raw coefficient arrays.

    Differentiability
    -----------------
    Not differentiable. This is a dtype/metadata decision.
    """

    if _has_complex_dtype(a_matrix) or _has_complex_dtype(b_matrix):
        return "complex"
    for term in force_terms:
        coeffs = getattr(term, "coeffs", term.get("coeffs") if isinstance(term, dict) else term)
        if _has_complex_dtype(coeffs):
            return "complex"
    if dim_manifold == 1 or choose_complex_comp:
        return "complex"
    return "real"


def check_comp_type(
    system_order: int,
    requested_comp_type: Literal["first", "second"] = "first",
    ds_type: Literal["real", "complex"] = "real",
    *,
    dim_manifold: int | None = None,
) -> tuple[Literal["first", "second"], Literal["real", "complex"]]:
    """Select first- or second-order Manifold computation mode.

    This is the functional equivalent of ``@Manifold/private/check_COMPtype.m``.
    First-order systems, complex dynamical systems, and one-dimensional
    manifolds use the first-order algorithm. Second-order computation is kept
    only for second-order real systems with ``requested_comp_type="second"``.

    Differentiability
    -----------------
    Not differentiable. This is a discrete algorithm-selection rule.
    """

    comp_type: Literal["first", "second"]
    ds_mode: Literal["real", "complex"] = ds_type
    if system_order == 1:
        comp_type = "first"
    elif requested_comp_type == "second" and ds_mode == "real":
        comp_type = "second"
    else:
        comp_type = "first"

    if dim_manifold == 1:
        comp_type = "first"
        ds_mode = "complex"
    return comp_type, ds_mode


def _resonance_multiples(dim: int, sigma: int, sigma_max: int) -> Array:
    sigma_check = min(sigma_max, sigma)
    rows = []
    for order in range(2, sigma_check + 1):
        rows.extend(tuple(int(value) for value in row) for row in nsumk(dim, order, "nonnegative").tolist())
    if not rows:
        return jnp.zeros((0, dim), dtype=jnp.int32)
    return jnp.asarray(rows, dtype=jnp.int32)


def resonance_analysis(
    lambda_master: Array,
    lambda_slave: Array,
    reltol: float,
    *,
    sigma_in_max: int = 10,
    sigma_out_max: int = 10,
) -> ResonanceAnalysis:
    """Check near inner and outer resonances for a chosen master spectrum.

    This ports the nested ``resonance_analysis`` routine in
    ``@Manifold/choose_E.m``. The result records the spectral quotients
    ``sigma_in`` and ``sigma_out`` and any near-resonant integer combinations
    of master eigenvalues up to the MATLAB default truncation order 10.

    Differentiability
    -----------------
    Not differentiable. The routine uses integer enumeration, truncation,
    thresholding, and discrete resonant-combination selection.
    """

    lambda_m = jnp.ravel(jnp.asarray(lambda_master))
    lambda_s = jnp.ravel(jnp.asarray(lambda_slave))
    if lambda_m.size == 0:
        raise ValueError("lambda_master must contain at least one eigenvalue")

    ref = jnp.min(jnp.abs(lambda_m))
    if bool(ref < 1e-10):
        ref = jnp.max(jnp.abs(lambda_m))
    abstol = reltol * ref

    all_lambda = jnp.concatenate([lambda_m, lambda_s]) if lambda_s.size else lambda_m
    sigma_in = int(jnp.trunc(jnp.min(jnp.real(all_lambda)) / jnp.max(jnp.real(lambda_m))))
    sigma_out = 0 if lambda_s.size == 0 else int(jnp.trunc(jnp.min(jnp.real(lambda_s)) / jnp.max(jnp.real(lambda_m))))

    empty_combinations = jnp.zeros((0, lambda_m.shape[0]), dtype=jnp.int32)
    empty_eigs = jnp.asarray([], dtype=lambda_m.dtype)

    if sigma_out < 2 or lambda_s.size == 0:
        outer = ResonanceSide(False, sigma_out, empty_combinations, empty_eigs)
    else:
        multiples = _resonance_multiples(lambda_m.shape[0], sigma_out, sigma_out_max)
        combinations = multiples @ lambda_m
        combo_matrix = combinations[:, None]
        slave_matrix = lambda_s[None, :]
        combo_idx, eig_idx = _matlab_find(jnp.abs(combo_matrix - slave_matrix) < abstol)
        outer = ResonanceSide(
            bool(combo_idx.size > 0),
            sigma_out,
            multiples[combo_idx] if combo_idx.size else empty_combinations,
            lambda_s[eig_idx] if eig_idx.size else empty_eigs,
        )

    if sigma_in < 2:
        inner = ResonanceSide(False, sigma_in, empty_combinations, empty_eigs)
    else:
        multiples = _resonance_multiples(lambda_m.shape[0], sigma_in, sigma_in_max)
        combinations = multiples @ lambda_m
        combo_matrix = combinations[:, None]
        master_matrix = lambda_m[None, :]
        combo_idx, eig_idx = _matlab_find(jnp.abs(combo_matrix - master_matrix) < abstol)
        inner = ResonanceSide(
            bool(combo_idx.size > 0),
            sigma_in,
            multiples[combo_idx] if combo_idx.size else empty_combinations,
            lambda_m[eig_idx] if eig_idx.size else empty_eigs,
        )

    return ResonanceAnalysis(inner=inner, outer=outer)


def choose_master_subspace(
    eigenvalues: Array,
    right_eigenvectors: Array,
    left_eigenvectors: Array,
    tangent_modes: Array,
    *,
    reltol: float,
) -> MasterSubspace:
    """Select a master modal subspace and run resonance analysis.

    This is the functional equivalent of ``@Manifold/choose_E.m`` once spectral
    data have already been computed. ``tangent_modes`` are zero-based Python
    indices; MATLAB's object mutation is replaced by a returned
    ``MasterSubspace`` container.

    Differentiability
    -----------------
    Not differentiable. This performs discrete mode selection and resonance
    analysis on supplied spectral data.
    """

    lambdas = jnp.ravel(jnp.asarray(eigenvalues))
    tangent = jnp.ravel(jnp.asarray(tangent_modes, dtype=jnp.int32))
    tangent_set = {int(value) for value in tangent.tolist()}
    normal = jnp.asarray([idx for idx in range(lambdas.shape[0]) if idx not in tangent_set], dtype=jnp.int32)
    lambda_master = lambdas[tangent]
    lambda_slave = lambdas[normal]
    return MasterSubspace(
        spectrum=lambda_master,
        basis=jnp.asarray(right_eigenvectors)[:, tangent],
        adjoint_basis=jnp.asarray(left_eigenvectors)[:, tangent],
        normal_modes=normal,
        resonance=resonance_analysis(lambda_master, lambda_slave, reltol),
    )


def autonomous_resonant_terms(
    lambda_master: Array,
    lambda_combinations: Array,
    reltol: float,
) -> tuple[Array, Array]:
    """Find autonomous internal resonance index pairs.

    Ports ``@Manifold/private/Aut_resonant_terms.m``. The returned arrays are
    zero-based ``(master_mode_index, combination_index)`` pairs, ordered like
    MATLAB ``find`` on the resonance mask.

    Differentiability
    -----------------
    Not differentiable. Resonance detection uses absolute-value thresholding
    and returns discrete index sets.
    """

    lambda_m = jnp.asarray(lambda_master).reshape((-1, 1))
    lambda_k = jnp.asarray(lambda_combinations).reshape((1, -1))
    abstol = reltol * jnp.min(jnp.abs(lambda_m))
    return _matlab_find(jnp.abs(lambda_m - lambda_k) < abstol)


def _matrix_from_rhs(rhs: Array, rows: int, cols: int) -> Array:
    rhs = jnp.asarray(rhs)
    if rhs.ndim == 2:
        return rhs
    if rhs.shape[0] != rows * cols:
        raise ValueError("Flattened RHS has incompatible length")
    return jnp.reshape(rhs, (cols, rows)).T


def _lambda_combinations(k_multi: Array, lambda_master: Array) -> Array:
    return jnp.sum(jnp.asarray(k_multi) * jnp.asarray(lambda_master)[:, None], axis=0)


def autonomous_first_order_reduced_dynamics(
    rhs: Array,
    lambda_master: Array,
    lambda_combinations: Array,
    left_basis: Array,
    right_basis: Array,
    b_matrix: Array,
    *,
    reltol: float = 1e-6,
) -> tuple[Array, Array]:
    """Project resonant first-order invariance RHS terms into reduced dynamics.

    This ports ``@Manifold/private/Aut_1stOrder_RedDyn.m``. Inputs use
    zero-based Python ordering and ``rhs`` may be either ``(N, z_k)`` or a
    MATLAB-style column-major flattened vector. The returned reduced-dynamics
    matrix has shape ``(l, z_k)``.

    Differentiability
    -----------------
    Differentiable with respect to ``rhs``, bases and ``b_matrix`` for a fixed
    resonance pattern. Resonance detection itself is discrete and thresholded.
    """

    lambda_m = jnp.asarray(lambda_master)
    lambda_k = jnp.asarray(lambda_combinations)
    left = jnp.asarray(left_basis)
    right = jnp.asarray(right_basis)
    b_matrix = jnp.asarray(b_matrix)
    n = right.shape[0]
    l = lambda_m.shape[0]
    z_k = lambda_k.shape[0]
    rhs_matrix = _matrix_from_rhs(rhs, n, z_k)
    mode_idx, combo_idx = autonomous_resonant_terms(lambda_m, lambda_k, reltol)
    r0 = jnp.zeros((l, z_k), dtype=jnp.result_type(rhs_matrix, left, right, b_matrix))
    for mode, combo in zip(mode_idx.tolist(), combo_idx.tolist(), strict=False):
        value = jnp.vdot(left[:, mode], rhs_matrix[:, combo])
        r0 = r0.at[mode, combo].add(value)
    updated_rhs = rhs_matrix - (b_matrix @ right) @ r0
    return r0, updated_rhs


def autonomous_first_order_ssm(
    rhs: Array,
    data: AutonomousFirstOrderData,
    a_matrix: Array,
    b_matrix: Array,
) -> AutonomousSSMSolveResult:
    """Solve autonomous first-order SSM coefficients at one polynomial order.

    This ports the functional core of ``@Manifold/private/Aut_1stOrder_SSM.m``:
    resonant RHS components are moved into reduced dynamics and the remaining
    cohomological equations ``(B*lambda_K - A) W_k = RHS_k`` are solved column
    by column.

    Differentiability
    -----------------
    Differentiable under fixed resonance pattern and nonsingular column
    cohomological matrices. The solver choice and multi-index data are static.
    """

    lambda_k = _lambda_combinations(data.k_multi, data.lambda_master)
    r0, updated_rhs = autonomous_first_order_reduced_dynamics(
        rhs,
        data.lambda_master,
        lambda_k,
        data.left_basis,
        data.right_basis,
        b_matrix,
        reltol=data.reltol,
    )
    a_matrix = jnp.asarray(a_matrix)
    b_matrix = jnp.asarray(b_matrix)
    columns = []
    for combo in range(lambda_k.shape[0]):
        matrix = b_matrix * lambda_k[combo] - a_matrix
        columns.append(solve_invariance_equation(matrix, updated_rhs[:, combo], data.solver))
    w0 = jnp.stack(columns, axis=1)
    return AutonomousSSMSolveResult(r0, w0, updated_rhs)


def autonomous_second_order_reduced_dynamics(
    mode_indices: Array,
    combo_indices: Array,
    theta: Array,
    phi: Array,
    damping: Array,
    lambda_combinations: Array,
    lambda_master: Array,
    mass: Array,
    velocity_rhs: Array,
    displacement_rhs: Array,
    *,
    dim: int | None = None,
    z_k: int | None = None,
) -> Array:
    """Compute second-order autonomous reduced-dynamics coefficients.

    This ports the non-1:1-resonance branch of
    ``@Manifold/private/Aut_2ndOrder_RedDyn.m``. If multiple master modes are
    resonant for the same multi-index, MATLAB raises for unsupported 1:1
    internal resonance; this port does the same.

    Differentiability
    -----------------
    Differentiable for a fixed single-mode resonance pattern and nonsingular
    scalar denominators. Resonance indices are discrete inputs.
    """

    mode_indices = jnp.asarray(mode_indices, dtype=jnp.int32)
    combo_indices = jnp.asarray(combo_indices, dtype=jnp.int32)
    theta = jnp.asarray(theta)
    phi = jnp.asarray(phi)
    damping = jnp.asarray(damping)
    lambda_k = jnp.asarray(lambda_combinations)
    lambda_m = jnp.asarray(lambda_master)
    mass = jnp.asarray(mass)
    velocity_rhs = jnp.asarray(velocity_rhs)
    displacement_rhs = jnp.asarray(displacement_rhs)
    l = lambda_m.shape[0] if dim is None else dim
    z_count = lambda_k.shape[0] if z_k is None else z_k
    result = jnp.zeros((l, z_count), dtype=jnp.result_type(theta, phi, damping, mass, velocity_rhs, displacement_rhs, lambda_m))

    unique_combos = sorted({int(value) for value in combo_indices.tolist()})
    for combo in unique_combos:
        modes = [int(mode_indices[idx]) for idx, value in enumerate(combo_indices.tolist()) if int(value) == combo]
        if len(modes) > 1:
            raise NotImplementedError("1:1 internal resonance is not supported for second-order autonomous solves")
        mode = modes[0]
        theta_f = theta[:, mode]
        phi_f = phi[:, mode]
        lhs = jnp.vdot(theta_f, damping @ phi_f + mass @ ((lambda_k[combo] + lambda_m[mode]) * phi_f))
        rhs = lambda_k[combo] * jnp.vdot(theta_f, mass @ velocity_rhs[:, combo]) + jnp.vdot(
            theta_f,
            displacement_rhs[:, combo],
        )
        result = result.at[mode, combo].set(-rhs / lhs)
    return result


def autonomous_second_order_ssm(
    wr: Array,
    fn: Array,
    data: AutonomousSecondOrderData,
    mass: Array,
    damping: Array,
    stiffness: Array,
) -> AutonomousSSMSolveResult:
    """Solve autonomous second-order SSM coefficients at one order.

    This ports the analytic reduced-dynamics branch of
    ``@Manifold/private/Aut_2ndOrder_SSM.m``. The returned parametrization is
    stacked as ``[w_k; dot(w_k)]`` to match MATLAB's first-order state layout.

    Differentiability
    -----------------
    Differentiable under fixed resonance pattern and nonsingular dynamic
    stiffness matrices. Unsupported multi-mode internal resonances raise
    ``NotImplementedError``.
    """

    wr = jnp.asarray(wr)
    fn = jnp.asarray(fn)
    mass = jnp.asarray(mass)
    damping = jnp.asarray(damping)
    stiffness = jnp.asarray(stiffness)
    n = mass.shape[0]
    lambda_k = _lambda_combinations(data.k_multi, data.lambda_master)
    theta = jnp.asarray(data.left_displacement_basis)
    phi = jnp.asarray(data.right_displacement_basis)
    displacement_rhs = -(damping @ wr[:n, :] + mass @ wr[n:, :]) - fn[:n, :]
    velocity_rhs = -wr[:n, :]
    mode_idx, combo_idx = autonomous_resonant_terms(data.lambda_master, lambda_k, data.reltol)
    r0 = autonomous_second_order_reduced_dynamics(
        mode_idx,
        combo_idx,
        theta,
        phi,
        damping,
        lambda_k,
        data.lambda_master,
        mass,
        velocity_rhs,
        displacement_rhs,
        dim=data.lambda_master.shape[0],
        z_k=lambda_k.shape[0],
    )
    w_columns = []
    wdot_columns = []
    for combo in range(lambda_k.shape[0]):
        lhs_term = (mass @ ((lambda_k[combo] + data.lambda_master)[None, :] * phi) + damping @ phi) @ r0[:, combo]
        rhs_column = lhs_term + lambda_k[combo] * (mass @ velocity_rhs[:, combo]) + displacement_rhs[:, combo]
        matrix = -(stiffness + lambda_k[combo] * damping + lambda_k[combo] ** 2 * mass)
        w_col = solve_invariance_equation(matrix, rhs_column, data.solver)
        wdot_col = lambda_k[combo] * w_col + phi @ r0[:, combo] + velocity_rhs[:, combo]
        w_columns.append(w_col)
        wdot_columns.append(wdot_col)
    w = jnp.stack(w_columns, axis=1)
    wdot = jnp.stack(wdot_columns, axis=1)
    return AutonomousSSMSolveResult(r0, jnp.concatenate((w, wdot), axis=0), jnp.concatenate((displacement_rhs, velocity_rhs), axis=0))


def nonautonomous_resonant_terms(
    k: int | None,
    kappa: Array,
    data: NonAutonomousResonanceData,
    order: ResonanceOrder,
) -> tuple[Array, Array, Array]:
    """Find non-autonomous resonance index pairs.

    Ports ``@Manifold/private/nonAut_resonant_terms.m`` for zeroth and
    positive spatial orders. Returned index arrays are zero-based and ordered
    like MATLAB ``find``.

    Differentiability
    -----------------
    Not differentiable. The function enumerates integer multi-indices and
    selects near-resonances by thresholding.
    """

    lambda_m = jnp.asarray(data.lambda_master).reshape((-1, 1))
    abstol = data.reltol * jnp.min(jnp.abs(lambda_m))
    kappa_omega = _kappa_omega(kappa, data.omega)

    if order == "zero":
        forcing_freqs = jnp.ravel(kappa_omega)
        resonance_matrix = lambda_m - 1j * forcing_freqs[None, :]
        e_idx, i_k = _matlab_find(jnp.abs(resonance_matrix) < abstol)
        return e_idx, i_k, jnp.asarray([], dtype=lambda_m.dtype).reshape((0,))

    if order != "k":
        raise ValueError("order must be 'zero' or 'k'")
    if k is None:
        raise ValueError("k must be provided for order='k'")

    if data.dim > 1:
        k_multi = jnp.flip(
            jnp.asarray(sorted(nsumk(data.dim, k, "nonnegative").tolist()), dtype=jnp.int32).T,
            axis=1,
        )
    else:
        k_multi = jnp.asarray([[k]], dtype=jnp.int32)

    lambda_vec = jnp.asarray(data.lambda_master).reshape((data.dim,))
    k_lambda = jnp.sum(k_multi * lambda_vec[:, None], axis=0)
    forcing_shift = jnp.ravel(kappa_omega)[0]
    resonance_matrix = lambda_m - (k_lambda[None, :] + 1j * forcing_shift)
    e_idx, i_k = _matlab_find(jnp.abs(resonance_matrix) < abstol)
    return e_idx, i_k, k_lambda


def nonautonomous_conjugate_reduction(
    kappa_set: Array,
    forcing_coefficients: Array,
    *,
    reltol: float = 1e-6,
) -> tuple[Array, tuple[Array, ...]]:
    """Detect conjugate-related non-autonomous forcing harmonics.

    This ports ``@Manifold/private/nonAut_conj_red.m``. The returned
    ``red_conj`` array contains zero-based representative harmonic indices, and
    ``map_conj`` maps each representative to the zero-based original columns it
    covers. Ordering follows MATLAB's stable ``setdiff`` loop.

    Differentiability
    -----------------
    Not differentiable. The routine performs exact uniqueness checks, harmonic
    matching, norm thresholding, and returns discrete index maps.
    """

    kappas = jnp.ravel(jnp.asarray(kappa_set))
    coeffs = jnp.asarray(forcing_coefficients)
    if coeffs.ndim == 1:
        coeffs = coeffs[None, :]
    if coeffs.shape[1] != kappas.shape[0]:
        raise ValueError("forcing_coefficients must have one column per kappa")

    kappa_values = [complex(value) for value in kappas.tolist()]
    if len(set(kappa_values)) != len(kappa_values):
        raise ValueError("there exist redundancy in kappa of external forcing")

    remaining = list(range(len(kappa_values)))
    representatives: list[int] = []
    maps: list[Array] = []
    while remaining:
        idx = remaining[0]
        ka = kappa_values[idx]
        representatives.append(idx)
        conj_idx = next((candidate for candidate, value in enumerate(kappa_values) if value == -ka), None)
        if conj_idx is not None:
            diff = jnp.linalg.norm(jnp.conj(coeffs[:, idx]) - coeffs[:, conj_idx])
            scale = reltol * jnp.linalg.norm(coeffs[:, conj_idx])
            if bool(diff < scale):
                maps.append(jnp.asarray([idx, conj_idx], dtype=jnp.int32))
                remaining = [candidate for candidate in remaining if candidate not in {idx, conj_idx}]
                continue
        maps.append(jnp.asarray([idx], dtype=jnp.int32))
        remaining = [candidate for candidate in remaining if candidate != idx]

    return jnp.asarray(representatives, dtype=jnp.int32), tuple(maps)


def nonautonomous_struct_setup(
    dim: int,
    state_dim: int,
    order: int,
    forcing: tuple[object, ...],
) -> NonAutonomousStructure:
    """Initialize non-autonomous coefficient containers.

    This is a functional equivalent of ``@Manifold/private/nonAut_struct_setup.m``.
    It creates one SSM coefficient series ``W1`` and one reduced-dynamics
    series ``R1`` per harmonic. Zeroth-order entries are initialized as sparse
    MATLAB did conceptually, represented here by dense zero
    ``MultiIndexPolynomial`` objects with one zero multi-index column.

    Differentiability
    -----------------
    Not differentiable. This routine initializes discrete data structures and
    shape metadata for later coefficient solvers.
    """

    if not forcing:
        empty_kappas = jnp.zeros((0, 0), dtype=jnp.float32)
        return NonAutonomousStructure((), (), empty_kappas, jnp.zeros((0,), dtype=jnp.int32))

    kappa_columns = []
    w_series = []
    r_series = []
    forcing_orders = []
    for item in forcing:
        kappa = jnp.ravel(_forcing_kappa(item))
        kappa_columns.append(kappa)
        forcing_orders.append(len(_forcing_terms(item)))

        empty_w = tuple(
            [MultiIndexPolynomial(jnp.zeros((state_dim, 1)), jnp.zeros((1, dim), dtype=jnp.int32))]
            + [MultiIndexPolynomial(jnp.zeros((state_dim, 0)), jnp.zeros((0, dim), dtype=jnp.int32)) for _ in range(order)]
        )
        empty_r = tuple(
            [MultiIndexPolynomial(jnp.zeros((dim, 1)), jnp.zeros((1, dim), dtype=jnp.int32))]
            + [MultiIndexPolynomial(jnp.zeros((dim, 0)), jnp.zeros((0, dim), dtype=jnp.int32)) for _ in range(order)]
        )
        w_series.append(NonAutonomousCoefficientSeries(kappa, empty_w))
        r_series.append(NonAutonomousCoefficientSeries(kappa, empty_r))

    kappas = jnp.stack(kappa_columns, axis=1)
    return NonAutonomousStructure(
        tuple(w_series),
        tuple(r_series),
        kappas,
        jnp.asarray(forcing_orders, dtype=jnp.int32),
    )


def nonautonomous_assemble_coefficients(
    w1: NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...],
    r1: NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...],
    w_coefficients: Array,
    r_coefficients: Array,
    order: int,
    *,
    dim: int,
    ordering: Ordering = "revlex",
) -> tuple[NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...], NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...]]:
    """Insert solved non-autonomous coefficients at one spatial order.

    This is the immutable Python equivalent of
    ``@Manifold/private/nonAut_assembleCoefficients.m`` for one harmonic. The
    MATLAB function mutates ``W1(i).W(k+1)`` and ``R1(i).R(k+1)``; this helper
    returns updated series instead. Python stores multi-indices row-wise in
    ``MultiIndexPolynomial.ind``.

    Differentiability
    -----------------
    Differentiable with respect to ``w_coefficients`` and ``r_coefficients`` for
    fixed order and dimension. The multi-index insertion metadata is discrete.
    """

    if ordering not in {"lex", "revlex"}:
        raise NotImplementedError("Only lex and revlex assembly are ported")
    if order < 0:
        raise ValueError("order must be nonnegative")

    w_terms = list(_series_terms(w1))
    r_terms = list(_series_terms(r1))
    if order >= len(w_terms) or order >= len(r_terms):
        raise ValueError("series must contain entries through the requested order")

    indices = _multi_indices(dim, order, ordering).T
    w_terms[order] = MultiIndexPolynomial(jnp.asarray(w_coefficients), indices)
    r_terms[order] = MultiIndexPolynomial(jnp.asarray(r_coefficients), indices)

    if isinstance(w1, NonAutonomousCoefficientSeries):
        w_out: NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...] = NonAutonomousCoefficientSeries(w1.kappa, tuple(w_terms))
    else:
        w_out = tuple(w_terms)
    if isinstance(r1, NonAutonomousCoefficientSeries):
        r_out: NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...] = NonAutonomousCoefficientSeries(r1.kappa, tuple(r_terms))
    else:
        r_out = tuple(r_terms)
    return w_out, r_out


def nonautonomous_zeroth_order_forcing(forcing_coefficients: Array) -> tuple[Array, Array]:
    """Extract active zeroth-order non-autonomous forcing columns.

    Ports the nested ``zeroth_order_forcing`` helper in
    ``@Manifold/private/nonAut_1stOrder_leadTerms.m`` for functional inputs.

    Differentiability
    -----------------
    Not differentiable. Active harmonic selection uses exact zero tests and
    returns discrete indices.
    """

    coefficients = jnp.asarray(forcing_coefficients)
    if coefficients.ndim != 2:
        raise ValueError("forcing_coefficients must have shape (state_dim, n_kappa)")
    active = jnp.nonzero(jnp.any(coefficients != 0, axis=0), size=None)[0].astype(jnp.int32)
    return coefficients, active


def nonautonomous_first_order_lead_terms(
    forcing_coefficients: Array,
    data: NonAutonomousFirstOrderData,
    *,
    solve_coefficients: bool = True,
) -> NonAutonomousLeadResult:
    """Solve leading non-autonomous first-order SSM coefficients.

    This ports the functional core of
    ``@Manifold/private/nonAut_1stOrder_leadTerms.m``. Inputs are dense arrays:
    ``forcing_coefficients[:, j]`` is the zeroth-order forcing coefficient for
    harmonic ``j``. Returned coefficient matrices include all harmonics, with
    inactive columns left at zero.

    Differentiability
    -----------------
    Differentiable with respect to forcing coefficients and matrices for fixed
    active-harmonic/resonance structure and nonsingular linear solves. Active
    harmonic selection, conjugate reduction and resonance detection are
    discrete preprocessing.
    """

    forcing, active = nonautonomous_zeroth_order_forcing(forcing_coefficients)
    n_state, n_kappa = forcing.shape
    lambda_master = jnp.asarray(data.lambda_master)
    l = lambda_master.shape[0]
    q0 = jnp.zeros((l, n_kappa), dtype=jnp.result_type(forcing, data.left_basis))
    rhs_active = -forcing[:, active]
    if active.shape[0] > 0:
        resonance_data = NonAutonomousResonanceData(
            omega=data.omega,
            lambda_master=lambda_master,
            dim=l,
            reltol=data.reltol,
        )
        ev_idx, harm_idx, _ = nonautonomous_resonant_terms(None, jnp.asarray(data.kappas)[:, active], resonance_data, "zero")
        if ev_idx.shape[0] > 0:
            for ev, harm in zip(ev_idx.tolist(), harm_idx.tolist(), strict=False):
                active_harmonic = int(active[harm])
                value = jnp.vdot(jnp.asarray(data.left_basis)[:, ev], forcing[:, active_harmonic])
                q0 = q0.at[ev, active_harmonic].set(value)
            rhs_active = jnp.asarray(data.b_matrix) @ jnp.asarray(data.right_basis) @ q0[:, active] - forcing[:, active]

    w10 = jnp.zeros((n_state, n_kappa), dtype=jnp.result_type(rhs_active, data.a_matrix, data.b_matrix))
    if solve_coefficients and active.shape[0] > 0:
        active_kappas = jnp.asarray(data.kappas)[:, active]
        _, maps = nonautonomous_conjugate_reduction(active_kappas, forcing[:, active], reltol=data.reltol)
        representatives = []
        seen = set()
        for map_item in maps:
            rep = int(map_item[0])
            if rep not in seen:
                representatives.append((rep, map_item))
                seen.update(int(v) for v in map_item.tolist())
        for rep, map_item in representatives:
            harmonic = int(active[rep])
            shift = 1j * jnp.dot(jnp.ravel(jnp.asarray(data.omega)), jnp.ravel(jnp.asarray(data.kappas)[:, harmonic]))
            matrix = jnp.asarray(data.a_matrix) - shift * jnp.asarray(data.b_matrix)
            solved = solve_invariance_equation(matrix, rhs_active[:, rep], data.solver)
            if map_item.shape[0] == 1:
                w10 = w10.at[:, harmonic].set(solved)
            elif map_item.shape[0] == 2:
                first = int(active[int(map_item[0])])
                second = int(active[int(map_item[1])])
                w10 = w10.at[:, first].set(solved)
                w10 = w10.at[:, second].set(jnp.conj(solved))
            else:
                raise ValueError("there exist redundancy in kappa of external forcing")

    rhs_full = jnp.zeros((n_state, n_kappa), dtype=rhs_active.dtype)
    if active.shape[0] > 0:
        rhs_full = rhs_full.at[:, active].set(rhs_active)
    return NonAutonomousLeadResult(w10, q0, rhs_full, active)


def nonautonomous_first_order_solve_invariance(
    forcing_and_nonlinearity: Array,
    mixed_terms: Array,
    data: NonAutonomousFirstOrderData,
    *,
    harmonic_index: int,
    order: int,
    mode_indices: Array,
    multi_indices: Array,
    lambda_combinations: Array,
    autonomous_parametrization: tuple[MultiIndexPolynomial, ...],
    dim: int,
    ordering: Ordering = "revlex",
) -> NonAutonomousSolveResult:
    """Solve one high-order non-autonomous first-order invariance equation.

    This ports ``@Manifold/private/nonAut_1stOrder_SolveInvEq.m`` as a
    functional kernel. ``forcing_and_nonlinearity`` corresponds to MATLAB
    ``FG`` and ``mixed_terms`` to ``WR``.

    Differentiability
    -----------------
    Differentiable with respect to numeric inputs for fixed resonance pattern,
    fixed multi-index structures and nonsingular coefficient matrices.
    Resonance indices and harmonic selection are discrete inputs.
    """

    fg = jnp.asarray(forcing_and_nonlinearity)
    wr = jnp.asarray(mixed_terms)
    lambda_k = jnp.asarray(lambda_combinations)
    mode_indices = jnp.asarray(mode_indices, dtype=jnp.int32)
    multi_indices = jnp.asarray(multi_indices, dtype=jnp.int32)
    state_dim, z_k = fg.shape
    l = jnp.asarray(data.lambda_master).shape[0]
    r1 = jnp.zeros((l, z_k), dtype=jnp.result_type(fg, wr, data.left_basis, data.b_matrix))
    residual = fg - jnp.asarray(data.b_matrix) @ wr
    for mode, multi in zip(mode_indices.tolist(), multi_indices.tolist(), strict=False):
        value = jnp.vdot(jnp.asarray(data.left_basis)[:, mode], residual[:, multi])
        r1 = r1.at[mode, multi].set(value)

    r1_terms = [MultiIndexPolynomial(jnp.zeros((l, 0), dtype=r1.dtype), jnp.zeros((0, dim), dtype=jnp.int32)) for _ in range(order + 1)]
    r1_terms[order] = MultiIndexPolynomial(r1, _multi_indices(dim, order, ordering).T)
    correction = coeffs_mixed_terms(
        order,
        1,
        autonomous_parametrization,
        tuple(r1_terms),
        dim=dim,
        output_dim=state_dim,
        mix="R1",
        ordering=ordering,
        explicit_indices=True,
    )
    rhs = jnp.asarray(data.b_matrix) @ (wr + correction) - fg
    w1_columns = []
    harmonic_shift = 1j * jnp.dot(jnp.ravel(jnp.asarray(data.omega)), jnp.ravel(jnp.asarray(data.kappas)[:, harmonic_index]))
    for col in range(z_k):
        matrix = jnp.asarray(data.a_matrix) - jnp.asarray(data.b_matrix) * (lambda_k[col] + harmonic_shift)
        w1_columns.append(solve_invariance_equation(matrix, rhs[:, col], data.solver))
    return NonAutonomousSolveResult(jnp.stack(w1_columns, axis=1), r1, rhs)


def nonautonomous_w1r0_plus_w0r1(
    order: int,
    w0: tuple[MultiIndexPolynomial, ...],
    w1: NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...],
    r0: tuple[MultiIndexPolynomial, ...],
    r1: NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...],
    *,
    dim: int,
    output_dim: int,
    ordering: Ordering = "revlex",
) -> Array:
    """Compose first-order non-autonomous mixed products at one spatial order.

    Ports ``@Manifold/private/nonAut_W1R0_plus_W0R1.m``:
    ``W1 R0 + W0 R1`` restricted to all multi-indices of ``order``. Inputs use
    the same polynomial tuple convention as ``coeffs_mixed_terms``; ``w1`` and
    ``r1`` may also be ``NonAutonomousCoefficientSeries`` containers.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values for fixed polynomial
    structures, order, and ordering. The loop bounds and index matching are
    discrete preprocessing.
    """

    w1_terms = _series_terms(w1)
    r1_terms = _series_terms(r1)
    z_k = comb(order + dim - 1, dim - 1)
    dtype = jnp.result_type(
        *(poly.coeffs for poly in w0),
        *(poly.coeffs for poly in w1_terms),
        *(poly.coeffs for poly in r0),
        *(poly.coeffs for poly in r1_terms),
    )
    result = jnp.zeros((output_dim, z_k), dtype=dtype)

    for m_order in range(1, order + 1):
        if m_order <= len(w1_terms) and w1_terms[m_order - 1].coeffs.size:
            result = result + coeffs_mixed_terms(
                order,
                m_order,
                w1_terms,
                r0,
                dim=dim,
                output_dim=output_dim,
                mix="W1",
                ordering=ordering,
                explicit_indices=True,
            )

    for m_order in range(2, order + 2):
        r1_order = order - m_order + 1
        if 0 <= r1_order < len(r1_terms) and r1_terms[r1_order].coeffs.size:
            result = result + coeffs_mixed_terms(
                order,
                m_order,
                w0,
                r1_terms,
                dim=dim,
                output_dim=output_dim,
                mix="R1",
                ordering=ordering,
                explicit_indices=True,
            )
    return result


def step_polynomial(
    function: Callable[[Array], Array],
    vectors: tuple[Array, ...],
    multi_indices: Array,
    nonlinear_order: int,
) -> Array:
    """Evaluate the symmetric multilinear part of a non-intrusive polynomial.

    Ports ``misc/StEP.m`` for polynomial orders one through three. The helper
    reconstructs the homogeneous multilinear contribution from calls to a
    nonlinear function handle, including the complex-input correction used by
    the MATLAB implementation.

    Differentiability
    -----------------
    Differentiable if ``function`` is JAX-transformable. The order-dependent
    polarization branch is static/discrete.
    """

    if nonlinear_order < 1 or nonlinear_order > 3:
        raise NotImplementedError("step_polynomial currently ports MATLAB StEP orders 1, 2, and 3")
    if len(vectors) != nonlinear_order:
        raise ValueError("vectors length must match nonlinear_order")

    vectors = tuple(jnp.asarray(vector) for vector in vectors)
    m = jnp.asarray(multi_indices)
    if m.ndim == 1:
        m = m[:, None]
    columns = [tuple(int(value) for value in m[:, col].tolist()) for col in range(m.shape[1])]
    unique_columns: list[tuple[int, ...]] = []
    inverse: list[int] = []
    for column in columns:
        if column not in unique_columns:
            unique_columns.append(column)
        inverse.append(unique_columns.index(column))
    multiplicities = [inverse.count(idx) for idx in range(len(unique_columns))]

    if nonlinear_order == 1:
        v1 = vectors[0]
        return 0.5 * (function(v1) - function(-v1))

    def f2(vector: Array) -> Array:
        return 1j * function(jnp.real(vector) + jnp.imag(vector)) + (1 - 1j) * function(jnp.real(vector)) - (1 + 1j) * function(jnp.imag(vector))

    if nonlinear_order == 2:
        v1, v2 = vectors
        if len(unique_columns) == 1:
            return 0.5 * (f2(v1) + f2(-v1))
        return 0.25 * (f2(v1 + v2) + f2(-v1 - v2) - f2(v1 - v2) - f2(-v1 + v2))

    def f3(vector: Array) -> Array:
        return 2 * function(jnp.real(vector)) + ((-1 + 1j) / 2) * function(jnp.real(vector) + jnp.imag(vector)) - ((1 + 1j) / 2) * function(
            jnp.real(vector) - jnp.imag(vector)
        ) - 2j * function(jnp.imag(vector))

    def h(vector: Array) -> Array:
        return 0.5 * (f3(vector) - f3(-vector))

    if len(unique_columns) == 1:
        return h(vectors[0])
    if len(unique_columns) == 2:
        repeated_unique = multiplicities.index(2)
        single_unique = multiplicities.index(1)
        v1 = vectors[inverse.index(repeated_unique)]
        v2 = vectors[inverse.index(single_unique)]
        return 0.5 * (h(v1 + v2) - h(v1 - v2)) - h(v2)
    v1, v2, v3 = vectors
    return h(v1 + v2 + v3) - h(v1 + v2) - h(v1 + v3) - h(v2 + v3) + h(v1) + h(v2) + h(v3)


def fnl_nonintrusive(
    function: Callable[[Array], Array],
    w: tuple[MultiIndexPolynomial, ...],
    nonlinear_order: int,
    k_indices: Array,
    *,
    input_dim: int,
    ordering: Ordering = "revlex",
) -> Array:
    """Compose a non-intrusive nonlinearity with autonomous SSM coefficients.

    This ports the reverse-lexicographic branch of
    ``@Manifold/private/fnl_nonIntrusive.m``. Conjugate-order symmetry remains
    part of the later cohomological-solver layer.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values if ``function`` is
    JAX-transformable and the multi-index structure is fixed.
    """

    if ordering != "revlex":
        raise NotImplementedError("Only revlex non-intrusive composition is ported")
    k_indices = jnp.asarray(k_indices, dtype=jnp.int32)
    if k_indices.ndim == 1:
        k_indices = k_indices[:, None]
    output_dim = w[0].coeffs.shape[0]
    result = jnp.zeros((output_dim, k_indices.shape[1]), dtype=jnp.result_type(*(poly.coeffs for poly in w), 1j))

    for col in range(k_indices.shape[1]):
        combos, _, _, _, _ = multi_nsumk(nonlinear_order, k_indices[:, col], unique=True)
        pages = combos[0][0]
        column_value = jnp.zeros((output_dim,), dtype=result.dtype)
        for page_idx in range(pages.shape[2]):
            page = pages[:, :, page_idx]
            orders = jnp.sum(page, axis=0)
            if bool(jnp.any(orders == 0)):
                continue
            vectors = []
            empty = False
            for arg_idx in range(nonlinear_order):
                subindex = int(multi_index_2_ordering(page[:, arg_idx : arg_idx + 1], ordering)[0]) - 1
                vector = _polynomial_vector(w, int(orders[arg_idx]), subindex, input_dim)
                if vector is None:
                    empty = True
                    break
                vectors.append(vector)
            if not empty:
                column_value = column_value + step_polynomial(function, tuple(vectors), page, nonlinear_order)
        result = result.at[:, col].set(column_value)
    return result


def fnl_semi_intrusive(
    function: Callable[[tuple[Array, ...]], Array],
    w: tuple[MultiIndexPolynomial, ...],
    nonlinear_order: int,
    k_indices: Array,
    *,
    input_dim: int,
    ordering: Ordering = "revlex",
    symmetric: bool = True,
) -> Array:
    """Compose a semi-intrusive multilinear nonlinearity with SSM coefficients.

    This ports the reverse-lexicographic branch of
    ``@Manifold/private/fnl_semiIntrusive.m``. When ``symmetric=True``, unique
    multi-index partitions are weighted by their permutation multiplicity, as
    in MATLAB.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values if ``function`` is
    JAX-transformable and the multi-index structure is fixed.
    """

    if ordering != "revlex":
        raise NotImplementedError("Only revlex semi-intrusive composition is ported")
    k_indices = jnp.asarray(k_indices, dtype=jnp.int32)
    if k_indices.ndim == 1:
        k_indices = k_indices[:, None]
    output_dim = w[0].coeffs.shape[0]
    result = jnp.zeros((output_dim, k_indices.shape[1]), dtype=jnp.result_type(*(poly.coeffs for poly in w)))

    for col in range(k_indices.shape[1]):
        if symmetric:
            combos, _, _, _, multiplicities = multi_nsumk(nonlinear_order, k_indices[:, col], unique=True)
            pages = combos[0][0]
            weights = multiplicities[0][0]
        else:
            combos, _, _, _ = multi_nsumk(nonlinear_order, k_indices[:, col])
            pages = combos[0][0]
            weights = jnp.ones((pages.shape[2],), dtype=jnp.int32)
        column_value = jnp.zeros((output_dim,), dtype=result.dtype)
        for page_idx in range(pages.shape[2]):
            page = pages[:, :, page_idx]
            orders = jnp.sum(page, axis=0)
            if bool(jnp.any(orders == 0)):
                continue
            vectors = []
            empty = False
            for arg_idx in range(nonlinear_order):
                subindex = int(multi_index_2_ordering(page[:, arg_idx : arg_idx + 1], ordering)[0]) - 1
                vector = _polynomial_vector(w, int(orders[arg_idx]), subindex, input_dim)
                if vector is None:
                    empty = True
                    break
                vectors.append(vector)
            if not empty:
                column_value = column_value + weights[page_idx] * function(tuple(vectors))
        result = result.at[:, col].set(column_value)
    return result


def _nonautonomous_vector(
    series: NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...],
    order: int,
    idx: int,
    input_dim: int,
) -> Array | None:
    terms = _series_terms(series)
    if order < 0 or order >= len(terms):
        return None
    coeffs = terms[order].coeffs
    if coeffs.size == 0 or idx < 0 or idx >= coeffs.shape[1]:
        return None
    return coeffs[:input_dim, idx]


def _dfnl_nonintrusive_at_k(
    jacobian_function: Callable[[Array], Array],
    nonlinear_order: int,
    w: tuple[MultiIndexPolynomial, ...],
    k_index: Array,
    x_vectors: tuple[Array, ...],
    *,
    input_dim: int,
    output_dim: int,
    ordering: Ordering,
) -> tuple[Array, ...]:
    if nonlinear_order <= 1:
        raise ValueError("nonlinear_order must be at least 2 for Jacobian composition")
    combos, _, _, _, _ = multi_nsumk(nonlinear_order - 1, k_index, unique=True)
    pages = combos[0][0]
    values = [jnp.zeros((output_dim,), dtype=jnp.result_type(*(poly.coeffs for poly in w), *x_vectors, 1j)) for _ in x_vectors]

    for page_idx in range(pages.shape[2]):
        page = pages[:, :, page_idx]
        orders = jnp.sum(page, axis=0)
        if bool(jnp.any(orders == 0)):
            continue
        vectors = []
        empty = False
        for arg_idx in range(nonlinear_order - 1):
            subindex = int(multi_index_2_ordering(page[:, arg_idx : arg_idx + 1], ordering)[0]) - 1
            vector = _polynomial_vector(w, int(orders[arg_idx]), subindex, input_dim)
            if vector is None:
                empty = True
                break
            vectors.append(vector)
        if empty:
            continue
        df_comp_w = step_polynomial(jacobian_function, tuple(vectors), page, nonlinear_order - 1)
        for harmonic, x_value in enumerate(x_vectors):
            values[harmonic] = values[harmonic] + df_comp_w[:, :input_dim] @ x_value
    return tuple(values)


def _dfnl_semi_intrusive_at_k(
    jacobian_function: Callable[[tuple[Array, ...]], Array],
    nonlinear_order: int,
    w: tuple[MultiIndexPolynomial, ...],
    k_index: Array,
    x_vectors: tuple[Array, ...],
    *,
    input_dim: int,
    output_dim: int,
    ordering: Ordering,
    symmetric: bool,
) -> tuple[Array, ...]:
    if nonlinear_order <= 1:
        raise ValueError("nonlinear_order must be at least 2 for Jacobian composition")
    if symmetric:
        combos, _, _, _, multiplicities = multi_nsumk(nonlinear_order - 1, k_index, unique=True)
        pages = combos[0][0]
        weights = multiplicities[0][0]
    else:
        combos, _, _, _ = multi_nsumk(nonlinear_order - 1, k_index)
        pages = combos[0][0]
        weights = jnp.ones((pages.shape[2],), dtype=jnp.int32)

    values = [jnp.zeros((output_dim,), dtype=jnp.result_type(*(poly.coeffs for poly in w), *x_vectors)) for _ in x_vectors]
    for page_idx in range(pages.shape[2]):
        page = pages[:, :, page_idx]
        orders = jnp.sum(page, axis=0)
        if bool(jnp.any(orders == 0)):
            continue
        vectors = []
        empty = False
        for arg_idx in range(nonlinear_order - 1):
            subindex = int(multi_index_2_ordering(page[:, arg_idx : arg_idx + 1], ordering)[0]) - 1
            vector = _polynomial_vector(w, int(orders[arg_idx]), subindex, input_dim)
            if vector is None:
                empty = True
                break
            vectors.append(vector)
        if empty:
            continue
        for harmonic, x_value in enumerate(x_vectors):
            values[harmonic] = values[harmonic] + weights[page_idx] * jacobian_function((x_value, *vectors))
    return tuple(values)


def dfnl_nonintrusive(
    jacobian_function: Callable[[Array], Array],
    nonlinear_order: int,
    w: tuple[MultiIndexPolynomial, ...],
    x_series: tuple[NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...], ...],
    m_indices: Array,
    *,
    input_dim: int,
    ordering: Ordering = "revlex",
) -> tuple[Array, ...]:
    """Compose a non-intrusive Jacobian with autonomous and non-autonomous SSM coefficients.

    This ports the reverse-lexicographic branch of
    ``@Manifold/private/dfnl_nonIntrusive.m``. The result is one dense array per
    harmonic, corresponding to MATLAB ``dfnl(jj).val``.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values if ``jacobian_function``
    is JAX-transformable and the multi-index structure is fixed.
    """

    if ordering != "revlex":
        raise NotImplementedError("Only revlex non-intrusive Jacobian composition is ported")
    m_indices = jnp.asarray(m_indices, dtype=jnp.int32)
    if m_indices.ndim == 1:
        m_indices = m_indices[:, None]
    output_dim = w[0].coeffs.shape[0]
    if output_dim % 2 == 1:
        input_dim = output_dim
    dtype = jnp.result_type(*(poly.coeffs for poly in w), *[poly.coeffs for series in x_series for poly in _series_terms(series)], 1j)
    results = [jnp.zeros((output_dim, m_indices.shape[1]), dtype=dtype) for _ in x_series]

    for col in range(m_indices.shape[1]):
        partitions, _, _, _ = multi_nsumk(2, m_indices[:, col])
        pages = partitions[0][0]
        column_values = [jnp.zeros((output_dim,), dtype=dtype) for _ in x_series]
        for page_idx in range(pages.shape[2]):
            page = pages[:, :, page_idx]
            orders = jnp.sum(page, axis=0)
            if int(orders[0]) == 0 or int(orders[0]) < nonlinear_order - 1:
                continue
            k_index = page[:, 0]
            l_index = page[:, 1]
            l_order = int(orders[1])
            l_subindex = int(multi_index_2_ordering(l_index[:, None], ordering)[0]) - 1
            x_vectors = tuple(_nonautonomous_vector(series, l_order, l_subindex, input_dim) for series in x_series)
            if any(vector is None for vector in x_vectors):
                continue
            contributions = _dfnl_nonintrusive_at_k(
                jacobian_function,
                nonlinear_order,
                w,
                k_index,
                x_vectors,  # type: ignore[arg-type]
                input_dim=input_dim,
                output_dim=output_dim,
                ordering=ordering,
            )
            column_values = [current + contribution for current, contribution in zip(column_values, contributions, strict=True)]
        for harmonic, value in enumerate(column_values):
            results[harmonic] = results[harmonic].at[:, col].set(value)
    return tuple(results)


def dfnl_semi_intrusive(
    jacobian_function: Callable[[tuple[Array, ...]], Array],
    nonlinear_order: int,
    w: tuple[MultiIndexPolynomial, ...],
    x_series: tuple[NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...], ...],
    m_indices: Array,
    *,
    input_dim: int,
    ordering: Ordering = "revlex",
    symmetric: bool = True,
) -> tuple[Array, ...]:
    """Compose a semi-intrusive Jacobian with autonomous and non-autonomous SSM coefficients.

    This ports the reverse-lexicographic branch of
    ``@Manifold/private/dfnl_semiIntrusive.m``. The result is one dense array per
    harmonic, corresponding to MATLAB ``dfnl(jj).val``.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values if ``jacobian_function``
    is JAX-transformable and the multi-index structure is fixed.
    """

    if ordering != "revlex":
        raise NotImplementedError("Only revlex semi-intrusive Jacobian composition is ported")
    m_indices = jnp.asarray(m_indices, dtype=jnp.int32)
    if m_indices.ndim == 1:
        m_indices = m_indices[:, None]
    output_dim = w[0].coeffs.shape[0]
    dtype = jnp.result_type(*(poly.coeffs for poly in w), *[poly.coeffs for series in x_series for poly in _series_terms(series)])
    results = [jnp.zeros((output_dim, m_indices.shape[1]), dtype=dtype) for _ in x_series]

    for col in range(m_indices.shape[1]):
        partitions, _, _, _ = multi_nsumk(2, m_indices[:, col])
        pages = partitions[0][0]
        column_values = [jnp.zeros((output_dim,), dtype=dtype) for _ in x_series]
        for page_idx in range(pages.shape[2]):
            page = pages[:, :, page_idx]
            orders = jnp.sum(page, axis=0)
            if int(orders[0]) == 0 or int(orders[0]) < nonlinear_order - 1:
                continue
            k_index = page[:, 0]
            l_index = page[:, 1]
            l_order = int(orders[1])
            l_subindex = int(multi_index_2_ordering(l_index[:, None], ordering)[0]) - 1
            x_vectors = tuple(_nonautonomous_vector(series, l_order, l_subindex, input_dim) for series in x_series)
            if any(vector is None for vector in x_vectors):
                continue
            contributions = _dfnl_semi_intrusive_at_k(
                jacobian_function,
                nonlinear_order,
                w,
                k_index,
                x_vectors,  # type: ignore[arg-type]
                input_dim=input_dim,
                output_dim=output_dim,
                ordering=ordering,
                symmetric=symmetric,
            )
            column_values = [current + contribution for current, contribution in zip(column_values, contributions, strict=True)]
        for harmonic, value in enumerate(column_values):
            results[harmonic] = results[harmonic].at[:, col].set(value)
    return tuple(results)


def _series_component_polynomial(
    series: tuple[MultiIndexPolynomial, ...],
    component: int,
    dim: int,
) -> dict[tuple[int, ...], Array]:
    result: dict[tuple[int, ...], Array] = {}
    for poly in series:
        coeffs = jnp.asarray(poly.coeffs)
        ind = jnp.asarray(poly.ind, dtype=jnp.int32)
        if coeffs.size == 0 or component >= coeffs.shape[0]:
            continue
        for col in range(ind.shape[0]):
            key = tuple(int(value) for value in ind[col].tolist())
            value = coeffs[component, col]
            result[key] = result.get(key, jnp.asarray(0, dtype=value.dtype)) + value
    if not result:
        result[(0,) * dim] = jnp.asarray(0.0)
    return result


def _poly_mul_truncated(
    left: dict[tuple[int, ...], Array],
    right: dict[tuple[int, ...], Array],
    target: tuple[int, ...],
) -> dict[tuple[int, ...], Array]:
    out: dict[tuple[int, ...], Array] = {}
    for key_left, value_left in left.items():
        for key_right, value_right in right.items():
            key = tuple(a + b for a, b in zip(key_left, key_right, strict=True))
            if all(key_i <= target_i for key_i, target_i in zip(key, target, strict=True)):
                out[key] = out.get(key, jnp.asarray(0, dtype=jnp.result_type(value_left, value_right))) + value_left * value_right
    return out


def _poly_power_truncated(
    base: dict[tuple[int, ...], Array],
    exponent: int,
    target: tuple[int, ...],
    dtype: Array,
) -> dict[tuple[int, ...], Array]:
    dim = len(target)
    result: dict[tuple[int, ...], Array] = {(0,) * dim: jnp.asarray(1, dtype=jnp.asarray(dtype).dtype)}
    for _ in range(exponent):
        result = _poly_mul_truncated(result, base, target)
    return result


def _composition_coefficient(
    powers: tuple[int, ...],
    target: tuple[int, ...],
    component_polys: tuple[dict[tuple[int, ...], Array], ...],
    dtype: Array,
) -> Array:
    result: dict[tuple[int, ...], Array] = {(0,) * len(target): jnp.asarray(1, dtype=jnp.asarray(dtype).dtype)}
    for component, exponent in enumerate(powers):
        if exponent == 0:
            continue
        powered = _poly_power_truncated(component_polys[component], exponent, target, dtype)
        result = _poly_mul_truncated(result, powered, target)
        if not result:
            break
    return result.get(target, jnp.asarray(0, dtype=jnp.asarray(dtype).dtype))


def fnl_intrusive(
    n_indices: Array,
    k_indices: Array,
    data: IntrusiveCompositionData,
) -> Array:
    """Compose intrusive monomial multi-indices with an SSM parametrization.

    This ports the mathematical core of
    ``@Manifold/private/fnl_intrusive.m`` in multi-index form. For each column
    ``n`` of ``n_indices`` and target reduced multi-index ``k`` in
    ``k_indices``, the returned value is the coefficient of ``p^k`` in
    ``prod_i W_i(p) ** n_i``. Zero multi-index rows/columns follow MATLAB's
    special case: the zero monomial evaluated at the zero target has
    coefficient one.

    Differentiability
    -----------------
    Differentiable with respect to parametrization coefficients for fixed
    ``n_indices``, ``k_indices`` and polynomial structures. The index algebra is
    discrete preprocessing.
    """

    if data.ordering not in {"revlex", "lex"}:
        raise NotImplementedError("Conjugate intrusive composition is not ported")
    n_indices = jnp.asarray(n_indices, dtype=jnp.int32)
    k_indices = jnp.asarray(k_indices, dtype=jnp.int32)
    if n_indices.ndim == 1:
        n_indices = n_indices[:, None]
    if k_indices.ndim == 1:
        k_indices = k_indices[:, None]
    dim = k_indices.shape[0]
    component_count = n_indices.shape[0]
    dtype = jnp.result_type(*(poly.coeffs for poly in data.w))
    component_polys = tuple(_series_component_polynomial(data.w, component, dim) for component in range(component_count))
    result = jnp.zeros((n_indices.shape[1], k_indices.shape[1]), dtype=dtype)
    for row in range(n_indices.shape[1]):
        powers = tuple(int(value) for value in n_indices[:, row].tolist())
        for col in range(k_indices.shape[1]):
            target = tuple(int(value) for value in k_indices[:, col].tolist())
            value = _composition_coefficient(powers, target, component_polys, jnp.zeros((), dtype=dtype))
            result = result.at[row, col].set(value)
    return result


def dfnl_intrusive(
    n_indices: Array,
    w1: tuple[NonAutonomousCoefficientSeries | tuple[MultiIndexPolynomial, ...], ...],
    m_indices: Array,
    data: IntrusiveCompositionData,
) -> tuple[Array, ...]:
    """Compose an intrusive Jacobian with non-autonomous SSM coefficients.

    This ports the core coefficient algebra of
    ``@Manifold/private/dfnl_intrusive.m``. For each harmonic ``W1`` series,
    each nonlinear monomial exponent ``n`` and each target reduced multi-index
    ``m``, the result contains the coefficient of
    ``D(prod_i x_i**n_i)(W0(p)) @ W1(p)`` at ``p^m``.

    Differentiability
    -----------------
    Differentiable with respect to autonomous and non-autonomous coefficient
    values for fixed index sets and polynomial structures. Index matching is
    discrete preprocessing.
    """

    if data.ordering not in {"revlex", "lex"}:
        raise NotImplementedError("Conjugate intrusive Jacobian composition is not ported")
    n_indices = jnp.asarray(n_indices, dtype=jnp.int32)
    m_indices = jnp.asarray(m_indices, dtype=jnp.int32)
    if n_indices.ndim == 1:
        n_indices = n_indices[:, None]
    if m_indices.ndim == 1:
        m_indices = m_indices[:, None]
    dim = m_indices.shape[0]
    component_count = n_indices.shape[0]
    base_dtype = jnp.result_type(*(poly.coeffs for poly in data.w), *[poly.coeffs for series in w1 for poly in _series_terms(series)])
    w0_polys = tuple(_series_component_polynomial(data.w, component, dim) for component in range(component_count))
    outputs = []
    for series in w1:
        terms = _series_terms(series)
        w1_polys = tuple(_series_component_polynomial(terms, component, dim) for component in range(component_count))
        result = jnp.zeros((n_indices.shape[1], m_indices.shape[1]), dtype=base_dtype)
        for row in range(n_indices.shape[1]):
            powers = tuple(int(value) for value in n_indices[:, row].tolist())
            for col in range(m_indices.shape[1]):
                target = tuple(int(value) for value in m_indices[:, col].tolist())
                total = jnp.asarray(0, dtype=base_dtype)
                for component, exponent in enumerate(powers):
                    if exponent == 0:
                        continue
                    derivative_powers = list(powers)
                    derivative_powers[component] -= 1
                    derivative_poly: dict[tuple[int, ...], Array] = {
                        (0,) * dim: jnp.asarray(exponent, dtype=base_dtype)
                    }
                    for base_component, power in enumerate(derivative_powers):
                        if power == 0:
                            continue
                        powered = _poly_power_truncated(w0_polys[base_component], power, target, jnp.zeros((), dtype=base_dtype))
                        derivative_poly = _poly_mul_truncated(derivative_poly, powered, target)
                    product = _poly_mul_truncated(derivative_poly, w1_polys[component], target)
                    total = total + product.get(target, jnp.asarray(0, dtype=base_dtype))
                result = result.at[row, col].set(total)
        outputs.append(result)
    return tuple(outputs)


def autonomous_invariance_residual(
    a_matrix: Array,
    b_matrix: Array,
    w: tuple[MultiIndexPolynomial, ...],
    r: tuple[MultiIndexPolynomial, ...],
    points: Array,
    nonlinear_terms: tuple[MultiIndexPolynomial | Array, ...] = (),
    nonlinear_function: Callable[[Array], Array] | None = None,
) -> Array:
    """Evaluate autonomous SSM invariance residual norms.

    This ports the autonomous branch of
    ``@Manifold/compuate_invariance_residual.m`` as a functional API:
    ``B DW(p) R(p) - A W(p) - F(W(p))`` is evaluated at column-wise reduced
    coordinates and reduced to one Euclidean residual norm per point.

    Differentiability
    -----------------
    Piecewise differentiable for fixed polynomial structures and differentiable
    nonlinear callables. The final Euclidean norm is not differentiable at zero
    residual.
    """

    point_matrix = jnp.asarray(points)
    if point_matrix.ndim == 1:
        point_matrix = point_matrix[:, None]
    output_dim = w[0].coeffs.shape[0]
    z = jnp.zeros((output_dim, point_matrix.shape[1]), dtype=jnp.result_type(point_matrix, *(poly.coeffs for poly in w)))
    dw = jnp.zeros((output_dim, point_matrix.shape[0], point_matrix.shape[1]), dtype=z.dtype)
    reduced = jnp.zeros((point_matrix.shape[0], point_matrix.shape[1]), dtype=jnp.result_type(point_matrix, *(poly.coeffs for poly in r)))
    for poly in w:
        z = z + jnp.real(expand_multiindex(poly, point_matrix))
        dw = dw + _expand_multiindex_derivative_explicit(poly, point_matrix)
    for poly in r:
        reduced = reduced + expand_multiindex(poly, point_matrix)

    tangent = jnp.einsum("ijp,jp->ip", dw, reduced)
    lhs = jnp.asarray(b_matrix) @ jnp.real(tangent)

    if nonlinear_function is not None:
        nonlinear = jnp.stack([jnp.asarray(nonlinear_function(z[:, idx])) for idx in range(z.shape[1])], axis=1)
    elif nonlinear_terms:
        nonlinear = evaluate_polynomial_terms(nonlinear_terms, z)
    else:
        nonlinear = jnp.zeros_like(z)
    rhs = jnp.asarray(a_matrix) @ z + nonlinear
    residual = lhs - rhs
    return jnp.sqrt(jnp.sum(residual**2, axis=0))


def compute_auto_invariance_error(
    a_matrix: Array,
    b_matrix: Array,
    w: tuple[MultiIndexPolynomial, ...],
    r: tuple[MultiIndexPolynomial, ...],
    rhos: Array,
    orders: Array,
    ntheta: int,
    nonlinear_terms: tuple[MultiIndexPolynomial | Array, ...] = (),
    nonlinear_function: Callable[[Array], Array] | None = None,
    *,
    nalpha: int | None = None,
) -> Array:
    """Average autonomous invariance residuals on 2D/4D SSM sampling grids.

    This ports ``@Manifold/compute_auto_invariance_error.m`` without object
    mutation or printing. For 2D SSMs the samples are
    ``[rho exp(i theta); rho exp(-i theta)]``. For 4D SSMs the MATLAB
    ``alpha`` split between two modal pairs is used.

    Differentiability
    -----------------
    Piecewise differentiable for fixed sampling dimensions and polynomial
    structures. The residual norm is non-smooth at zero residual.
    """

    dim = w[0].ind.shape[1]
    if dim not in {2, 4}:
        raise ValueError("Only 2D and 4D SSMs are supported")
    orders_array = jnp.ravel(jnp.asarray(orders, dtype=jnp.int32))
    rhos_array = jnp.ravel(jnp.asarray(rhos))
    if bool(jnp.max(orders_array) > len(r)):
        raise ValueError("Some requested expansion orders are higher than the approximation order of SSM")
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta + 1)[:-1]
    rows = []
    for order in [int(value) for value in orders_array.tolist()]:
        values = []
        for rho in rhos_array:
            if dim == 2:
                p1 = rho * jnp.exp(1j * theta)
                samples = jnp.vstack([p1, jnp.conj(p1)])
                residuals = autonomous_invariance_residual(a_matrix, b_matrix, w[:order], r[:order], samples, nonlinear_terms, nonlinear_function)
            else:
                if nalpha is None:
                    raise ValueError("nalpha is required for 4D SSMs")
                alphas = jnp.linspace(0.0, jnp.pi / 2.0, nalpha)
                collected = []
                for alpha in alphas:
                    rho1 = rho * jnp.cos(alpha)
                    rho2 = rho * jnp.sin(alpha)
                    for angle in theta:
                        p1 = rho1 * jnp.exp(1j * angle) * jnp.ones((ntheta,), dtype=jnp.result_type(rho, 1j))
                        p3 = rho2 * jnp.exp(1j * theta)
                        samples = jnp.vstack([p1, jnp.conj(p1), p3, jnp.conj(p3)])
                        collected.append(autonomous_invariance_residual(a_matrix, b_matrix, w[:order], r[:order], samples, nonlinear_terms, nonlinear_function))
                residuals = jnp.concatenate(collected)
            values.append(jnp.mean(residuals))
        rows.append(jnp.stack(values))
    return jnp.stack(rows)


def coeffs_composition(
    w0: tuple[MultiIndexPolynomial, ...],
    h: tuple[Array, ...],
    order: int,
    ordering: Ordering = "revlex",
) -> Array:
    """Compute power-series composition coefficients for one order.

    This ports the lexicographic/reverse-lexicographic branches of
    ``@Manifold/private/coeffs_composition.m``. The conjugate-order branch is
    intentionally left for the cohomological-solver layer.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values for fixed order and
    ordering. The combinatorial indexing is discrete preprocessing.
    """

    if ordering not in {"lex", "revlex"}:
        raise NotImplementedError("Only lex and revlex composition are ported")
    dim = h[0].shape[1]
    z_k = comb(order + dim - 1, dim - 1)
    output_dim = h[0].shape[0]
    k_indices = _multi_indices(dim, order, ordering)
    h_k = jnp.zeros((output_dim, z_k, order), dtype=jnp.result_type(*(item.coeffs for item in w0), *h))

    shifted = k_indices + (order + 1) * (k_indices == 0)
    kj = jnp.min(shifted, axis=0)
    ikj = jnp.argmin(shifted, axis=0)

    for ord_value in range(1, order):
        w_coeffs = w0[ord_value - 1].coeffs
        if w_coeffs.size == 0:
            continue
        g = _multi_indices(dim, order - ord_value, ordering)
        gmink, i_k, i_g = multi_subtraction(k_indices, g, "Arbitrary")
        if gmink.shape[1] == 0:
            continue
        i_gmk = multi_index_2_ordering(gmink, ordering) - 1
        pos = ikj[i_k]
        gmk_j = gmink[pos, jnp.arange(i_k.shape[0])]
        h_prev = h[order - ord_value - 1]
        if h_prev.ndim == 2:
            h_prev = h_prev[:, :, None]
        coeff = (w_coeffs[:, i_gmk] * gmk_j[None, :])[:, :, None] * h_prev[:, i_g, :]
        for target in sorted(set(int(value) for value in i_k.tolist())):
            mask = i_k == target
            update = jnp.sum(coeff[:, mask, :], axis=1)
            h_k = h_k.at[:, target, 1 : order - ord_value + 1].add(update)

    scale = (1.0 / kj)[:, None] * jnp.arange(1, order + 1, dtype=h_k.dtype)[None, :]
    h_k = jnp.transpose(jnp.transpose(h_k, (1, 2, 0)) * scale[:, :, None], (2, 0, 1))
    if len(w0) >= order and w0[order - 1].coeffs.size:
        h_k = h_k.at[:, :, 0].set(w0[order - 1].coeffs)
    return h_k


def coeffs_mixed_terms(
    order: int,
    m_order: int,
    w: tuple[MultiIndexPolynomial, ...],
    r: tuple[MultiIndexPolynomial, ...],
    *,
    dim: int,
    output_dim: int,
    mix: Literal["R1", "W1", "aut"] = "aut",
    ordering: Ordering = "revlex",
    explicit_indices: bool = False,
) -> Array:
    """Compute mixed coefficient products in lex/revlex ordering.

    This ports the reverse-lexicographic branch of
    ``@Manifold/private/coeffs_mixed_terms.m`` and also supports lexicographic
    ordering by using the same formula with lex-ordered multi-indices.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values for fixed index
    structure. The index assembly and column matching are discrete.
    """

    if ordering not in {"lex", "revlex"}:
        raise NotImplementedError("Only lex and revlex mixed terms are ported")
    z_k = comb(order + dim - 1, dim - 1)
    result = jnp.zeros((output_dim, z_k), dtype=jnp.result_type(*(item.coeffs for item in w), *(item.coeffs for item in r)))

    if mix == "R1":
        u_order = order + 1 - m_order + 1
    elif mix == "W1":
        u_order = order + 1 - (m_order - 1)
    elif mix == "aut":
        u_order = order - m_order + 1
    else:
        raise ValueError("mix must be 'R1', 'W1', or 'aut'")

    w_poly = w[m_order - 1]
    r_poly = r[u_order - 1]
    if w_poly.coeffs.size == 0 or r_poly.coeffs.size == 0:
        return result

    k_indices = _multi_indices(dim, order, ordering)
    if explicit_indices:
        m_indices = jnp.asarray(w_poly.ind, dtype=jnp.int32).T
        u_indices = jnp.asarray(r_poly.ind, dtype=jnp.int32).T
    else:
        m_indices = _multi_indices(dim, m_order, ordering)
        u_indices = _multi_indices(dim, u_order, ordering)

    summed, i_m, i_u = multi_addition(m_indices, u_indices)
    one = jnp.eye(dim, dtype=jnp.int32)
    k_lookup = {tuple(int(value) for value in k_indices[:, col].tolist()): col for col in range(k_indices.shape[1])}

    for row in range(dim):
        shifted = summed - one[:, row : row + 1]
        keep = jnp.all(shifted >= 0, axis=0)
        shifted_keep = shifted[:, keep]
        i_m_keep = i_m[keep]
        i_u_keep = i_u[keep]
        if shifted_keep.shape[1] == 0:
            continue
        w_values = w_poly.coeffs[:, i_m_keep]
        r_factor = r_poly.coeffs[row, i_u_keep] * m_indices[row, i_m_keep]
        wr = w_values * r_factor[None, :]
        for col in range(shifted_keep.shape[1]):
            key = tuple(int(value) for value in shifted_keep[:, col].tolist())
            target = k_lookup.get(key)
            if target is not None:
                result = result.at[:, target].add(wr[:, col])
    return result
