"""Manifold coefficient algebra kernels."""

from __future__ import annotations

from math import comb
from typing import Callable, Literal, NamedTuple

import jax.numpy as jnp

from ssmtoolpy.multi_index import MultiIndexPolynomial, multi_addition, multi_index_2_ordering, multi_nsumk, multi_subtraction, nsumk


Array = jnp.ndarray
Ordering = Literal["lex", "revlex"]
ResonanceOrder = Literal["zero", "k"]


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
