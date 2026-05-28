"""Manifold coefficient algebra kernels."""

from __future__ import annotations

from math import comb
from typing import Literal, NamedTuple

import jax.numpy as jnp

from ssmtoolpy.multi_index import MultiIndexPolynomial, multi_addition, multi_index_2_ordering, multi_subtraction, nsumk


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
