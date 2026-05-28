"""Multi-index polynomial utilities ported from SSMTool.

Differentiability summary
-------------------------
``expand_multiindex`` is differentiable with respect to coefficients and
evaluation points away from the usual non-smooth/undefined real-power cases.
``expand_multiindex_derivative`` is differentiable under the same assumptions.
The indexing/conversion utilities are discrete preprocessing operations and are
not differentiable.
"""

from __future__ import annotations

from itertools import combinations, product
from math import comb
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jnp.ndarray


class MultiIndexPolynomial(NamedTuple):
    """Polynomial coefficients in SSMTool multi-index format.

    Parameters
    ----------
    coeffs:
        Matrix with shape ``(output_dim, n_terms)``.
    ind:
        Integer exponent matrix with shape ``(n_terms, input_dim)``.

    Differentiability
    -----------------
    Not differentiable as a container. Evaluation with ``expand_multiindex`` is
    differentiable where the represented polynomial is differentiable.
    """

    coeffs: Array
    ind: Array


def nsumk(n: int, k: int, option: str = "nonnegative") -> Array:
    """List integer ``n``-tuples summing to ``k``.

    This follows the MATLAB ``nsumk`` ordering induced by ``nchoosek``.

    Differentiability
    -----------------
    Not differentiable. This is a discrete combinatorial preprocessing helper.
    """

    if n < 1:
        raise ValueError("n must be positive")
    if option == "nonnegative":
        rows = []
        for dividers in combinations(range(1, k + n), n - 1):
            points = (0, *dividers, k + n)
            rows.append([points[i + 1] - points[i] - 1 for i in range(n)])
        return jnp.asarray(rows, dtype=jnp.int32)
    if option == "positive":
        if k < n:
            return jnp.zeros((0, n), dtype=jnp.int32)
        rows = []
        for dividers in combinations(range(2, k + 1), n - 1):
            points = (1, *dividers, k + 1)
            rows.append([points[i + 1] - points[i] for i in range(n)])
        return jnp.asarray(rows, dtype=jnp.int32)
    raise ValueError("option must be 'nonnegative' or 'positive'")


def sub2multiind(subs: Array, r: int) -> Array:
    """Convert one-based tensor subscripts to multi-index exponents.

    ``subs`` has shape ``(n_subscripts, degree)`` and contains MATLAB-style
    one-based variable indices.

    Differentiability
    -----------------
    Not differentiable. This maps discrete subscripts to exponents.
    """

    subs = jnp.asarray(subs, dtype=jnp.int32)
    return jnp.sum(subs[..., None] == jnp.arange(1, r + 1), axis=1).astype(jnp.int32)


def expand_multiindex(poly: MultiIndexPolynomial, points: Array) -> Array:
    """Evaluate a multi-index polynomial at column-wise points.

    Parameters
    ----------
    poly:
        Multi-index polynomial ``(C, I)``.
    points:
        Matrix with shape ``(input_dim, n_points)``.

    Returns
    -------
    Array with shape ``(output_dim, n_points)`` matching MATLAB
    ``expand_multiindex``.

    Differentiability
    -----------------
    Differentiable. Representative tests exercise ``jax.grad`` and ``jax.jit``.
    """

    coeffs = jnp.asarray(poly.coeffs)
    ind = jnp.asarray(poly.ind)
    points = jnp.asarray(points)
    if ind.size == 0:
        return jnp.zeros((coeffs.shape[0], points.shape[1]), dtype=points.dtype)
    monomials = jnp.prod(points.T[:, None, :] ** ind[None, :, :], axis=-1)
    return coeffs @ monomials.T


def expand_multiindex_derivative(poly: MultiIndexPolynomial, points: Array) -> Array:
    """Evaluate the Jacobian of a multi-index polynomial at each point.

    Returns an array with shape ``(output_dim, input_dim, n_points)``.

    Differentiability
    -----------------
    Differentiable where the polynomial derivative is defined. Representative
    tests compare this routine with ``jax.jacfwd``.
    """

    coeffs = jnp.asarray(poly.coeffs)
    ind = jnp.asarray(poly.ind)
    points = jnp.asarray(points)
    if ind.size == 0:
        return jnp.zeros((coeffs.shape[0], points.shape[0], points.shape[1]), dtype=points.dtype)

    def eval_one(point: Array) -> Array:
        return expand_multiindex(poly, point[:, None])[:, 0]

    jac = jnp.stack([jax.jacfwd(eval_one)(p) for p in points.T])
    return jnp.moveaxis(jac, 0, -1)


def _lexsort_rows(rows: Array) -> Array:
    rows = jnp.asarray(rows)
    if rows.shape[0] == 0:
        return rows
    order = sorted(range(rows.shape[0]), key=lambda i: tuple(int(x) for x in rows[i]))
    return rows[jnp.asarray(order, dtype=jnp.int32)]


def tensor_to_multi_index(tensor: Array) -> MultiIndexPolynomial:
    """Convert a dense tensor polynomial to multi-index coefficients.

    The first tensor axis is the polynomial output index; all remaining axes
    are one-based tensor variable modes in the MATLAB reference. This dense
    implementation sums coefficients over all permutations of each monomial.

    Differentiability
    -----------------
    Not differentiable as implemented because it performs discrete sparse
    extraction and grouping. Use ``expand_tensor`` or ``expand_multiindex`` in
    differentiable code paths.
    """

    tensor = jnp.asarray(tensor)
    if tensor.ndim < 2:
        raise ValueError("tensor must have at least two dimensions")
    output_dim = tensor.shape[0]
    variable_dim = tensor.shape[1]
    nonzero = jnp.argwhere(tensor != 0, size=None)
    if nonzero.shape[0] == 0:
        return MultiIndexPolynomial(
            jnp.zeros((output_dim, 0), dtype=tensor.dtype),
            jnp.zeros((0, variable_dim), dtype=jnp.int32),
        )

    groups: dict[tuple[int, ...], Array] = {}
    for raw_idx in nonzero.tolist():
        out = raw_idx[0]
        exponents = [0] * variable_dim
        for sub in raw_idx[1:]:
            exponents[sub] += 1
        key = tuple(exponents)
        if key not in groups:
            groups[key] = jnp.zeros((output_dim,), dtype=tensor.dtype)
        groups[key] = groups[key].at[out].add(tensor[tuple(raw_idx)])

    ordered_keys = sorted(groups)
    ind = jnp.asarray(ordered_keys, dtype=jnp.int32)
    coeffs = jnp.stack([groups[key] for key in ordered_keys], axis=1)
    return MultiIndexPolynomial(coeffs, ind)


def multi_index_to_tensor(coeffs: Array, ind: Array) -> Array:
    """Convert multi-index coefficients to a sparse-equivalent dense tensor.

    Matching the MATLAB routine, each monomial coefficient is placed at one
    canonical subscript, not symmetrically distributed across permutations.

    Differentiability
    -----------------
    Not differentiable. This is a discrete representation conversion.
    """

    coeffs = jnp.asarray(coeffs)
    ind = jnp.asarray(ind, dtype=jnp.int32)
    if ind.shape[0] == 0:
        raise ValueError("ind must contain at least one multi-index")
    degree = int(jnp.sum(ind[0]))
    variable_dim = ind.shape[1]
    tensor = jnp.zeros((coeffs.shape[0], *([variable_dim] * degree)), dtype=coeffs.dtype)
    for term in range(ind.shape[0]):
        subs = []
        for variable in range(variable_dim):
            subs.extend([variable] * int(ind[term, variable]))
        for output in range(coeffs.shape[0]):
            value = coeffs[output, term]
            if value != 0:
                tensor = tensor.at[(output, *subs)].set(value)
    return tensor


def multi_addition(summand1: Array, summand2: Array) -> tuple[Array, Array, Array]:
    """Add all columns of two multi-index matrices.

    Differentiability
    -----------------
    Not differentiable. This is discrete index algebra.
    """

    summand1 = jnp.asarray(summand1, dtype=jnp.int32)
    summand2 = jnp.asarray(summand2, dtype=jnp.int32)
    n1 = summand1.shape[1]
    n2 = summand2.shape[1]
    i_sum1 = jnp.tile(jnp.arange(n1), n2)
    i_sum2 = jnp.ravel(jnp.tile(jnp.arange(n2), (n1, 1)).T)
    return summand1[:, i_sum1] + summand2[:, i_sum2], i_sum1, i_sum2


def multi_subtraction(minuend: Array, subtrahend: Array, kind: str = "Arbitrary") -> tuple[Array, Array, Array]:
    """Subtract multi-index columns and keep nonnegative results.

    Indices returned are zero-based Python indices.

    Differentiability
    -----------------
    Not differentiable. This is discrete index algebra with branch selection.
    """

    minuend = jnp.asarray(minuend, dtype=jnp.int32)
    subtrahend = jnp.asarray(subtrahend, dtype=jnp.int32)
    if kind == "Arbitrary":
        n_minu = minuend.shape[1]
        n_subt = subtrahend.shape[1]
        i_subt = jnp.ravel(jnp.tile(jnp.arange(n_subt), (n_minu, 1)).T)
        i_minu = jnp.tile(jnp.arange(n_minu), n_subt)
        res = minuend[:, i_minu] - subtrahend[:, i_subt]
        keep = jnp.all(res >= 0, axis=0)
        return res[:, keep], i_minu[keep], i_subt[keep]
    if kind == "Identity":
        pairs = []
        for row in range(minuend.shape[0]):
            for col in range(minuend.shape[1]):
                if int(minuend[row, col]) != 0:
                    pairs.append((col, row))
        i_minu = jnp.asarray([col for col, _ in pairs], dtype=jnp.int32)
        i_subt = jnp.asarray([row for _, row in pairs], dtype=jnp.int32)
        return minuend[:, i_minu] - subtrahend[:, i_subt], i_minu, i_subt
    if kind == "Unit":
        unit_pairs = []
        for row in range(subtrahend.shape[0]):
            for col in range(subtrahend.shape[1]):
                if int(subtrahend[row, col]) != 0:
                    unit_pairs.append((col, row))
        if len(unit_pairs) != subtrahend.shape[1]:
            raise ValueError("Unit mode expects each subtrahend column to have exactly one nonzero entry")
        pairs = []
        for subt_col, row in unit_pairs:
            for minu_col in range(minuend.shape[1]):
                if int(minuend[row, minu_col]) != 0:
                    pairs.append((minu_col, subt_col))
        i_minu = jnp.asarray([col for col, _ in pairs], dtype=jnp.int32)
        i_subt = jnp.asarray([col for _, col in pairs], dtype=jnp.int32)
        return minuend[:, i_minu] - subtrahend[:, i_subt], i_minu, i_subt
    raise ValueError("kind must be 'Arbitrary', 'Identity', or 'Unit'")


def _unique_with_inverse(values: list[int]) -> tuple[list[int], list[int]]:
    unique_values = sorted(set(values))
    inverse = [unique_values.index(value) for value in values]
    return unique_values, inverse


def _multi_nsumk_single(n_vectors: int, vector: Array) -> Array:
    pieces = []
    for entry in jnp.asarray(vector, dtype=jnp.int32).tolist():
        if entry == 1 and n_vectors == 1:
            piece = jnp.ones((1, 1), dtype=jnp.int32)
        else:
            piece = nsumk(n_vectors, int(entry), "nonnegative").T
        pieces.append(piece)

    pages = []
    ranges = [range(piece.shape[1]) for piece in pieces]
    for reversed_choice in product(*reversed(ranges)):
        choice = tuple(reversed(reversed_choice))
        page = jnp.stack([pieces[row][:, choice[row]] for row in range(len(pieces))], axis=0)
        pages.append(page)
    if not pages:
        return jnp.zeros((vector.shape[0], n_vectors, 0), dtype=jnp.int32)
    return jnp.stack(pages, axis=2)


def _remove_zero_vector_pages(combinations_: Array) -> Array:
    if combinations_.shape[2] == 0:
        return combinations_
    vector_sums = jnp.sum(combinations_, axis=0)
    keep = jnp.all(vector_sums != 0, axis=0)
    return combinations_[:, :, keep]


def _stable_unique_pages(combinations_: Array) -> tuple[Array, Array]:
    pages: list[Array] = []
    counts: list[int] = []
    keys: list[tuple[int, ...]] = []
    for page_index in range(combinations_.shape[2]):
        page = combinations_[:, :, page_index]
        columns = [tuple(int(value) for value in page[:, col].tolist()) for col in range(page.shape[1])]
        order = sorted(range(len(columns)), key=lambda idx: columns[idx])
        sorted_page = page[:, jnp.asarray(order, dtype=jnp.int32)]
        key = tuple(int(value) for value in sorted_page.T.reshape(-1).tolist())
        if key in keys:
            counts[keys.index(key)] += 1
        else:
            keys.append(key)
            pages.append(sorted_page)
            counts.append(1)
    if not pages:
        empty_counts = jnp.zeros((0,), dtype=jnp.int32)
        return combinations_, empty_counts
    return jnp.stack(pages, axis=2), jnp.asarray(counts, dtype=jnp.int32)


def multi_nsumk(
    n_vectors: Array,
    vectors: Array,
    *,
    remove_zero: bool = False,
    unique: bool = False,
) -> tuple[tuple[tuple[Array, ...], ...], Array, Array, Array] | tuple[tuple[tuple[Array, ...], ...], Array, Array, Array, tuple[tuple[Array, ...], ...]]:
    """Partition multi-index vectors into sums of integer vectors.

    This ports ``@Manifold/private/multi_nsumk.m``. The returned ``g`` is a
    nested tuple with shape ``(n_input_vectors, n_unique_n_vectors)``; each
    entry has shape ``(vector_dim, n_vectors, n_combinations)``.

    Unlike MATLAB, ``nv_ic`` is zero-based to match Python indexing.

    Differentiability
    -----------------
    Not differentiable. This is discrete combinatorial preprocessing with
    optional zero-vector filtering and permutation deduplication.
    """

    nv_list = [int(value) for value in jnp.ravel(jnp.asarray(n_vectors, dtype=jnp.int32)).tolist()]
    vectors = jnp.asarray(vectors, dtype=jnp.int32)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    nv_unique, nv_inverse = _unique_with_inverse(nv_list)

    all_rows: list[tuple[Array, ...]] = []
    multiplicity_rows: list[tuple[Array, ...]] = []
    counts = []
    for vector_col in range(vectors.shape[1]):
        row_entries = []
        multiplicity_entries = []
        count_row = []
        for n_value in nv_unique:
            combos = _multi_nsumk_single(n_value, vectors[:, vector_col])
            if remove_zero:
                combos = _remove_zero_vector_pages(combos)
            if unique:
                combos, multiplicity = _stable_unique_pages(combos)
            else:
                multiplicity = jnp.ones((combos.shape[2],), dtype=jnp.int32)
            row_entries.append(combos)
            multiplicity_entries.append(multiplicity)
            count_row.append(combos.shape[2])
        all_rows.append(tuple(row_entries))
        multiplicity_rows.append(tuple(multiplicity_entries))
        counts.append(count_row)

    result = (
        tuple(all_rows),
        jnp.asarray(counts, dtype=jnp.int32),
        jnp.asarray(nv_unique, dtype=jnp.int32),
        jnp.asarray(nv_inverse, dtype=jnp.int32),
    )
    if unique:
        return (*result, tuple(multiplicity_rows))
    return result


def multi_index_2_ordering(multi_indices: Array, ordering: str = "lex", lex2conj: dict[int, Array] | None = None) -> Array:
    """Return one-based MATLAB ordering positions for multi-index columns.

    Differentiability
    -----------------
    Not differentiable. This is a discrete combinatorial lookup.
    """

    m = jnp.asarray(multi_indices, dtype=jnp.int32)
    if m.size == 0:
        return jnp.asarray([], dtype=jnp.int32)
    if m.shape[0] == 1:
        return jnp.ones((m.shape[1],), dtype=jnp.int32)

    positions = []
    length = m.shape[0]
    for col in range(m.shape[1]):
        total = int(jnp.sum(m[:, col]))
        bij = 0
        first = int(m[0, col])
        for i in range(first):
            bij += comb(total - i + length - 2, length - 2)
        prefix = first
        for row in range(1, length - 1):
            value = int(m[row, col])
            for i in range(value):
                bij += comb(total - prefix - i + length - row - 2, length - row - 2)
            prefix += value
        pos = bij + 1
        if ordering in {"revlex", "conjugate"}:
            pos = comb(total + length - 1, length - 1) + 1 - pos
        if ordering == "conjugate":
            if lex2conj is None:
                raise ValueError("lex2conj is required for conjugate ordering")
            idx = list(map(int, jnp.asarray(lex2conj[total]).tolist()))
            pos = idx.index(pos) + 1
        elif ordering != "lex" and ordering != "revlex":
            raise ValueError("ordering must be 'lex', 'revlex', or 'conjugate'")
        positions.append(pos)
    return jnp.asarray(positions, dtype=jnp.int32)
