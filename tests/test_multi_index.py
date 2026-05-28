import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy import (
    MultiIndexPolynomial,
    expand_multiindex,
    multi_addition,
    multi_index_2_ordering,
    multi_index_to_tensor,
    multi_nsumk,
    multi_subtraction,
    nsumk,
    sub2multiind,
    tensor_to_multi_index,
)


def test_nsumk_matches_matlab_source_examples():
    np.testing.assert_array_equal(
        np.asarray(nsumk(3, 2, "nonnegative")),
        np.array(
            [
                [0, 0, 2],
                [0, 1, 1],
                [0, 2, 0],
                [1, 0, 1],
                [1, 1, 0],
                [2, 0, 0],
            ]
        ),
    )
    np.testing.assert_array_equal(np.asarray(nsumk(3, 4, "positive")), np.array([[1, 1, 2], [1, 2, 1], [2, 1, 1]]))


def test_sub2multiind_counts_one_based_subscripts():
    subs = jnp.array([[1, 1, 3], [1, 3, 1], [3, 1, 1], [2, 3, 3]])
    expected = jnp.array([[2, 0, 1], [2, 0, 1], [2, 0, 1], [0, 1, 2]])
    np.testing.assert_array_equal(np.asarray(sub2multiind(subs, 3)), np.asarray(expected))


def test_expand_multiindex_values_and_jax_transforms():
    poly = MultiIndexPolynomial(
        coeffs=jnp.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]]),
        ind=jnp.array([[2, 0], [1, 1], [0, 3]]),
    )
    points = jnp.array([[2.0, -1.0], [3.0, 4.0]])
    expected = jnp.array([[2 * 4 - 6 + 13.5, 2 * 1 + 4 + 32], [-0 + 18 + 27, -12 + 64]])
    np.testing.assert_allclose(np.asarray(expand_multiindex(poly, points)), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(jax.jit(expand_multiindex)(poly, points)), np.asarray(expected))

    def scalar(x):
        return jnp.sum(expand_multiindex(poly, x.reshape(2, 1)))

    grad = jax.grad(scalar)(jnp.array([2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(grad), np.array([14.0, 44.5]), rtol=1e-6)


def test_tensor_multi_index_round_trip_semantics_for_canonical_terms():
    coeffs = jnp.array([[5.0, -2.0], [1.5, 0.0]])
    ind = jnp.array([[2, 0], [1, 1]])
    tensor = multi_index_to_tensor(coeffs, ind)
    poly = tensor_to_multi_index(tensor)
    points = jnp.array([[2.0], [3.0]])
    original = MultiIndexPolynomial(coeffs=coeffs, ind=ind)
    np.testing.assert_allclose(np.asarray(expand_multiindex(poly, points)), np.asarray(expand_multiindex(original, points)))


def test_multi_addition_subtraction_and_ordering():
    a = jnp.array([[1, 0], [0, 2]])
    b = jnp.array([[0, 1], [1, 0]])
    added, ia, ib = multi_addition(a, b)
    np.testing.assert_array_equal(np.asarray(added), np.array([[1, 0, 2, 1], [1, 3, 0, 2]]))
    np.testing.assert_array_equal(np.asarray(ia), np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(np.asarray(ib), np.array([0, 0, 1, 1]))

    subbed, im, isub = multi_subtraction(a, b)
    np.testing.assert_array_equal(np.asarray(subbed), np.array([[0, 0], [1, 0]]))
    np.testing.assert_array_equal(np.asarray(im), np.array([1, 0]))
    np.testing.assert_array_equal(np.asarray(isub), np.array([0, 1]))

    ordering = multi_index_2_ordering(jnp.array([[0, 1, 2], [2, 1, 0]]), "lex")
    np.testing.assert_array_equal(np.asarray(ordering), np.array([1, 2, 3]))


def test_multi_subtraction_identity_and_unit_match_matlab_reference():
    minuend = jnp.array([[1, 0, 2], [0, 2, 1], [1, 1, 0]])
    identity = jnp.eye(3, dtype=jnp.int32)
    res, i_minu, i_subt = multi_subtraction(minuend, identity, "Identity")
    np.testing.assert_array_equal(
        np.asarray(res),
        np.array([[0, 1, 0, 2, 1, 0], [0, 1, 1, 0, 0, 2], [1, 0, 1, 0, 0, 0]]),
    )
    np.testing.assert_array_equal(np.asarray(i_minu), np.array([0, 2, 1, 2, 0, 1]))
    np.testing.assert_array_equal(np.asarray(i_subt), np.array([0, 0, 1, 1, 2, 2]))

    unit = jnp.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])
    res, i_minu, i_subt = multi_subtraction(minuend, unit, "Unit")
    np.testing.assert_array_equal(
        np.asarray(res),
        np.array([[0, 1, 0, 1, 0, 2, 1, 0], [0, 1, 0, 1, 1, 0, 0, 2], [1, 0, 1, 0, 1, 0, 0, 0]]),
    )
    np.testing.assert_array_equal(np.asarray(i_minu), np.array([0, 2, 0, 2, 1, 2, 0, 1]))
    np.testing.assert_array_equal(np.asarray(i_subt), np.array([0, 0, 3, 3, 1, 1, 2, 2]))


def test_multi_nsumk_modes_match_matlab_reference():
    g, g_comb, nv_un, nv_ic = multi_nsumk(jnp.array([2]), jnp.array([[2], [1]]))
    np.testing.assert_array_equal(
        np.asarray(g[0][0]),
        np.array([[[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0]], [[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]]]),
    )
    np.testing.assert_array_equal(np.asarray(g_comb), np.array([[6]]))
    np.testing.assert_array_equal(np.asarray(nv_un), np.array([2]))
    np.testing.assert_array_equal(np.asarray(nv_ic), np.array([0]))

    g, g_comb, _, _ = multi_nsumk(jnp.array([2]), jnp.array([[2], [1]]), remove_zero=True)
    np.testing.assert_array_equal(
        np.asarray(g[0][0]),
        np.array([[[1, 2, 0, 1], [1, 0, 2, 1]], [[0, 0, 1, 1], [1, 1, 0, 0]]]),
    )
    np.testing.assert_array_equal(np.asarray(g_comb), np.array([[4]]))

    g, g_comb, nv_un, nv_ic, multiplicity = multi_nsumk(
        jnp.array([2, 1, 2]), jnp.array([[2], [1]]), remove_zero=True, unique=True
    )
    np.testing.assert_array_equal(np.asarray(g[0][0]), np.array([[[2]], [[1]]]))
    np.testing.assert_array_equal(np.asarray(g[0][1]), np.array([[[1, 0], [1, 2]], [[0, 1], [1, 0]]]))
    np.testing.assert_array_equal(np.asarray(g_comb), np.array([[1, 2]]))
    np.testing.assert_array_equal(np.asarray(nv_un), np.array([1, 2]))
    np.testing.assert_array_equal(np.asarray(nv_ic), np.array([1, 0, 1]))
    np.testing.assert_array_equal(np.asarray(multiplicity[0][0]), np.array([1]))
    np.testing.assert_array_equal(np.asarray(multiplicity[0][1]), np.array([2, 2]))
