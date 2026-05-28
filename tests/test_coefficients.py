import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy import (
    MultiIndexPolynomial,
    coeffs_conj2full,
    coeffs_conj2lex,
    coeffs_lex2revlex,
    coeffs_output,
    conjugate_flip,
    conjugate_ordering,
    number_of_multis,
)


def test_coeffs_conj2full_taylor_matches_matlab_reference():
    coeff_part = MultiIndexPolynomial(
        coeffs=jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]),
        ind=jnp.zeros((0, 2), dtype=jnp.int32),
    )
    out = coeffs_conj2full(coeff_part, None, 1, jnp.array([0, 2, 1]), "TaylorCoeffs")
    np.testing.assert_allclose(np.asarray(jnp.real(out.coeffs)), np.array([[1, 1, 2], [3, 3, 4]]))
    np.testing.assert_allclose(np.asarray(jnp.imag(out.coeffs)), np.array([[1, -1, 2], [3, -3, 4]]))

    row_out = coeffs_conj2full(coeff_part, jnp.array([1, 0]), 1, jnp.array([0, 2, 1]), "TaylorCoeffs")
    np.testing.assert_allclose(np.asarray(jnp.real(row_out.coeffs)), np.array([[1, 3, 2], [3, 1, 4]]))
    np.testing.assert_allclose(np.asarray(jnp.imag(row_out.coeffs)), np.array([[1, -3, 2], [3, -1, 4]]))


def test_coeffs_conj2full_comp_and_grad():
    coeffs = jnp.zeros((2, 2, 2), dtype=jnp.complex64)
    coeffs = coeffs.at[:, :, 0].set(jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]))
    coeffs = coeffs.at[:, :, 1].set(jnp.array([[5 + 1j, 6 + 2j], [7 + 3j, 8 + 4j]]))
    out = coeffs_conj2full(coeffs, None, 1, jnp.array([0, 2, 1]), "CompCoeffs")
    expected_real = np.array([[[1, 5], [1, 5], [2, 6]], [[3, 7], [3, 7], [4, 8]]])
    expected_imag = np.array([[[1, 1], [-1, -1], [2, 2]], [[3, 3], [-3, -3], [4, 4]]])
    np.testing.assert_allclose(np.asarray(jnp.real(out)), expected_real)
    np.testing.assert_allclose(np.asarray(jnp.imag(out)), expected_imag)

    def scalar(real_coeffs):
        complex_coeffs = real_coeffs.reshape(2, 2, 2).astype(jnp.complex64)
        return jnp.real(jnp.sum(coeffs_conj2full(complex_coeffs, None, 1, jnp.array([0, 2, 1]), "CompCoeffs")))

    grad = jax.grad(scalar)(jnp.arange(8.0))
    np.testing.assert_allclose(np.asarray(grad), np.array([2, 2, 1, 1, 2, 2, 1, 1]))


def test_coeffs_lex2revlex_and_grad():
    first = MultiIndexPolynomial(coeffs=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ind=jnp.array([[0, 2], [1, 1], [2, 0]]))
    second = MultiIndexPolynomial(coeffs=jnp.array([[7.0, 8.0], [9.0, 10.0]]), ind=jnp.array([[0, 1], [1, 0]]))
    flipped = coeffs_lex2revlex((first, second), "TaylorCoeff")
    np.testing.assert_allclose(np.asarray(flipped[0].coeffs), np.array([[3, 2, 1], [6, 5, 4]]))
    np.testing.assert_allclose(np.asarray(flipped[1].coeffs), np.array([[8, 7], [10, 9]]))

    grad = jax.grad(lambda x: jnp.sum(coeffs_lex2revlex((x.reshape(2, 3),), "CompCoeff")[0]))(jnp.arange(6.0))
    np.testing.assert_allclose(np.asarray(grad), np.ones(6))


def test_conjugate_ordering_source_derived_small_cases():
    np.testing.assert_array_equal(np.asarray(number_of_multis(3, 3)), np.array([3, 6, 10]))
    np.testing.assert_array_equal(np.asarray(conjugate_flip(1, 1)), np.array([1, 0, 2]))

    ordering = conjugate_ordering(3, l_r=1, l_i=1)
    np.testing.assert_array_equal(np.asarray(ordering.z_cci), np.array([2, 4, 6]))
    np.testing.assert_array_equal(np.asarray(ordering.conj2revlex[0]), np.array([0, 2, 1]))
    np.testing.assert_array_equal(np.asarray(ordering.revlex2conj[1]), np.array([0, 2, 1, 5, 4, 3]))


def test_coeffs_conj2lex_uses_ordering_maps():
    ordering = conjugate_ordering(1, l_r=1, l_i=1)
    w = (MultiIndexPolynomial(coeffs=jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]]), ind=jnp.zeros((0, 3))),)
    r = (MultiIndexPolynomial(coeffs=jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j], [5 + 5j, 6 + 6j]]), ind=jnp.zeros((0, 3))),)
    w_lex, r_lex = coeffs_conj2lex(ordering, 1, w, r)
    np.testing.assert_allclose(np.asarray(jnp.real(w_lex[0].coeffs)), np.array([[2, 1, 1], [4, 3, 3], [6, 5, 5]]))
    np.testing.assert_allclose(np.asarray(jnp.real(r_lex[0].coeffs)), np.array([[2, 3, 1], [4, 1, 3], [6, 5, 5]]))

    def scalar(values):
        poly = MultiIndexPolynomial(coeffs=values.reshape(3, 2), ind=jnp.zeros((0, 3)))
        w_full, r_full = coeffs_conj2lex(ordering, 1, (poly,), (poly,))
        return jnp.sum(w_full[0].coeffs) + jnp.sum(r_full[0].coeffs)

    grad = jax.grad(scalar)(jnp.arange(6.0))
    assert grad.shape == (6,)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_coeffs_output_matches_matlab_reference():
    w = (
        MultiIndexPolynomial(coeffs=jnp.array([[1.0, 0.0], [0.0, 2.0]]), ind=jnp.zeros((0, 2))),
        MultiIndexPolynomial(coeffs=jnp.array([[0.0, 3.0, 0.0], [4.0, 0.0, 0.0]]), ind=jnp.zeros((0, 2))),
    )
    r = (
        MultiIndexPolynomial(coeffs=jnp.array([[1.0, 0.0], [0.0, 0.0]]), ind=jnp.zeros((0, 2))),
        MultiIndexPolynomial(coeffs=jnp.array([[0.0, 5.0, 0.0], [6.0, 0.0, 0.0]]), ind=jnp.zeros((0, 2))),
    )
    w_out, r_out = coeffs_output(w, r, 2)
    np.testing.assert_allclose(np.asarray(w_out[0].coeffs), np.array([[1, 0], [0, 2]]))
    np.testing.assert_array_equal(np.asarray(w_out[0].ind), np.array([[0, 1], [1, 0]]))
    np.testing.assert_allclose(np.asarray(w_out[1].coeffs), np.array([[0, 3], [4, 0]]))
    np.testing.assert_array_equal(np.asarray(w_out[1].ind), np.array([[0, 2], [1, 1]]))
    np.testing.assert_allclose(np.asarray(r_out[0].coeffs), np.array([[1], [0]]))
    np.testing.assert_array_equal(np.asarray(r_out[0].ind), np.array([[0, 1]]))
    np.testing.assert_allclose(np.asarray(r_out[1].coeffs), np.array([[0, 5], [6, 0]]))
    np.testing.assert_array_equal(np.asarray(r_out[1].ind), np.array([[0, 2], [1, 1]]))
