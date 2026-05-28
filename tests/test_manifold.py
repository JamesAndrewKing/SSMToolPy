import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy import MultiIndexPolynomial, coeffs_composition, coeffs_mixed_terms


def _w_fixture():
    return (
        MultiIndexPolynomial(coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
        MultiIndexPolynomial(coeffs=jnp.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
    )


def test_coeffs_composition_revlex_matches_matlab_reference_and_grad():
    w = _w_fixture()
    h = (w[0].coeffs,)
    got = coeffs_composition(w, h, order=2, ordering="revlex")
    expected = np.array([[[5, 1], [6, 4], [7, 4]], [[8, 9], [9, 24], [10, 16]]])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6)

    def scalar(values):
        w_local = (
            MultiIndexPolynomial(coeffs=values[:4].reshape(2, 2), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
            MultiIndexPolynomial(coeffs=values[4:].reshape(2, 3), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
        )
        return jnp.sum(coeffs_composition(w_local, (w_local[0].coeffs,), order=2, ordering="revlex"))

    grad = jax.grad(scalar)(jnp.arange(10.0))
    assert grad.shape == (10,)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_coeffs_composition_lex_source_derived():
    w = _w_fixture()
    h = (w[0].coeffs,)
    got = coeffs_composition(w, h, order=2, ordering="lex")
    expected = np.array([[[5, 1], [6, 4], [7, 4]], [[8, 9], [9, 24], [10, 16]]])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6)


def test_coeffs_mixed_terms_revlex_matches_matlab_reference():
    w = _w_fixture()
    r = (
        MultiIndexPolynomial(coeffs=jnp.array([[0.5, 1.0], [1.5, 2.0]]), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
        MultiIndexPolynomial(coeffs=jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
    )
    got = coeffs_mixed_terms(2, 1, w, r, dim=2, output_dim=2, mix="aut", ordering="revlex")
    np.testing.assert_allclose(np.asarray(got), np.array([[12.0, 15.0, 18.0], [26.0, 33.0, 40.0]]), rtol=1e-6)


def test_coeffs_mixed_terms_explicit_indices_for_nonaut_style_terms():
    w = (
        MultiIndexPolynomial(coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]), ind=jnp.array([[1, 0], [0, 1]])),
        MultiIndexPolynomial(coeffs=jnp.zeros((2, 0)), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
    )
    r = (
        MultiIndexPolynomial(coeffs=jnp.array([[0.5, 1.0], [1.5, 2.0]]), ind=jnp.array([[1, 0], [0, 1]])),
        MultiIndexPolynomial(coeffs=jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]), ind=jnp.array([[2, 0], [1, 1], [0, 2]])),
        MultiIndexPolynomial(coeffs=jnp.zeros((2, 0)), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
    )
    got = coeffs_mixed_terms(1, 1, w, r, dim=2, output_dim=2, mix="R1", ordering="revlex", explicit_indices=True)
    assert got.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(got)))
