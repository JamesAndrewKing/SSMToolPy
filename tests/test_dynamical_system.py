import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ssmtoolpy import (
    FourierForcingTerm,
    MultiIndexPolynomial,
    PeriodicForcing,
    evaluate_first_order_vector_field,
    evaluate_periodic_forcing,
    evaluate_polynomial_terms,
    first_order_polynomial_terms_from_second_order,
    first_order_from_second_order_nonlinearity,
    first_order_nonlinearity,
    first_order_tensor_terms_from_second_order,
    first_order_terms_from_second_order,
    mechanical_binv_a,
    mechanical_a_matrix,
    mechanical_b_matrix,
    polynomial_degree,
    polynomial_input_dim,
    second_order_internal_force,
    second_order_internal_force_jacobian_x,
    second_order_internal_force_jacobian_xd,
    second_order_residual,
)


def _first_order_terms():
    return (
        MultiIndexPolynomial(coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]), ind=jnp.array([[1, 0], [0, 2]])),
        MultiIndexPolynomial(coeffs=jnp.array([[5.0], [6.0]]), ind=jnp.array([[1, 1]])),
    )


def _second_order_terms():
    return (MultiIndexPolynomial(coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]), ind=jnp.array([[1, 0], [0, 2]])),)


def test_first_and_second_order_nonlinearity_match_matlab_reference():
    state = jnp.array([2.0, 3.0])
    np.testing.assert_allclose(np.asarray(first_order_nonlinearity(state, _first_order_terms())), np.array([50.0, 78.0]))
    np.testing.assert_allclose(np.asarray(jax.jit(lambda z: first_order_nonlinearity(z, _first_order_terms()))(state)), np.array([50.0, 78.0]))
    grad = jax.jacfwd(lambda z: first_order_nonlinearity(z, _first_order_terms()))(state)
    assert grad.shape == (2, 2)

    fnl = second_order_internal_force(state, None, _second_order_terms())
    np.testing.assert_allclose(np.asarray(fnl), np.array([20.0, 42.0]))
    first_order = first_order_from_second_order_nonlinearity(fnl)
    np.testing.assert_allclose(np.asarray(first_order), np.array([-20.0, -42.0, 0.0, 0.0]))


def test_second_order_internal_force_jacobians_match_matlab_reference():
    x = jnp.array([2.0, 3.0])
    jac = second_order_internal_force_jacobian_x(x, _second_order_terms())
    np.testing.assert_allclose(np.asarray(jac), np.array([[1.0, 12.0], [3.0, 24.0]]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(second_order_internal_force_jacobian_xd(2)), np.zeros((2, 2)))
    jac_grad = jax.jacfwd(lambda q: second_order_internal_force_jacobian_x(q, _second_order_terms()).reshape(-1))(x)
    assert jac_grad.shape == (4, 2)


def test_second_order_residual_matches_matlab_reference_and_transforms():
    m = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    c = jnp.array([[0.1, 0.0], [0.0, 0.2]])
    k = jnp.array([[5.0, 1.0], [1.0, 4.0]])
    q = jnp.array([1.0, 2.0])
    qd = jnp.array([0.5, -1.0])
    qdd = jnp.array([0.25, 0.5])
    fext = jnp.array([0.2, 0.3])
    result = second_order_residual(m, c, k, q, qd, qdd, _second_order_terms(), fext)
    np.testing.assert_allclose(np.asarray(result.residual), np.array([16.35, 29.0]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(result.c0), 34.396877, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(result.drdqdd), np.asarray(m))
    np.testing.assert_allclose(np.asarray(result.drdqd), np.asarray(c))
    grad = jax.jacfwd(lambda qq: second_order_residual(m, c, k, qq, qd, qdd, _second_order_terms(), fext).residual)(q)
    assert grad.shape == (2, 2)


def test_periodic_forcing_and_vector_field_are_transformable():
    forcing = PeriodicForcing(
        terms=(
            FourierForcingTerm(
                kappa=2,
                terms=(
                    MultiIndexPolynomial(coeffs=jnp.array([[1.0 + 1.0j], [2.0 + 0.0j]]), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
                    MultiIndexPolynomial(coeffs=jnp.array([[3.0 + 0.0j], [0.0 + 1.0j]]), ind=jnp.array([[1, 0]])),
                ),
            ),
        ),
        epsilon=jnp.array(0.1),
        omega=jnp.array(1.25),
    )
    coordinates = jnp.array([2.0, 3.0])
    fext = evaluate_periodic_forcing(0.5, coordinates, forcing)
    phase = np.exp(1j * 2 * 1.25 * 0.5)
    expected = 0.1 * np.real(np.array([[1 + 1j], [2 + 0j]]) * phase + np.array([[6 + 0j], [0 + 2j]]) * phase)
    np.testing.assert_allclose(np.asarray(fext), expected, rtol=1e-6)
    jac = jax.jacfwd(lambda z: evaluate_periodic_forcing(0.5, z, forcing).reshape(-1))(coordinates)
    assert jac.shape == (2, 2)

    a = jnp.array([[1.0, 0.2], [0.0, -0.5]])
    b = jnp.eye(2)
    value = evaluate_first_order_vector_field(a, b, coordinates, _first_order_terms(), None, 0.0)
    expected_rhs = a @ coordinates + first_order_nonlinearity(coordinates, _first_order_terms())
    np.testing.assert_allclose(np.asarray(value), np.asarray(expected_rhs), rtol=1e-6)
    jit_value = jax.jit(lambda z: evaluate_first_order_vector_field(a, b, z, _first_order_terms()))(coordinates)
    np.testing.assert_allclose(np.asarray(jit_value), np.asarray(expected_rhs), rtol=1e-6)


def test_mechanical_binv_a_matches_formula():
    m = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    c = jnp.array([[0.1, 0.0], [0.0, 0.2]])
    k = jnp.array([[5.0, 1.0], [1.0, 4.0]])
    expected = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-2.5, -0.5, -0.05, 0.0], [-0.25, -1.0, 0.0, -0.05]])
    np.testing.assert_allclose(np.asarray(mechanical_binv_a(m, c, k)), expected, rtol=1e-7)
    grad = jax.jacfwd(lambda kk: mechanical_binv_a(m, c, kk.reshape(2, 2)).sum())(k.ravel())
    assert grad.shape == (4,)


def test_mechanical_a_b_matrices_and_polynomial_metadata():
    m = jnp.array([[2.0, 0.0], [0.0, 4.0]])
    c = jnp.array([[0.1, 0.0], [0.0, 0.2]])
    k = jnp.array([[5.0, 1.0], [1.0, 4.0]])
    expected_a = np.array([[-5.0, -1.0, 0.0, 0.0], [-1.0, -4.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    expected_b = np.array([[0.1, 0.0, 2.0, 0.0], [0.0, 0.2, 0.0, 4.0], [2.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(mechanical_a_matrix(m, k)), expected_a)
    np.testing.assert_allclose(np.asarray(mechanical_b_matrix(m, c)), expected_b)

    terms = _second_order_terms()
    assert polynomial_input_dim(terms) == 2
    assert polynomial_degree(terms) == 1
    with pytest.raises(ValueError):
        polynomial_input_dim((MultiIndexPolynomial(coeffs=jnp.zeros((2, 0)), ind=jnp.zeros((0, 2), dtype=jnp.int32)),))


def test_first_order_terms_from_second_order_multiindex_sign_and_transform():
    terms = _second_order_terms()
    converted = first_order_polynomial_terms_from_second_order(terms, n=2, total_dim=4, system_order=2)
    assert converted[0].coeffs.shape == (4, 2)
    np.testing.assert_allclose(np.asarray(converted[0].coeffs), np.array([[-1.0, -2.0], [-3.0, -4.0], [0.0, 0.0], [0.0, 0.0]]))
    value = evaluate_polynomial_terms(converted, jnp.array([2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(value[:, 0]), np.array([-20.0, -42.0, 0.0, 0.0]))

    positive = first_order_terms_from_second_order(terms, n=2, total_dim=4, system_order=1)
    np.testing.assert_allclose(np.asarray(positive[0].coeffs[:2]), np.array([[1.0, 2.0], [3.0, 4.0]]))
    jac = jax.jacfwd(lambda coeff: first_order_polynomial_terms_from_second_order((MultiIndexPolynomial(coeff.reshape(2, 2), terms[0].ind),), n=2)[0].coeffs.sum())(
        terms[0].coeffs.ravel()
    )
    assert jac.shape == (4,)


def test_first_order_tensor_terms_from_second_order_dense_embedding():
    tensor = jnp.zeros((2, 2, 2))
    tensor = tensor.at[0, 0, 0].set(2.0)
    tensor = tensor.at[1, 1, 1].set(3.0)
    converted = first_order_tensor_terms_from_second_order((tensor,), n=2, total_dim=4, system_order=2)
    assert converted[0].shape == (4, 4, 4)
    np.testing.assert_allclose(np.asarray(converted[0][0, 0, 0]), -2.0)
    np.testing.assert_allclose(np.asarray(converted[0][1, 1, 1]), -3.0)
    np.testing.assert_allclose(np.asarray(converted[0][2:, :, :]), np.zeros((2, 4, 4)))
