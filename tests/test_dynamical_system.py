import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ssmtoolpy import (
    DSOptions,
    DynamicalSystem,
    FourierForcingTerm,
    LinearSpectrum,
    MultiIndexPolynomial,
    PeriodicForcing,
    add_forcing,
    evaluate_first_order_vector_field,
    evaluate_periodic_forcing,
    evaluate_polynomial_terms,
    first_order_forcing_terms_from_second_order,
    first_order_polynomial_terms_from_second_order,
    first_order_from_second_order_nonlinearity,
    first_order_nonlinearity,
    first_order_tensor_terms_from_second_order,
    first_order_terms_from_second_order,
    forcing_kappas,
    infer_callable_input_dim,
    infer_semi_intrusive_input_dim,
    linear_spectral_analysis,
    mechanical_binv_a,
    mechanical_a_matrix,
    mechanical_b_matrix,
    polynomial_degree,
    polynomial_input_dim,
    normalize_modes,
    second_order_internal_force,
    second_order_internal_force_jacobian_x,
    second_order_internal_force_jacobian_xd,
    second_order_residual,
    sort_modes,
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


def test_forcing_kappas_and_second_order_forcing_conversion():
    forcing_terms = (
        FourierForcingTerm(
            kappa=jnp.array([1, 0]),
            terms=(
                MultiIndexPolynomial(coeffs=jnp.array([[1.0], [2.0]]), ind=jnp.zeros((1, 2), dtype=jnp.int32)),
                MultiIndexPolynomial(coeffs=jnp.array([[3.0, 4.0], [5.0, 6.0]]), ind=jnp.array([[1, 0], [0, 2]], dtype=jnp.int32)),
            ),
        ),
        FourierForcingTerm(
            kappa=jnp.array([-1, 1]),
            terms=(MultiIndexPolynomial(coeffs=jnp.array([[7.0], [8.0]]), ind=jnp.zeros((1, 2), dtype=jnp.int32)),),
        ),
    )
    periodic = PeriodicForcing(terms=forcing_terms, epsilon=jnp.array(0.25), omega=jnp.array([1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(forcing_kappas(periodic)), np.array([[1, 0], [-1, 1]]))

    converted = first_order_forcing_terms_from_second_order(forcing_terms, n=2, total_dim=4)
    np.testing.assert_allclose(np.asarray(converted[0].terms[0].coeffs), np.array([[1.0], [2.0], [0.0], [0.0]]))
    np.testing.assert_array_equal(np.asarray(converted[0].terms[1].ind), np.array([[1, 0, 0, 0], [0, 2, 0, 0]]))

    def scalar(values):
        local_terms = (
            FourierForcingTerm(
                kappa=1,
                terms=(MultiIndexPolynomial(coeffs=values.reshape(2, 2), ind=jnp.array([[1, 0], [0, 2]], dtype=jnp.int32)),),
            ),
        )
        return first_order_forcing_terms_from_second_order(local_terms, n=2)[0].terms[0].coeffs.sum()

    grad = jax.grad(scalar)(jnp.arange(4.0))
    np.testing.assert_allclose(np.asarray(grad), np.ones(4))


def test_callable_input_dimension_inference_helpers():
    def primary_only(vector):
        if vector.shape[0] != 2:
            raise ValueError("wrong dimension")
        return vector

    def total_only(vector):
        if vector.shape[0] != 4:
            raise ValueError("wrong dimension")
        return vector

    assert infer_callable_input_dim(primary_only, 2, 4) == 2
    assert infer_callable_input_dim(total_only, 2, 4) == 4

    def semi_total(inputs):
        if len(inputs) != 2 or inputs[0].shape[0] != 4:
            raise ValueError("wrong dimension")
        return inputs[0] + inputs[1]

    assert infer_semi_intrusive_input_dim((None, semi_total), 2, 4) == 4

    with pytest.raises(ValueError):
        infer_callable_input_dim(lambda vector: (_ for _ in ()).throw(ValueError("nope")), 2, 4)


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
    value = evaluate_polynomial_terms(converted, jnp.array([2.0, 3.0, 0.0, 0.0]))
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


def test_add_forcing_and_dynamical_system_wrapper_methods():
    m = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    c = jnp.array([[0.1, 0.0], [0.0, 0.2]])
    k = jnp.array([[5.0, 1.0], [1.0, 4.0]])
    forcing = add_forcing(1, jnp.array([[1.0, 2.0], [3.0, 4.0]]), kappas=jnp.array([1, -1]), epsilon=0.2, total_dim=4)
    assert isinstance(forcing, PeriodicForcing)
    assert len(forcing.terms) == 2
    np.testing.assert_allclose(np.asarray(forcing.terms[0].terms[0].coeffs[:, 0]), np.array([1.0, 3.0, 0.0, 0.0]))

    system = DynamicalSystem(order=2, m_matrix=m, c_matrix=c, k_matrix=k, fnl_terms=_second_order_terms()).with_forcing(
        jnp.array([[1.0], [0.5]]),
        kappas=jnp.array([1]),
        epsilon=0.0,
    )
    np.testing.assert_allclose(np.asarray(system.a_matrix), np.asarray(mechanical_a_matrix(m, k)))
    np.testing.assert_allclose(np.asarray(system.b_matrix), np.asarray(mechanical_b_matrix(m, c)))
    assert system.n == 2
    assert system.N == 4
    assert system.degree == 1
    np.testing.assert_array_equal(np.asarray(system.kappas), np.array([[1]]))

    z = jnp.array([1.0, 2.0, 0.5, -1.0])
    ode_value = system.odefun(0.0, z)
    converted = first_order_terms_from_second_order(_second_order_terms(), n=2, total_dim=4, system_order=2)
    expected = evaluate_first_order_vector_field(system.a_matrix, system.b_matrix, z, converted, system.forcing, 0.0)
    np.testing.assert_allclose(np.asarray(ode_value), np.asarray(expected), rtol=1e-6)
    jac = jax.jacfwd(lambda state: system.odefun(0.0, state))(z)
    assert jac.shape == (4, 4)

    residual = system.residual(jnp.array([1.0, 2.0]), jnp.array([0.5, -1.0]), jnp.array([0.25, 0.5]))
    assert residual.residual.shape == (2,)


def test_linear_spectral_analysis_sorting_and_normalization():
    a = jnp.array([[-0.1, -2.0], [2.0, -0.1]])
    spectrum = linear_spectral_analysis(a, jnp.eye(2), DSOptions(remove_zeros=False))
    assert isinstance(spectrum, LinearSpectrum)
    assert spectrum.eigenvalues.shape == (2,)
    assert np.imag(np.asarray(spectrum.eigenvalues[0])) > 0
    gram = jnp.conj(spectrum.left_eigenvectors).T @ spectrum.right_eigenvectors
    np.testing.assert_allclose(np.asarray(gram), np.eye(2), rtol=1e-5, atol=1e-5)

    unsorted = sort_modes(jnp.eye(2), jnp.array([-2.0 + 0.0j, -1.0 + 0.0j]), jnp.eye(2))
    np.testing.assert_allclose(np.asarray(unsorted.eigenvalues), np.array([-1.0 + 0.0j, -2.0 + 0.0j]))
    right, left = normalize_modes(jnp.eye(2), 2.0 * jnp.eye(2), jnp.eye(2))
    np.testing.assert_allclose(np.asarray(jnp.conj(left).T @ right), np.eye(2), rtol=1e-6)
