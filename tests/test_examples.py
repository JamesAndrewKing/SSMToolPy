import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jax_sparse

from ssmtoolpy import (
    benchmark_ssm_1st_order_model,
    evaluate_first_order_vector_field,
    lorenz_first_order_model,
    lorenz_vector_field,
    planar_system_model,
    planar_system_vector_field,
)


def test_planar_system_model_matches_matlab_build_model_source():
    model = planar_system_model()
    np.testing.assert_allclose(np.asarray(model.a_matrix), np.array([[-1.0, 0.0], [0.0, -np.sqrt(24.0)]]))
    np.testing.assert_allclose(np.asarray(model.b_matrix), np.eye(2))
    assert [term.shape for term in model.terms] == [(2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)]
    for degree, term in enumerate(model.terms, start=2):
        expected = np.zeros((2,) + (2,) * degree)
        expected[(1,) + (0,) * degree] = 1.0
        np.testing.assert_allclose(np.asarray(term), expected)


def test_planar_system_vector_field_matches_tensor_model_and_transforms():
    model = planar_system_model()
    state = jnp.array([0.5, -0.25])
    expected = np.array([-0.5, np.sqrt(24.0) * 0.25 + 0.5**2 + 0.5**3 + 0.5**4 + 0.5**5])
    np.testing.assert_allclose(np.asarray(planar_system_vector_field(state)), expected, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(model.system.odefun(0.0, state)), expected, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(evaluate_first_order_vector_field(model.a_matrix, model.b_matrix, state, model.terms)),
        expected,
        rtol=1e-6,
    )

    jit_value = jax.jit(planar_system_vector_field)(state)
    np.testing.assert_allclose(np.asarray(jit_value), expected, rtol=1e-6)
    jac = jax.jacfwd(planar_system_vector_field)(state)
    np.testing.assert_allclose(np.asarray(jac), np.array([[-1.0, 0.0], [2.5625, -np.sqrt(24.0)]]), rtol=1e-6)


def test_benchmark_ssm_first_order_model_matches_planar_source_copy():
    planar = planar_system_model()
    benchmark = benchmark_ssm_1st_order_model()
    np.testing.assert_allclose(np.asarray(benchmark.a_matrix), np.asarray(planar.a_matrix))
    np.testing.assert_allclose(np.asarray(benchmark.b_matrix), np.asarray(planar.b_matrix))
    for benchmark_term, planar_term in zip(benchmark.terms, planar.terms, strict=True):
        np.testing.assert_allclose(np.asarray(benchmark_term), np.asarray(planar_term))


def test_example_sparse_terms_use_bcoo_storage():
    planar = planar_system_model(sparse=True)
    assert all(isinstance(term, jax_sparse.BCOO) for term in planar.terms)
    assert [term.nse for term in planar.terms] == [1, 1, 1, 1]
    dense_planar = planar_system_model()
    for sparse_term, dense_term in zip(planar.terms, dense_planar.terms, strict=True):
        np.testing.assert_allclose(np.asarray(sparse_term.todense()), np.asarray(dense_term))

    lorenz = lorenz_first_order_model(sparse=True)
    assert len(lorenz.terms) == 1
    assert isinstance(lorenz.terms[0], jax_sparse.BCOO)
    assert lorenz.terms[0].nse == 2


def test_lorenz_model_matches_matlab_lorenz_formula():
    model = lorenz_first_order_model(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    np.testing.assert_allclose(
        np.asarray(model.a_matrix),
        np.array([[-10.0, 10.0, 0.0], [28.0, -1.0, 0.0], [0.0, 0.0, -8.0 / 3.0]]),
        rtol=1e-6,
    )
    expected_f2 = np.zeros((3, 3, 3))
    expected_f2[1, 0, 2] = -1.0
    expected_f2[2, 0, 1] = 1.0
    np.testing.assert_allclose(np.asarray(model.terms[0]), expected_f2)

    state = jnp.array([1.0, 2.0, 3.0])
    expected = np.array([10.0, 23.0, -6.0])
    np.testing.assert_allclose(np.asarray(lorenz_vector_field(state)), expected, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(model.system.odefun(0.0, state)), expected, rtol=1e-6)


def test_lorenz_vector_field_parameter_transforms():
    state = jnp.array([1.0, 2.0, 3.0])
    jit_value = jax.jit(lambda x: lorenz_vector_field(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0))(state)
    np.testing.assert_allclose(np.asarray(jit_value), np.array([10.0, 23.0, -6.0]), rtol=1e-6)

    jac_state = jax.jacfwd(lorenz_vector_field)(state)
    np.testing.assert_allclose(
        np.asarray(jac_state),
        np.array([[-10.0, 10.0, 0.0], [25.0, -1.0, -1.0], [2.0, 1.0, -8.0 / 3.0]]),
        rtol=1e-6,
    )

    def scalar_from_params(params):
        return lorenz_vector_field(state, sigma=params[0], rho=params[1], beta=params[2]).sum()

    grad_params = jax.grad(scalar_from_params)(jnp.array([10.0, 28.0, 8.0 / 3.0]))
    np.testing.assert_allclose(np.asarray(grad_params), np.array([1.0, 1.0, -3.0]), rtol=1e-6)
