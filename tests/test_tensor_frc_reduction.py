import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy import MultiIndexPolynomial, expand_tensor, expand_tensor_derivative, frc_ab, khatri_rao_product, reduced_to_full


def test_khatri_rao_product_and_jacfwd():
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    expected = jnp.array([[5.0, 12.0], [7.0, 16.0], [9.0, 20.0], [15.0, 24.0], [21.0, 32.0], [27.0, 40.0]])
    np.testing.assert_allclose(np.asarray(khatri_rao_product(a, b)), np.asarray(expected))
    jac = jax.jacfwd(lambda x: khatri_rao_product(x.reshape(2, 2), b).sum())(a.ravel())
    assert jac.shape == (4,)


def test_expand_tensor_matches_direct_quadratic_and_transforms():
    tensor = jnp.zeros((2, 2, 2))
    tensor = tensor.at[0, 0, 0].set(2.0)
    tensor = tensor.at[0, 0, 1].set(3.0)
    tensor = tensor.at[1, 1, 1].set(-1.0)
    point = jnp.array([2.0, 5.0])
    expected = jnp.array([2.0 * 4.0 + 3.0 * 10.0, -25.0])
    np.testing.assert_allclose(np.asarray(expand_tensor(tensor, point)), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(jax.jit(expand_tensor)(tensor, point)), np.asarray(expected))
    grad = jax.grad(lambda x: expand_tensor(tensor, x).sum())(point)
    np.testing.assert_allclose(np.asarray(grad), np.array([8.0 + 15.0, 6.0 - 10.0]), rtol=1e-6)


def test_expand_tensor_derivative_matches_jacfwd():
    tensor = jnp.arange(8.0).reshape(2, 2, 2)
    point = jnp.array([0.25, -0.5])
    expected = jax.jacfwd(lambda x: expand_tensor(tensor, x))(point)
    np.testing.assert_allclose(np.asarray(expand_tensor_derivative(tensor, point)), np.asarray(expected), rtol=1e-6)


def test_frc_ab_values_grad_and_vmap():
    gamma = jnp.array([1.0 + 2.0j, -0.5 + 0.25j])
    lam = -0.1 + 1.5j
    a, b = frc_ab(jnp.array(2.0), jnp.array(1.25), gamma, lam)
    np.testing.assert_allclose(np.asarray(a), -0.2 + 1.0 * 8.0 - 0.5 * 32.0)
    np.testing.assert_allclose(np.asarray(b), 2.0 * 0.25 + 2.0 * 8.0 + 0.25 * 32.0)
    grad = jax.grad(lambda rho: sum(frc_ab(rho, 1.25, gamma, lam)))(2.0)
    assert np.isfinite(np.asarray(grad))
    batched_a, batched_b = jax.vmap(lambda rho: frc_ab(rho, 1.25, gamma, lam))(jnp.array([1.0, 2.0]))
    assert batched_a.shape == (2,)
    assert batched_b.shape == (2,)


def test_reduced_to_full_autonomous_jit():
    poly = MultiIndexPolynomial(
        coeffs=jnp.array([[1.0, 2.0], [-1.0, 0.5]]),
        ind=jnp.array([[1, 0], [0, 2]]),
    )
    points = jnp.array([[3.0, 4.0], [2.0, -1.0]])
    expected = jnp.array([[11.0, 6.0], [-1.0, -3.5]])
    np.testing.assert_allclose(np.asarray(reduced_to_full(points, (poly,))), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(jax.jit(lambda pts: reduced_to_full(pts, (poly,)))(points)), np.asarray(expected))
