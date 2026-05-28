import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy import (
    MultiIndexPolynomial,
    check_stability,
    compute_fixed_points_2d,
    compute_gamma,
    frc_jacobian,
    frc_psi,
    get_contour_xy,
)


def test_compute_gamma_matches_matlab_reference():
    empty = MultiIndexPolynomial(coeffs=jnp.zeros((1, 0)), ind=jnp.zeros((0, 2), dtype=jnp.int32))
    r0 = (
        empty,
        empty,
        MultiIndexPolynomial(coeffs=jnp.array([[9.0, 8.0, 7.0]]), ind=jnp.array([[2, 1], [0, 3], [1, 2]])),
        empty,
        MultiIndexPolynomial(coeffs=jnp.array([[1.0, 2.0]]), ind=jnp.array([[3, 2], [1, 4]])),
    )
    np.testing.assert_allclose(np.asarray(compute_gamma(r0)), np.array([9.0, 1.0]))


def test_frc_psi_jacobian_stability_match_matlab_reference_and_transform():
    gamma = jnp.array([1.0 + 2.0j, -0.5 + 0.25j])
    lam = -0.1 + 1.5j
    rho = jnp.array(2.0)
    omega = jnp.array(1.25)
    forcing = 3.0 + 4.0j
    psi = frc_psi(rho, omega, gamma, lam, forcing)
    np.testing.assert_allclose(np.asarray(psi), 2.3322810541098757, rtol=1e-7)

    jac = frc_jacobian(rho, psi, gamma, lam, 0.2, forcing)
    expected = np.array([[-28.1, -0.9862849063896964], [16.246571226597425, -0.08252587992240335]])
    np.testing.assert_allclose(np.asarray(jac), expected, rtol=1e-6)

    stability = check_stability(jnp.array([rho, 1.0]), jnp.array([psi, 0.3]), gamma, lam, 0.2, forcing)
    np.testing.assert_array_equal(np.asarray(stability), np.array([True, False]))

    grad = jax.grad(lambda r: frc_psi(r, omega, gamma, lam, forcing))(rho)
    assert np.isfinite(np.asarray(grad))
    jac_grad = jax.jacfwd(lambda r: frc_jacobian(r, psi, gamma, lam, 0.2, forcing).reshape(-1))(rho)
    assert jac_grad.shape == (4,)
    batched = jax.vmap(lambda r: frc_psi(r, omega, gamma, lam, forcing))(jnp.array([1.5, 2.0]))
    assert batched.shape == (2,)


def test_get_contour_xy_matches_matlab_reference():
    contour = jnp.array([[0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 5.0], [2.0, 10.0, 11.0, 3.0, 30.0, 31.0, 32.0]])
    x, y, n = get_contour_xy(contour)
    np.testing.assert_allclose(np.asarray(x), np.array([np.nan, 1, 2, np.nan, 3, 4, 5]), equal_nan=True)
    np.testing.assert_allclose(np.asarray(y), np.array([np.nan, 10, 11, np.nan, 30, 31, 32]), equal_nan=True)
    assert n == 2


def test_compute_fixed_points_2d_finds_grid_zero_intersection():
    x = jnp.linspace(-1.0, 1.0, 5)
    y = jnp.linspace(-1.0, 1.0, 5)
    xx, yy = jnp.meshgrid(x, y)
    x_dot = xx - 0.25
    y_dot = yy + 0.5
    x0, y0 = compute_fixed_points_2d(x, y, x_dot, y_dot)
    assert x0.shape == y0.shape == (1,)
    np.testing.assert_allclose(np.asarray(x0), np.array([0.25]), atol=1e-7)
    np.testing.assert_allclose(np.asarray(y0), np.array([-0.5]), atol=1e-7)
