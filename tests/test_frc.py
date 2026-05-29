import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy import (
    MultiIndexPolynomial,
    ReducedDynamics2mDData,
    ReducedDynamicsHarmonic,
    cal_ab_dab,
    check_stability,
    compute_fixed_points_2d,
    compute_gamma,
    compute_reduced_dynamics_2d_polar,
    frc_ab,
    frc_jacobian,
    frc_psi,
    get_contour_xy,
    ode_2d_ssm_cartesian,
    ode_2d_ssm_cartesian_fixrom,
    ode_2d_ssm_cartesian_fixrom_jac_params,
    ode_2d_ssm_cartesian_fixrom_jac_x,
    ode_2d_ssm_cartesian_jac_params,
    ode_2d_ssm_cartesian_jac_x,
    ode_2md_ssm_cartesian,
    ode_2md_ssm_cartesian_jac_params,
    ode_2md_ssm_cartesian_jac_x,
    ode_2md_ssm_polar,
    ode_2md_ssm_polar_jac_params,
    ode_2md_ssm_polar_jac_x,
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


def test_cal_ab_dab_matches_source_derived_reference_and_jacfwd():
    rho = jnp.array(1.2)
    gamma = jnp.array([0.3 + 0.2j, -0.1 + 0.05j])
    lam = -0.4 + 2.0j
    a, b, da, db = cal_ab_dab(rho, gamma, lam)
    expected_a = rho * lam.real + gamma[0].real * rho**3 + gamma[1].real * rho**5
    expected_b = lam.imag + gamma[0].imag * rho**2 + gamma[1].imag * rho**4
    expected_da = lam.real + 3 * gamma[0].real * rho**2 + 5 * gamma[1].real * rho**4
    expected_db = 2 * gamma[0].imag * rho + 4 * gamma[1].imag * rho**3
    np.testing.assert_allclose(np.asarray(a), np.asarray(expected_a), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(b), np.asarray(expected_b), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(da), np.asarray(expected_da), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(db), np.asarray(expected_db), rtol=1e-6, atol=1e-7)

    jac = jax.jacfwd(lambda r: jnp.stack(cal_ab_dab(r, gamma, lam)))(rho)
    assert jac.shape == (4,)


def test_compute_reduced_dynamics_2d_polar_includes_higher_harmonic_terms_and_grad():
    harmonic = ReducedDynamicsHarmonic(
        kappa=2,
        terms=(
            MultiIndexPolynomial(coeffs=jnp.array([[1.0 + 0.5j]]), ind=jnp.zeros((1, 2), dtype=jnp.int32)),
            MultiIndexPolynomial(coeffs=jnp.array([[0.25 - 0.1j]]), ind=jnp.array([[2, 1]], dtype=jnp.int32)),
        ),
    )
    rho = jnp.array(1.1)
    psi = jnp.array(0.4)
    gamma = jnp.array([0.2 + 0.3j])
    rhodot, rhopsidot, kappa = compute_reduced_dynamics_2d_polar(
        rho,
        psi,
        -0.1 + 1.5j,
        gamma,
        (harmonic,),
        omega=jnp.array(0.7),
        epsilon=jnp.array(0.2),
    )
    a, b = jnp.asarray(frc_ab(rho, 2 * 0.7, gamma, -0.1 + 1.5j))
    lead = 0.2 * (1.0 + 0.5j)
    high = 0.2 * rho**3 * (0.25 - 0.1j)
    expected_rho = a + jnp.cos(psi) * lead.real + jnp.sin(psi) * lead.imag
    expected_rho = expected_rho + high.real * jnp.cos(psi) + high.imag * jnp.sin(psi)
    expected_rhopsi = b + jnp.cos(psi) * lead.imag - jnp.sin(psi) * lead.real
    expected_rhopsi = expected_rhopsi + high.imag * jnp.cos(psi) - high.real * jnp.sin(psi)
    np.testing.assert_allclose(np.asarray(rhodot), np.asarray(expected_rho), rtol=1e-7)
    np.testing.assert_allclose(np.asarray(rhopsidot), np.asarray(expected_rhopsi), rtol=1e-7)
    assert int(kappa) == 2

    grad = jax.grad(
        lambda e: compute_reduced_dynamics_2d_polar(rho, psi, -0.1 + 1.5j, gamma, (harmonic,), 0.7, e)[0]
    )(jnp.array(0.2))
    assert np.isfinite(np.asarray(grad))


def test_ode_2d_ssm_cartesian_matches_source_derived_reference_and_jit_jacobian():
    empty = MultiIndexPolynomial(coeffs=jnp.zeros((1, 0)), ind=jnp.zeros((0, 2), dtype=jnp.int32))
    autonomous = (
        MultiIndexPolynomial(coeffs=jnp.array([[-0.1 + 1.2j]]), ind=jnp.array([[1, 0]], dtype=jnp.int32)),
        empty,
        MultiIndexPolynomial(coeffs=jnp.array([[0.3 + 0.2j]]), ind=jnp.array([[2, 1]], dtype=jnp.int32)),
    )
    harmonic = ReducedDynamicsHarmonic(
        kappa=1,
        terms=(
            MultiIndexPolynomial(coeffs=jnp.array([[0.4 - 0.2j]]), ind=jnp.zeros((1, 2), dtype=jnp.int32)),
            MultiIndexPolynomial(coeffs=jnp.array([[0.1 + 0.05j]]), ind=jnp.array([[1, 1]], dtype=jnp.int32)),
        ),
    )
    x = jnp.array([0.7, -0.3])
    params = jnp.array([1.5, 0.2])
    y = ode_2d_ssm_cartesian(0.25, x, params, autonomous, (harmonic,))
    q1 = x[0] + 1j * x[1]
    q2 = x[0] - 1j * x[1]
    exp_k = jnp.exp(1j * params[0] * 0.25)
    expected = (-0.1 + 1.2j) * q1 + (0.3 + 0.2j) * q1**2 * q2
    expected = expected + params[1] * ((0.4 - 0.2j) + (0.1 + 0.05j) * q1 * q2) * exp_k
    np.testing.assert_allclose(np.asarray(y), np.asarray([expected.real, expected.imag]), rtol=1e-7)

    jitted = jax.jit(lambda state: ode_2d_ssm_cartesian(0.25, state, params, autonomous, (harmonic,)))(x)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(y), rtol=1e-7)
    jac = ode_2d_ssm_cartesian_jac_x(0.25, x, params, autonomous, (harmonic,))
    expected_jac = jax.jacfwd(lambda state: ode_2d_ssm_cartesian(0.25, state, params, autonomous, (harmonic,)))(x)
    np.testing.assert_allclose(np.asarray(jac), np.asarray(expected_jac), rtol=1e-7)

    jac_params = ode_2d_ssm_cartesian_jac_params(0.25, x, params, autonomous, (harmonic,))
    expected_jac_params = jax.jacfwd(lambda pars: ode_2d_ssm_cartesian(0.25, x, pars, autonomous, (harmonic,)))(params)
    np.testing.assert_allclose(np.asarray(jac_params), np.asarray(expected_jac_params), rtol=1e-7)

    lead_only = ode_2d_ssm_cartesian_jac_params(
        0.25,
        x,
        params,
        autonomous,
        (harmonic,),
        include_higher_nonautonomous=False,
    )
    lead_harmonic = ReducedDynamicsHarmonic(kappa=1, terms=harmonic.terms[:1])
    expected_lead_only = jax.jacfwd(lambda pars: ode_2d_ssm_cartesian(0.25, x, pars, autonomous, (lead_harmonic,)))(params)
    np.testing.assert_allclose(np.asarray(lead_only), np.asarray(expected_lead_only), rtol=1e-7)

    np.testing.assert_allclose(
        np.asarray(ode_2d_ssm_cartesian_fixrom(0.25, x, params, autonomous, (harmonic,))),
        np.asarray(y),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(ode_2d_ssm_cartesian_fixrom_jac_x(0.25, x, params, autonomous, (harmonic,))),
        np.asarray(jac),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(ode_2d_ssm_cartesian_fixrom_jac_params(0.25, x, params, autonomous, (harmonic,))),
        np.asarray(jac_params),
        rtol=1e-7,
    )


def test_ode_2md_ssm_cartesian_vectorized_and_transformable():
    data = ReducedDynamics2mDData(
        beta=(
            jnp.array([0.2 + 0.1j]),
            jnp.array([-0.05 + 0.03j]),
        ),
        kappa=(
            jnp.array([[2, 1, 0, 0]], dtype=jnp.int32),
            jnp.array([[0, 0, 1, 2]], dtype=jnp.int32),
        ),
        lambda_real=jnp.array([-0.1, -0.2]),
        lambda_imag=jnp.array([1.0, 1.5]),
        modal_frequencies=jnp.array([1.0, 2.0]),
        nonauto_indices=(0, 1),
        nonauto_coefficients=jnp.array([0.3 - 0.2j, -0.1 + 0.4j]),
        is_base_force=True,
    )
    z = jnp.array([[0.7, 0.8], [-0.2, -0.1], [0.4, 0.5], [0.3, 0.2]])
    params = jnp.array([[1.2, 1.3], [0.05, 0.06]])
    y = ode_2md_ssm_cartesian(z, params, data)
    assert y.shape == z.shape

    first = ode_2md_ssm_cartesian(z[:, 0], params[:, 0], data)
    q = jnp.array([z[0, 0] + 1j * z[1, 0], z[2, 0] + 1j * z[3, 0]])
    q_conj = jnp.conj(q)
    expected0 = (-0.1 + 1j * (1.0 - 1.2)) * q[0] + (0.2 + 0.1j) * q[0] ** 2 * q_conj[0]
    expected0 = expected0 + 0.05 * (0.3 - 0.2j) * 1.2**2
    expected1 = (-0.2 + 1j * (1.5 - 2.0 * 1.2)) * q[1] + (-0.05 + 0.03j) * q[1] * q_conj[1] ** 2
    expected1 = expected1 + 0.05 * (-0.1 + 0.4j) * 1.2**2
    np.testing.assert_allclose(
        np.asarray(first),
        np.asarray([expected0.real, expected0.imag, expected1.real, expected1.imag]),
        rtol=1e-6,
        atol=1e-7,
    )

    jitted = jax.jit(lambda state: ode_2md_ssm_cartesian(state, params[:, 0], data))(z[:, 0])
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(first), rtol=1e-6, atol=1e-7)
    jac_x = ode_2md_ssm_cartesian_jac_x(z, params, data)
    assert jac_x.shape == (4, 4, 2)
    np.testing.assert_allclose(
        np.asarray(jac_x[:, :, 0]),
        np.asarray(jax.jacfwd(lambda state: ode_2md_ssm_cartesian(state, params[:, 0], data))(z[:, 0])),
        rtol=1e-6,
        atol=1e-7,
    )
    jac_p = ode_2md_ssm_cartesian_jac_params(z, params, data)
    assert jac_p.shape == (4, 2, 2)
    np.testing.assert_allclose(
        np.asarray(jac_p[:, :, 0]),
        np.asarray(jax.jacfwd(lambda pars: ode_2md_ssm_cartesian(z[:, 0], pars, data))(params[:, 0])),
        rtol=1e-6,
        atol=1e-7,
    )


def test_ode_2md_ssm_polar_vectorized_and_transformable():
    data = ReducedDynamics2mDData(
        beta=(jnp.array([0.2 + 0.1j]), jnp.array([-0.05 + 0.03j])),
        kappa=(
            jnp.array([[2, 1, 0, 0]], dtype=jnp.int32),
            jnp.array([[0, 0, 1, 2]], dtype=jnp.int32),
        ),
        lambda_real=jnp.array([-0.1, -0.2]),
        lambda_imag=jnp.array([1.0, 1.5]),
        modal_frequencies=jnp.array([1.0, 2.0]),
        nonauto_indices=(0, 1),
        nonauto_coefficients=jnp.array([0.3 - 0.2j, -0.1 + 0.4j]),
        is_base_force=False,
    )
    z = jnp.array([[1.2, 1.1], [0.2, 0.3], [0.9, 1.0], [-0.4, -0.2]])
    params = jnp.array([[1.2, 1.3], [0.05, 0.06]])
    y = ode_2md_ssm_polar(z, params, data)
    assert y.shape == z.shape

    first = ode_2md_ssm_polar(z[:, 0], params[:, 0], data)
    rho = jnp.array([z[0, 0], z[2, 0]])
    theta = jnp.array([z[1, 0], z[3, 0]])
    angle0 = (2 - 1 - 1) * theta[0]
    term0_rho = rho[0] ** 3 * (0.2 * jnp.cos(angle0) - 0.1 * jnp.sin(angle0))
    term0_theta = rho[0] ** 2 * (0.2 * jnp.sin(angle0) + 0.1 * jnp.cos(angle0))
    force0_rho = 0.05 * (0.3 * jnp.cos(theta[0]) - 0.2 * jnp.sin(theta[0]))
    force0_theta = 0.05 * (-0.3 * jnp.sin(theta[0]) / rho[0] - 0.2 * jnp.cos(theta[0]) / rho[0])
    expected0_rho = -0.1 * rho[0] + term0_rho + force0_rho
    expected0_theta = 1.0 - 1.2 + term0_theta + force0_theta
    assert np.isfinite(np.asarray(first)).all()
    np.testing.assert_allclose(np.asarray(first[0]), np.asarray(expected0_rho), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(first[1]), np.asarray(expected0_theta), rtol=1e-6)

    jitted = jax.jit(lambda state: ode_2md_ssm_polar(state, params[:, 0], data))(z[:, 0])
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(first), rtol=1e-6, atol=1e-7)
    jac_x = ode_2md_ssm_polar_jac_x(z, params, data)
    assert jac_x.shape == (4, 4, 2)
    np.testing.assert_allclose(
        np.asarray(jac_x[:, :, 0]),
        np.asarray(jax.jacfwd(lambda state: ode_2md_ssm_polar(state, params[:, 0], data))(z[:, 0])),
        rtol=1e-6,
        atol=1e-7,
    )
    jac_p = ode_2md_ssm_polar_jac_params(z, params, data)
    assert jac_p.shape == (4, 2, 2)
    np.testing.assert_allclose(
        np.asarray(jac_p[:, :, 0]),
        np.asarray(jax.jacfwd(lambda pars: ode_2md_ssm_polar(z[:, 0], pars, data))(params[:, 0])),
        rtol=1e-6,
        atol=1e-7,
    )
