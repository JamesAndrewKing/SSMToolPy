import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ssmtoolpy import (
    MultiIndexPolynomial,
    auto_ode_2md_ssm_cartesian,
    cal_rhos,
    check_auto_reduced_dynamics,
    check_spectrum_and_internal_resonance,
    create_po_amplitude_data,
    create_reduced_dynamics_data,
    detect_resonant_modes,
    initial_fixed_point_guess,
    monitor_state_names,
    reduced_data_to_2md,
    scale_parameters,
)


def test_cal_rhos_monitor_names_scaling_and_initial_guess():
    radii = cal_rhos(jnp.array([3.0, 4.0, 5.0, 12.0]), jnp.array([2.0, 0.5]))
    np.testing.assert_allclose(np.asarray(radii), np.array([10.0, 6.5]))
    grad = jax.grad(lambda values: jnp.sum(cal_rhos(values, jnp.ones(2))))(jnp.array([3.0, 4.0, 5.0, 12.0]))
    assert grad.shape == (4,)

    assert monitor_state_names(True, 2) == (("rho1", "rho2"), ("th1", "th2"))
    assert monitor_state_names(False, 2) == (("Rez1", "Rez2"), ("Imz1", "Imz2"))
    np.testing.assert_allclose(np.asarray(scale_parameters(jnp.array([2.0, 3.0]), jnp.array([4.0, 5.0]))), np.array([8.0, 15.0]))

    _, z0 = initial_fixed_point_guess(jnp.array([2.0, 0.1]), is_polar=True, m=2, provided=(jnp.array([2.0]), jnp.array([-1.0, -0.5, 0.2, 7.0])))
    assert z0[0] > 0
    assert 0 <= z0[1] < 2 * np.pi


def test_spectrum_and_auto_reduced_dynamics_validation():
    assert check_spectrum_and_internal_resonance(
        jnp.array([-0.1, -0.1, -0.2, -0.2]),
        jnp.array([1.0, -1.0, 2.0, -2.0]),
        jnp.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError):
        check_spectrum_and_internal_resonance(jnp.array([-0.1, -0.2]), jnp.array([1.0, -1.0]), jnp.array([3.0]))

    empty = MultiIndexPolynomial(jnp.zeros((4, 0)), jnp.zeros((0, 4), dtype=jnp.int32))
    poly = MultiIndexPolynomial(
        coeffs=jnp.array([[2.0], [0.0], [0.0], [0.0]]),
        ind=jnp.array([[2, 1, 0, 0]], dtype=jnp.int32),
    )
    beta, kappa = check_auto_reduced_dynamics((empty, poly), order=2, modal_frequencies=jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(beta[0]), np.array([2.0]))
    assert kappa[0].shape == (1, 4)


def test_reduced_dynamics_and_po_amplitude_data_builders():
    beta = (jnp.array([1.0 + 0.5j]),)
    kappa = (jnp.array([[2, 1]], dtype=jnp.int32),)
    data = create_reduced_dynamics_data(
        beta,
        kappa,
        lambda_real=jnp.array([-0.1, -0.1]),
        lambda_imag=jnp.array([1.0, -1.0]),
        modal_frequencies=jnp.array([1.0]),
        nonauto_indices=(0,),
        nonauto_coefficients=jnp.array([0.2 + 0.3j]),
        nonauto_kappas=jnp.array([1]),
        order=3,
        resonant_modes=jnp.array([1]),
    )
    assert data.order == 3
    ode_data = reduced_data_to_2md(data, is_base_force=True)
    assert ode_data.is_base_force is True
    np.testing.assert_allclose(np.asarray(ode_data.lambda_real), np.array([-0.1]))

    po_data = create_po_amplitude_data(
        cind=jnp.array([[2, 0], [1, 1], [0, 2]], dtype=jnp.int32),
        dind=jnp.array([[1, 0], [0, 1], [0, 1]], dtype=jnp.int32),
        modal_frequencies=jnp.array([1.0, 2.0]),
        wcoeffs=jnp.ones((2, 3)),
        optdof=jnp.array([0, 1]),
        is_nonauto=False,
    )
    np.testing.assert_allclose(np.asarray(po_data.hatr_values), np.array([1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(po_data.uidxpo), np.array([1, 2, 3, 4, 5]))


def test_auto_ode_2md_cartesian_and_detect_resonant_modes():
    data = create_reduced_dynamics_data(
        beta=(jnp.array([0.2 + 0.1j]),),
        kappa=(jnp.array([[2, 1]], dtype=jnp.int32),),
        lambda_real=jnp.array([-0.1, -0.1]),
        lambda_imag=jnp.array([1.0, -1.0]),
        modal_frequencies=jnp.array([1.0]),
        nonauto_indices=(),
        nonauto_coefficients=jnp.zeros((0,), dtype=jnp.complex64),
        nonauto_kappas=jnp.zeros((0,)),
        order=2,
        resonant_modes=jnp.array([0]),
    )
    z = jnp.array([0.7, -0.2])
    y = auto_ode_2md_ssm_cartesian(z, data)
    q = z[0] + 1j * z[1]
    expected = (-0.1 + 1j) * q + (0.2 + 0.1j) * q**2 * jnp.conj(q)
    np.testing.assert_allclose(np.asarray(y), np.asarray([expected.real, expected.imag]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(jax.jit(lambda state: auto_ode_2md_ssm_cartesian(state, data))(z)), np.asarray(y), rtol=1e-6)

    modes, freqs = detect_resonant_modes(1.0j, jnp.array([1.0j, 2.0j, 6.0j]), 1e-6)
    np.testing.assert_array_equal(np.asarray(modes), np.array([0, 1]))
    np.testing.assert_allclose(np.asarray(freqs), np.array([1.0, 2.0]))
