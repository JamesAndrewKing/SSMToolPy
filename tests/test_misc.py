import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy import (
    AutoReducedDynamicsData,
    MultiIndexPolynomial,
    NonAutonomousTerm,
    ProjectionData,
    ReducedDynamicsSymbolicOptions,
    assemble_auto_reduced_dynamics,
    auto_red_dyn,
    extract_output,
    first_order_linear_response,
    nonlinear_projection_objective,
    project_to_ssm_linear,
    real_to_conjugate_state,
    reduced_dynamics_symbolic,
    reduced_to_full_traj,
    second_order_linear_response,
    solve_invariance_equation,
    spblkdiag,
    squared_distance_to_point_ssm,
    transient_traj_on_auto_ssm,
)


def _w0_fixture():
    return (
        MultiIndexPolynomial(coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]), ind=jnp.array([[1, 0], [0, 1]])),
        MultiIndexPolynomial(coeffs=jnp.array([[5.0, 6.0], [7.0, 8.0]]), ind=jnp.array([[2, 0], [1, 1]])),
    )


def test_reduced_to_full_traj_autonomous_matches_matlab_and_transforms():
    point = jnp.array([2.0, 3.0])
    expected = jnp.array([64.0, 94.0])
    np.testing.assert_allclose(np.asarray(reduced_to_full_traj(0.5, point, _w0_fixture())), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(jax.jit(lambda p: reduced_to_full_traj(0.5, p, _w0_fixture()))(point)), np.asarray(expected))
    grad = jax.jacfwd(lambda p: reduced_to_full_traj(0.5, p, _w0_fixture()))(point)
    assert grad.shape == (2, 2)


def test_reduced_to_full_traj_nonautonomous_matches_matlab_reference():
    w0 = (_w0_fixture()[0],)
    w1 = (
        NonAutonomousTerm(
            kappa=2,
            terms=(
                MultiIndexPolynomial(coeffs=jnp.array([[1.0 + 1.0j], [2.0 + 0.0j]]), ind=jnp.zeros((0, 2), dtype=jnp.int32)),
                MultiIndexPolynomial(coeffs=jnp.array([[3.0 + 0.0j, 0.0], [0.0, 4.0 + 1.0j]]), ind=jnp.array([[1, 0], [0, 1]])),
            ),
        ),
    )
    got = reduced_to_full_traj(0.5, jnp.array([2.0, 3.0]), w0, w1, 0.1, 1.25)
    np.testing.assert_allclose(np.asarray(got), np.array([8.125827191741129, 18.1567559215467]), rtol=1e-6)


def test_extract_output_matches_matlab_reference_and_grad():
    z = jnp.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    out = extract_output(z, jnp.array([0, 1]))
    np.testing.assert_allclose(np.asarray(out.z_out), np.asarray(z))
    np.testing.assert_allclose(np.asarray(out.a_out), np.array([3.0, 6.0]))
    np.testing.assert_allclose(np.asarray(out.z_norm), 6.7453687816160217, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(out.z_out_norm), 6.7453687816160217, rtol=1e-7)
    grad = jax.grad(lambda values: extract_output(values.reshape(2, 3), jnp.array([0, 1])).z_norm)(z.ravel())
    assert grad.shape == (6,)


def test_spblkdiag_solve_and_autoredyn_match_matlab_reference():
    blocks = jnp.stack((jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[5.0, 6.0], [7.0, 8.0]])), axis=2)
    expected = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 5, 6], [0, 0, 7, 8]])
    np.testing.assert_allclose(np.asarray(spblkdiag(blocks)), expected)
    jac = jax.jacfwd(lambda x: spblkdiag(x.reshape(2, 2, 2)).sum())(blocks.ravel())
    np.testing.assert_allclose(np.asarray(jac), np.ones(8))

    sol = solve_invariance_equation(jnp.array([[3.0, 1.0], [1.0, 2.0]]), jnp.array([9.0, 8.0]), "backslash")
    np.testing.assert_allclose(np.asarray(sol), np.array([2.0, 3.0]), rtol=1e-7)

    matrix = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = jnp.array([9.0, 8.0])
    np.testing.assert_allclose(np.asarray(solve_invariance_equation(matrix, rhs, "gmres", tol=1e-8)), np.array([2.0, 3.0]), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(solve_invariance_equation(lambda vector: matrix @ vector, rhs, "bicgstab", tol=1e-8)),
        np.array([2.0, 3.0]),
        rtol=1e-6,
    )
    multi_rhs = jnp.array([[9.0, 4.0], [8.0, 5.0]])
    expected_multi = np.linalg.solve(np.array(matrix), np.array(multi_rhs))
    np.testing.assert_allclose(np.asarray(solve_invariance_equation(matrix, multi_rhs, "gmres", tol=1e-8)), expected_multi, rtol=1e-6)

    data = AutoReducedDynamicsData(
        lamd=jnp.array([1.0 + 1.0j, -0.5 + 0.2j]),
        beta=jnp.array([[2.0 + 0.0j, 0.0], [0.0, 3.0 - 1.0j]]),
        kappa=jnp.array([[2, 0], [1, 1]]),
    )
    state = jnp.array([[2.0, 3.0], [4.0, 5.0]])
    got = auto_red_dyn(state, data)
    np.testing.assert_allclose(np.asarray(jnp.real(got)), np.array([[10.0, 21.0], [22.0, 42.5]]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.imag(got)), np.array([[2.0, 3.0], [-7.2, -14.0]]), rtol=1e-6)
    grad = jax.grad(lambda s: jnp.real(auto_red_dyn(s.reshape(2, 2), data)).sum())(state.ravel())
    assert grad.shape == (4,)


def test_projection_helpers_and_distance_objective():
    data = ProjectionData(realx=jnp.array([2]), compx=jnp.array([0, 1]), dim=3)
    state = real_to_conjugate_state(jnp.array([1.0, 2.0, 5.0]), data)
    np.testing.assert_allclose(np.asarray(state), np.array([1.0 + 2.0j, 1.0 - 2.0j, 5.0 + 0.0j]))

    w0 = (
        MultiIndexPolynomial(
            coeffs=jnp.array([[1.0 + 0.0j, 2.0 + 0.0j, 0.0], [0.0, 1.0 + 0.0j, 1.0 + 0.0j]]),
            ind=jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
    )
    z0 = jnp.array([10.0 + 0.0j, 20.0 + 0.0j])
    value = squared_distance_to_point_ssm(z0, jnp.array([1.0, 2.0, 5.0]), w0, data)
    np.testing.assert_allclose(np.asarray(value), (3 - 10) ** 2 + (6 - 20) ** 2)
    objective = nonlinear_projection_objective(z0, w0, data)
    grad = jax.grad(lambda u: jnp.real(objective(u)))(jnp.array([1.0, 2.0, 5.0]))
    assert grad.shape == (3,)

    q = project_to_ssm_linear(jnp.array([3.0, 4.0, 5.0]), jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), jnp.eye(3))
    np.testing.assert_allclose(np.asarray(q), np.array([8.0, 9.0]))


def test_first_order_linear_response_matches_frequency_domain_solve_and_transforms():
    a_matrix = jnp.array([[-2.0, 0.0], [0.0, -3.0]])
    b_matrix = jnp.eye(2)
    kappas = jnp.array([1.0, -1.0])
    coeffs = jnp.array([[1.0 + 0.5j, 1.0 - 0.5j], [0.25 - 0.5j, 0.25 + 0.5j]])
    omegas = jnp.array([0.5, 1.25])

    result = first_order_linear_response(a_matrix, b_matrix, kappas, coeffs, omegas, epsilon=0.2, outdof=jnp.array([0]), nt=16)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, 16)
    modal0 = jnp.stack(
        [
            jnp.linalg.solve(a_matrix - 1j * kappas[0] * omegas[0] * b_matrix, coeffs[:, 0]),
            jnp.linalg.solve(a_matrix - 1j * kappas[1] * omegas[0] * b_matrix, coeffs[:, 1]),
        ],
        axis=1,
    )
    expected0 = 0.2 * jnp.real(modal0 @ jnp.exp(1j * kappas[:, None] * phi[None, :]))
    np.testing.assert_allclose(np.asarray(result.response[0]), np.asarray(expected0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(result.a_out[0]), np.max(np.abs(np.asarray(expected0[0]))), rtol=1e-6)

    jit_value = jax.jit(lambda omega: first_order_linear_response(a_matrix, b_matrix, kappas, coeffs, omega, nt=8).response)(omegas)
    assert jit_value.shape == (2, 2, 8)
    jac = jax.jacfwd(lambda omega: first_order_linear_response(a_matrix, b_matrix, kappas, coeffs, omega, nt=8).z_norm)(omegas)
    assert jac.shape == (2, 2)


def test_second_order_linear_response_matches_scalar_oscillator_and_grad():
    mass = jnp.array([[2.0]])
    damping = jnp.array([[0.1]])
    stiffness = jnp.array([[5.0]])
    kappas = jnp.array([1.0, -1.0])
    coeffs = jnp.array([[1.0 + 0.2j, 1.0 - 0.2j]])
    omegas = jnp.array([0.75, 1.5])

    result = second_order_linear_response(mass, damping, stiffness, kappas, coeffs, omegas, epsilon=0.3, outdof=jnp.array([0]), nt=32)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, 32)
    dynamic0 = stiffness[0, 0] - (kappas * omegas[0]) ** 2 * mass[0, 0] + 1j * kappas * omegas[0] * damping[0, 0]
    modal0 = coeffs[0] / dynamic0
    expected0 = 0.3 * jnp.real(modal0 @ jnp.exp(1j * kappas[:, None] * phi[None, :]))
    np.testing.assert_allclose(np.asarray(result.response[0, 0]), np.asarray(expected0), rtol=1e-6, atol=1e-6)

    grad = jax.grad(lambda omega: jnp.sum(second_order_linear_response(mass, damping, stiffness, kappas, coeffs, jnp.array([omega]), nt=16).z_norm))(0.9)
    assert np.isfinite(np.asarray(grad))


def test_second_order_linear_response_conjugate_symmetric_branch_matches_matlab_convention():
    mass = jnp.array([[1.0]])
    damping = jnp.array([[0.2]])
    stiffness = jnp.array([[4.0]])
    kappas = jnp.array([1.0, -1.0])
    coeffs = jnp.array([[2.0 + 1.0j, 0.0 + 0.0j]])
    omega = jnp.array([0.5])

    result = second_order_linear_response(
        mass,
        damping,
        stiffness,
        kappas,
        coeffs,
        omega,
        nt=12,
        conjugate_symmetric=True,
    )
    first = coeffs[0, 0] / (stiffness[0, 0] - omega[0] ** 2 * mass[0, 0] + 1j * omega[0] * damping[0, 0])
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, 12)
    expected = jnp.real(jnp.array([first, jnp.conj(first)]) @ jnp.exp(1j * kappas[:, None] * phi[None, :]))
    np.testing.assert_allclose(np.asarray(result.response[0, 0]), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_assemble_auto_reduced_dynamics_concatenates_higher_order_terms():
    r0 = (
        MultiIndexPolynomial(coeffs=jnp.eye(2), ind=jnp.array([[1, 0], [0, 1]])),
        MultiIndexPolynomial(coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]), ind=jnp.array([[2, 0], [1, 1]])),
        MultiIndexPolynomial(coeffs=jnp.array([[5.0], [6.0]]), ind=jnp.array([[0, 2]])),
    )
    data = assemble_auto_reduced_dynamics(jnp.array([-1.0, -2.0]), r0)
    np.testing.assert_allclose(np.asarray(data.beta), np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]))
    np.testing.assert_array_equal(np.asarray(data.kappa), np.array([[2, 0], [1, 1], [0, 2]]))


def test_transient_traj_on_auto_ssm_linear_system_matches_exponential_and_jit_grad():
    r0 = (MultiIndexPolynomial(coeffs=jnp.eye(2), ind=jnp.array([[1, 0], [0, 1]])),)
    w0 = (MultiIndexPolynomial(coeffs=jnp.eye(2), ind=jnp.array([[1, 0], [0, 1]])),)
    lamd = jnp.array([-1.0, -2.0])
    q0 = jnp.array([1.0, 2.0])

    traj = transient_traj_on_auto_ssm(lamd, r0, w0, tf=1.0, nsteps=100, outdof=jnp.array([0, 1]), initial_reduced=q0)
    expected_final = q0 * jnp.exp(lamd)
    np.testing.assert_allclose(np.asarray(traj.red[-1]), np.asarray(expected_final), rtol=1e-7, atol=2e-7)
    np.testing.assert_allclose(np.asarray(traj.phy[-1]), np.asarray(expected_final), rtol=1e-7, atol=2e-7)

    jit_red = jax.jit(lambda initial: transient_traj_on_auto_ssm(lamd, r0, w0, 0.5, 20, jnp.array([0]), initial_reduced=initial).red)(q0)
    assert jit_red.shape == (21, 2)
    jac = jax.jacfwd(lambda initial: transient_traj_on_auto_ssm(lamd, r0, w0, 0.25, 10, jnp.array([0]), initial_reduced=initial).phy[-1, 0])(q0)
    assert jac.shape == (2,)


def test_transient_traj_on_auto_ssm_uses_linear_projection_when_needed():
    r0 = (MultiIndexPolynomial(coeffs=jnp.eye(1), ind=jnp.array([[1]])),)
    w0 = (MultiIndexPolynomial(coeffs=jnp.array([[2.0]]), ind=jnp.array([[1]])),)
    traj = transient_traj_on_auto_ssm(
        jnp.array([0.0]),
        r0,
        w0,
        tf=0.2,
        nsteps=2,
        outdof=jnp.array([0]),
        z0=jnp.array([3.0, 4.0]),
        left_eigenvectors=jnp.array([[1.0], [0.0]]),
        b_matrix=jnp.eye(2),
    )
    np.testing.assert_allclose(np.asarray(traj.red[:, 0]), np.array([3.0, 3.0, 3.0]))
    np.testing.assert_allclose(np.asarray(traj.phy[:, 0]), np.array([6.0, 6.0, 6.0]))


def test_reduced_dynamics_symbolic_autonomous_source_derived():
    r0 = (
        MultiIndexPolynomial(coeffs=jnp.eye(2, dtype=jnp.complex64), ind=jnp.array([[1, 0], [0, 1]])),
        MultiIndexPolynomial(coeffs=jnp.array([[2.0 + 3.0j], [2.0 - 3.0j]]), ind=jnp.array([[2, 1]])),
    )
    result = reduced_dynamics_symbolic(
        jnp.array([-0.1 + 1.5j, -0.1 - 1.5j]),
        r0,
        ReducedDynamicsSymbolicOptions(isauto=True, isdamped=True, num_digits=4),
    )
    assert result.rho_equations == ("rho_1_dot = -0.1*rho_1 + 2*rho_1^3",)
    assert result.theta_equations == ("theta_1_dot = 1.5 + 3*rho_1^2",)
    assert result.equations == result.rho_equations + result.theta_equations
