import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy import (
    AutoReducedDynamicsData,
    MultiIndexPolynomial,
    NonAutonomousTerm,
    ProjectionData,
    auto_red_dyn,
    extract_output,
    nonlinear_projection_objective,
    project_to_ssm_linear,
    real_to_conjugate_state,
    reduced_to_full_traj,
    solve_invariance_equation,
    spblkdiag,
    squared_distance_to_point_ssm,
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
