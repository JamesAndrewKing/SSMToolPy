import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy import (
    MasterSubspace,
    MultiIndexPolynomial,
    NonAutonomousResonanceData,
    ResonanceAnalysis,
    autonomous_invariance_residual,
    autonomous_resonant_terms,
    check_comp_type,
    check_ds_type,
    choose_master_subspace,
    coeffs_composition,
    coeffs_mixed_terms,
    compute_auto_invariance_error,
    dfnl_nonintrusive,
    dfnl_semi_intrusive,
    fnl_nonintrusive,
    fnl_semi_intrusive,
    nonautonomous_assemble_coefficients,
    nonautonomous_conjugate_reduction,
    nonautonomous_resonant_terms,
    nonautonomous_struct_setup,
    nonautonomous_w1r0_plus_w0r1,
    resonance_analysis,
    step_polynomial,
)


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


def test_check_ds_type_matches_matlab_decision_rules():
    assert check_ds_type(jnp.eye(2), jnp.eye(2)) == "real"
    assert check_ds_type(jnp.eye(2, dtype=jnp.complex64), jnp.eye(2)) == "complex"
    assert check_ds_type(jnp.eye(2), jnp.eye(2), ({"coeffs": jnp.array([1.0 + 0.0j])},)) == "complex"
    assert check_ds_type(jnp.eye(2), jnp.eye(2), dim_manifold=1) == "complex"
    assert check_ds_type(jnp.eye(2), jnp.eye(2), choose_complex_comp=True) == "complex"


def test_check_comp_type_matches_matlab_decision_rules():
    assert check_comp_type(1, "second", "real", dim_manifold=2) == ("first", "real")
    assert check_comp_type(2, "second", "real", dim_manifold=2) == ("second", "real")
    assert check_comp_type(2, "second", "complex", dim_manifold=2) == ("first", "complex")
    assert check_comp_type(2, "second", "real", dim_manifold=1) == ("first", "complex")


def test_autonomous_resonant_terms_source_derived_zero_based():
    lambda_master = jnp.array([-1.0 + 2.0j, -2.0 + 4.0j])
    lambda_combinations = jnp.array([-1.0 + 2.0j, -3.0 + 6.0j, -2.0 + 4.0j])
    modes, combinations = autonomous_resonant_terms(lambda_master, lambda_combinations, reltol=0.1)
    np.testing.assert_array_equal(np.asarray(modes), np.array([0, 1]))
    np.testing.assert_array_equal(np.asarray(combinations), np.array([0, 2]))


def test_resonance_analysis_detects_outer_and_inner_resonances():
    outer = resonance_analysis(jnp.array([-1.0 + 1.0j, -1.0 - 1.0j]), jnp.array([-2.0 + 0.0j, -3.0 + 0.0j]), reltol=1e-8)
    assert isinstance(outer, ResonanceAnalysis)
    assert outer.outer.occurs is True
    assert outer.outer.sigma == 3
    np.testing.assert_array_equal(np.asarray(outer.outer.combinations[0]), np.array([1, 1]))
    np.testing.assert_allclose(np.asarray(outer.outer.eigs[0]), -2.0 + 0.0j)
    assert outer.inner.sigma == 3

    inner = resonance_analysis(jnp.array([-1.0, -2.0]), jnp.array([-5.0]), reltol=1e-8)
    assert inner.inner.occurs is True
    assert any(np.array_equal(row, np.array([2, 0])) for row in np.asarray(inner.inner.combinations))
    assert any(np.isclose(value, -2.0) for value in np.asarray(inner.inner.eigs))


def test_choose_master_subspace_selects_modes_and_normal_modes():
    eigenvalues = jnp.array([-1.0 + 1.0j, -1.0 - 1.0j, -2.0 + 0.0j, -3.0 + 0.0j])
    v = jnp.arange(16.0).reshape(4, 4)
    w = jnp.arange(100.0, 116.0).reshape(4, 4)
    subspace = choose_master_subspace(eigenvalues, v, w, jnp.array([0, 1]), reltol=1e-8)

    assert isinstance(subspace, MasterSubspace)
    np.testing.assert_allclose(np.asarray(subspace.spectrum), np.array([-1.0 + 1.0j, -1.0 - 1.0j]))
    np.testing.assert_allclose(np.asarray(subspace.basis), np.asarray(v[:, :2]))
    np.testing.assert_allclose(np.asarray(subspace.adjoint_basis), np.asarray(w[:, :2]))
    np.testing.assert_array_equal(np.asarray(subspace.normal_modes), np.array([2, 3]))
    assert subspace.resonance.outer.occurs is True


def test_autonomous_invariance_residual_matches_exact_linear_invariance_and_grad():
    w = (
        MultiIndexPolynomial(
            coeffs=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            ind=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        ),
    )
    r = (
        MultiIndexPolynomial(
            coeffs=jnp.array([[1.0, 0.0], [0.0, 2.0]]),
            ind=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        ),
    )
    a_matrix = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    points = jnp.array([[1.0 + 2.0j, -0.5 + 0.25j], [1.0 - 2.0j, -0.5 - 0.25j]])
    residual = autonomous_invariance_residual(a_matrix, jnp.eye(2), w, r, points)
    np.testing.assert_allclose(np.asarray(residual), np.zeros(2), atol=1e-6)

    nonzero_grad = jax.grad(
        lambda values: jnp.sum(
            autonomous_invariance_residual(
                jnp.zeros((2, 2)),
                jnp.eye(2),
                w,
                r,
                values.reshape(2, 1),
            )
        )
    )(jnp.array([1.0, 2.0]))
    assert nonzero_grad.shape == (2,)
    assert np.all(np.isfinite(np.asarray(nonzero_grad)))


def test_compute_auto_invariance_error_2d_source_derived():
    w = (
        MultiIndexPolynomial(
            coeffs=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
            ind=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        ),
    )
    r = (
        MultiIndexPolynomial(
            coeffs=jnp.array([[1.0, 0.0], [0.0, 2.0]]),
            ind=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        ),
    )
    got = compute_auto_invariance_error(
        jnp.array([[1.0, 0.0], [0.0, 2.0]]),
        jnp.eye(2),
        w,
        r,
        jnp.array([0.1, 0.2]),
        jnp.array([1]),
        8,
    )
    np.testing.assert_allclose(np.asarray(got), np.zeros((1, 2)), atol=1e-6)


def test_nonautonomous_resonant_terms_zero_order_source_derived():
    data = NonAutonomousResonanceData(
        omega=jnp.array([1.0]),
        lambda_master=jnp.array([1j, 2j]),
        dim=2,
        reltol=0.1,
    )
    modes, harmonics, k_lambda = nonautonomous_resonant_terms(None, jnp.array([[1.0, 2.0, 3.0]]), data, "zero")
    np.testing.assert_array_equal(np.asarray(modes), np.array([0, 1]))
    np.testing.assert_array_equal(np.asarray(harmonics), np.array([0, 1]))
    assert k_lambda.shape == (0,)


def test_nonautonomous_resonant_terms_order_k_source_derived():
    data = NonAutonomousResonanceData(
        omega=jnp.array([1.0]),
        lambda_master=jnp.array([1j, 2j]),
        dim=2,
        reltol=0.1,
    )
    modes, multi_indices, k_lambda = nonautonomous_resonant_terms(2, jnp.array([0.0]), data, "k")
    np.testing.assert_array_equal(np.asarray(modes), np.array([1]))
    np.testing.assert_array_equal(np.asarray(multi_indices), np.array([0]))
    np.testing.assert_allclose(np.asarray(k_lambda), np.array([2j, 3j, 4j]))


def test_nonautonomous_conjugate_reduction_matches_matlab_example():
    kappa_set = jnp.array([1, -1, 2, 3, -3])
    forcing = jnp.array([[1.0, 1.0, 2.0, 3.0, 4.0]])
    red_conj, map_conj = nonautonomous_conjugate_reduction(kappa_set, forcing)

    np.testing.assert_array_equal(np.asarray(red_conj), np.array([0, 2, 3, 4]))
    assert [np.asarray(item).tolist() for item in map_conj] == [[0, 1], [2], [3], [4]]


def test_nonautonomous_conjugate_reduction_detects_complex_conjugates_and_duplicates():
    kappa_set = jnp.array([2, -2])
    forcing = jnp.array([[1.0 + 2.0j, 1.0 - 2.0j], [3.0 - 1.0j, 3.0 + 1.0j]])
    red_conj, map_conj = nonautonomous_conjugate_reduction(kappa_set, forcing)
    np.testing.assert_array_equal(np.asarray(red_conj), np.array([0]))
    assert [np.asarray(item).tolist() for item in map_conj] == [[0, 1]]

    with np.testing.assert_raises(ValueError):
        nonautonomous_conjugate_reduction(jnp.array([1, 1]), jnp.ones((1, 2)))


def test_nonautonomous_struct_setup_source_derived_shapes():
    forcing = (
        {
            "kappa": jnp.array([1.0]),
            "terms": (
                MultiIndexPolynomial(jnp.ones((3, 1)), jnp.zeros((1, 1), dtype=jnp.int32)),
                MultiIndexPolynomial(jnp.ones((3, 2)), jnp.zeros((2, 1), dtype=jnp.int32)),
            ),
        },
        {
            "kappa": jnp.array([-1.0]),
            "terms": (MultiIndexPolynomial(jnp.ones((3, 1)), jnp.zeros((1, 1), dtype=jnp.int32)),),
        },
    )
    structure = nonautonomous_struct_setup(dim=2, state_dim=3, order=2, forcing=forcing)

    np.testing.assert_allclose(np.asarray(structure.kappas), np.array([[1.0, -1.0]]))
    np.testing.assert_array_equal(np.asarray(structure.forcing_orders), np.array([2, 1]))
    assert len(structure.w1) == 2
    assert len(structure.w1[0].terms) == 3
    assert structure.w1[0].terms[0].coeffs.shape == (3, 1)
    assert structure.w1[0].terms[1].coeffs.shape == (3, 0)
    assert structure.r1[0].terms[0].coeffs.shape == (2, 1)
    assert structure.r1[0].terms[1].ind.shape == (0, 2)


def test_nonautonomous_assemble_coefficients_updates_one_harmonic_and_grad():
    structure = nonautonomous_struct_setup(
        dim=2,
        state_dim=3,
        order=2,
        forcing=({"kappa": jnp.array([1.0]), "terms": (MultiIndexPolynomial(jnp.ones((3, 1)), jnp.zeros((1, 2), dtype=jnp.int32)),)},),
    )
    w_coeffs = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    r_coeffs = jnp.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
    w_updated, r_updated = nonautonomous_assemble_coefficients(structure.w1[0], structure.r1[0], w_coeffs, r_coeffs, 2, dim=2)

    assert w_updated.kappa.shape == (1,)
    np.testing.assert_allclose(np.asarray(w_updated.terms[2].coeffs), np.asarray(w_coeffs))
    np.testing.assert_allclose(np.asarray(r_updated.terms[2].coeffs), np.asarray(r_coeffs))
    np.testing.assert_array_equal(np.asarray(w_updated.terms[2].ind), np.array([[2, 0], [1, 1], [0, 2]]))
    np.testing.assert_array_equal(np.asarray(r_updated.terms[2].ind), np.array([[2, 0], [1, 1], [0, 2]]))

    def scalar(values):
        updated, _ = nonautonomous_assemble_coefficients(structure.w1[0], structure.r1[0], values.reshape(3, 3), r_coeffs, 2, dim=2)
        return jnp.sum(updated.terms[2].coeffs)

    grad = jax.grad(scalar)(w_coeffs.ravel())
    np.testing.assert_allclose(np.asarray(grad), np.ones(9))


def test_nonautonomous_w1r0_plus_w0r1_source_derived_and_grad():
    empty_w0_order_1 = MultiIndexPolynomial(jnp.zeros((2, 0)), jnp.zeros((0, 2), dtype=jnp.int32))
    w0_order_2 = MultiIndexPolynomial(
        coeffs=jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ind=jnp.array([[2, 0], [1, 1], [0, 2]], dtype=jnp.int32),
    )
    w0 = (empty_w0_order_1, w0_order_2)
    r0 = (
        MultiIndexPolynomial(jnp.zeros((2, 0)), jnp.zeros((0, 2), dtype=jnp.int32)),
        MultiIndexPolynomial(jnp.zeros((2, 0)), jnp.zeros((0, 2), dtype=jnp.int32)),
    )
    w1 = (
        MultiIndexPolynomial(jnp.array([[2.0], [3.0]]), jnp.array([[0, 0]], dtype=jnp.int32)),
    )
    r1 = (
        MultiIndexPolynomial(jnp.array([[0.5], [1.5]]), jnp.array([[0, 0]], dtype=jnp.int32)),
    )

    got = nonautonomous_w1r0_plus_w0r1(1, w0, w1, r0, r1, dim=2, output_dim=2)
    expected = np.array([[4.0, 10.0], [11.5, 20.5]])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6)

    def scalar(values):
        local_w0 = (empty_w0_order_1, MultiIndexPolynomial(values.reshape(2, 3), w0_order_2.ind))
        return jnp.sum(nonautonomous_w1r0_plus_w0r1(1, local_w0, w1, r0, r1, dim=2, output_dim=2))

    grad = jax.grad(scalar)(w0_order_2.coeffs.ravel())
    np.testing.assert_allclose(np.asarray(grad), np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]), rtol=1e-6)


def test_step_polynomial_quadratic_polarization_and_grad():
    def nonlinear(vector):
        return vector**2

    v1 = jnp.array([1.0, 3.0])
    v2 = jnp.array([2.0, 4.0])
    same = step_polynomial(nonlinear, (v1, v1), jnp.array([[1, 1], [0, 0]], dtype=jnp.int32), 2)
    mixed = step_polynomial(nonlinear, (v1, v2), jnp.array([[1, 0], [0, 1]], dtype=jnp.int32), 2)
    np.testing.assert_allclose(np.asarray(same), np.array([1.0, 9.0]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(mixed), np.array([4.0, 24.0]), rtol=1e-6)

    grad = jax.grad(lambda x: jnp.real(jnp.sum(step_polynomial(nonlinear, (x, v2), jnp.array([[1, 0], [0, 1]], dtype=jnp.int32), 2))))(v1)
    np.testing.assert_allclose(np.asarray(grad), np.array([4.0, 8.0]), rtol=1e-6)


def test_fnl_nonintrusive_revlex_quadratic_source_derived_and_grad():
    w = (
        MultiIndexPolynomial(
            coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            ind=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        ),
    )
    k_indices = jnp.array([[2, 1, 0], [0, 1, 2]], dtype=jnp.int32)

    got = fnl_nonintrusive(lambda vector: vector**2, w, 2, k_indices, input_dim=2)
    expected = np.array([[1.0, 4.0, 4.0], [9.0, 24.0, 16.0]])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6)

    def scalar(values):
        local_w = (MultiIndexPolynomial(coeffs=values.reshape(2, 2), ind=w[0].ind),)
        return jnp.real(jnp.sum(fnl_nonintrusive(lambda vector: vector**2, local_w, 2, k_indices, input_dim=2)))

    grad = jax.grad(scalar)(w[0].coeffs.ravel())
    assert grad.shape == (4,)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_fnl_semi_intrusive_revlex_symmetric_source_derived_and_grad():
    w = (
        MultiIndexPolynomial(
            coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            ind=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        ),
    )
    k_indices = jnp.array([[2, 1, 0], [0, 1, 2]], dtype=jnp.int32)

    def bilinear(vectors):
        return vectors[0] * vectors[1]

    got = fnl_semi_intrusive(bilinear, w, 2, k_indices, input_dim=2, symmetric=True)
    expected = np.array([[1.0, 4.0, 4.0], [9.0, 24.0, 16.0]])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6)

    jac = jax.jacfwd(lambda coeffs: fnl_semi_intrusive(bilinear, (MultiIndexPolynomial(coeffs.reshape(2, 2), w[0].ind),), 2, k_indices, input_dim=2).reshape(-1))(
        w[0].coeffs.ravel()
    )
    assert jac.shape == (6, 4)


def test_dfnl_nonintrusive_revlex_quadratic_source_derived_and_grad():
    w = (
        MultiIndexPolynomial(
            coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            ind=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        ),
    )
    x_series = (
        (
            MultiIndexPolynomial(jnp.array([[10.0], [20.0]]), jnp.array([[0, 0]], dtype=jnp.int32)),
            MultiIndexPolynomial(jnp.zeros((2, 0)), jnp.zeros((0, 2), dtype=jnp.int32)),
        ),
    )
    m_indices = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)

    def jacobian(vector):
        return jnp.diag(2.0 * vector)

    got = dfnl_nonintrusive(jacobian, 2, w, x_series, m_indices, input_dim=2)
    expected = np.array([[20.0, 40.0], [120.0, 160.0]])
    assert len(got) == 1
    np.testing.assert_allclose(np.asarray(jnp.real(got[0])), expected, rtol=1e-6)

    def scalar(values):
        local_w = (MultiIndexPolynomial(values.reshape(2, 2), w[0].ind),)
        return jnp.real(jnp.sum(dfnl_nonintrusive(jacobian, 2, local_w, x_series, m_indices, input_dim=2)[0]))

    grad = jax.grad(scalar)(w[0].coeffs.ravel())
    np.testing.assert_allclose(np.asarray(grad), np.array([20.0, 20.0, 40.0, 40.0]), rtol=1e-6)


def test_dfnl_semi_intrusive_revlex_quadratic_source_derived_and_jacfwd():
    w = (
        MultiIndexPolynomial(
            coeffs=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            ind=jnp.array([[1, 0], [0, 1]], dtype=jnp.int32),
        ),
    )
    x_series = (
        (
            MultiIndexPolynomial(jnp.array([[10.0], [20.0]]), jnp.array([[0, 0]], dtype=jnp.int32)),
            MultiIndexPolynomial(jnp.zeros((2, 0)), jnp.zeros((0, 2), dtype=jnp.int32)),
        ),
    )
    m_indices = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)

    def jacobian_action(vectors):
        return 2.0 * vectors[0] * vectors[1]

    got = dfnl_semi_intrusive(jacobian_action, 2, w, x_series, m_indices, input_dim=2)
    expected = np.array([[20.0, 40.0], [120.0, 160.0]])
    assert len(got) == 1
    np.testing.assert_allclose(np.asarray(got[0]), expected, rtol=1e-6)

    jac = jax.jacfwd(lambda coeffs: dfnl_semi_intrusive(jacobian_action, 2, (MultiIndexPolynomial(coeffs.reshape(2, 2), w[0].ind),), x_series, m_indices, input_dim=2)[0].reshape(-1))(
        w[0].coeffs.ravel()
    )
    assert jac.shape == (4, 4)
