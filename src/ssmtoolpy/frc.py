"""Forced-response curve helper kernels."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ssmtoolpy.multi_index import MultiIndexPolynomial


Array = jnp.ndarray


class ReducedDynamicsHarmonic(NamedTuple):
    """One non-autonomous harmonic of a 2D reduced dynamics expansion.

    Differentiability
    -----------------
    Not differentiable as a container. Evaluation kernels are differentiable
    for fixed harmonic and multi-index structure.
    """

    kappa: Array
    terms: tuple[MultiIndexPolynomial, ...]


class ReducedDynamics2mDData(NamedTuple):
    """Explicit coefficient data for 2m-dimensional SSM reduced dynamics.

    ``nonauto_indices`` are zero-based Python mode indices. MATLAB stores these
    indices one-based in ``data.iNonauto``.

    Differentiability
    -----------------
    Not differentiable as a container. The ODE kernels are differentiable for
    fixed term structure and fixed non-autonomous index sets.
    """

    beta: tuple[Array, ...]
    kappa: tuple[Array, ...]
    lambda_real: Array
    lambda_imag: Array
    modal_frequencies: Array
    nonauto_indices: tuple[int, ...]
    nonauto_coefficients: Array
    is_base_force: bool = False


def frc_ab(rho: Array, omega: Array, gamma: Array, lam: complex | Array) -> tuple[Array, Array]:
    """Compute the real amplitude equations ``a`` and ``b`` from SSMTool.

    This ports ``SSMTool/src/frc/frc_ab.m``.

    Differentiability
    -----------------
    Differentiable with respect to ``rho``, ``omega``, ``gamma`` and ``lam``
    away from any downstream branch decisions made by callers. Representative
    tests exercise ``jax.grad`` and ``jax.vmap``.
    """

    rho = jnp.asarray(rho)
    omega = jnp.asarray(omega)
    gamma = jnp.asarray(gamma)
    lam = jnp.asarray(lam)
    powers = rho[..., None] ** (2 * jnp.arange(gamma.shape[0]) + 3)
    a = rho * jnp.real(lam) + jnp.sum(jnp.real(gamma) * powers, axis=-1)
    b = rho * (jnp.imag(lam) - omega) + jnp.sum(jnp.imag(gamma) * powers, axis=-1)
    return a, b


def cal_ab_dab(rho: Array, gamma: Array, lam: complex | Array) -> tuple[Array, Array, Array, Array]:
    """Compute autonomous polar terms and radius derivatives.

    This ports ``SSMTool/src/@SSM/private/cal_ab_dab.m``. Unlike
    :func:`frc_ab`, MATLAB's helper returns ``b`` before subtracting the forcing
    frequency term.

    Differentiability
    -----------------
    Differentiable with respect to ``rho``, ``gamma`` and ``lam``. The
    derivative ``db`` contains negative powers only if callers provide
    malformed powers; for the MATLAB odd-order normal form it is regular at
    positive ``rho``. Representative tests exercise ``jax.jacfwd``.
    """

    rho = jnp.asarray(rho)
    gamma = jnp.asarray(gamma)
    lam = jnp.asarray(lam)
    ell = jnp.arange(1, gamma.shape[0] + 1)
    powers_a = rho[..., None] ** (2 * ell + 1)
    powers_b = rho[..., None] ** (2 * ell)
    powers_da = rho[..., None] ** (2 * ell)
    powers_db = rho[..., None] ** (2 * ell - 1)
    a = rho * jnp.real(lam) + jnp.sum(jnp.real(gamma) * powers_a, axis=-1)
    b = jnp.imag(lam) + jnp.sum(jnp.imag(gamma) * powers_b, axis=-1)
    da = jnp.real(lam) + jnp.sum((2 * ell + 1) * jnp.real(gamma) * powers_da, axis=-1)
    db = jnp.sum((2 * ell) * jnp.imag(gamma) * powers_db, axis=-1)
    return a, b, da, db


def _first_row_sum(poly: MultiIndexPolynomial) -> Array:
    coeffs = jnp.asarray(poly.coeffs)
    if coeffs.size == 0:
        return jnp.asarray(0.0)
    return jnp.sum(coeffs[0, :])


def compute_reduced_dynamics_2d_polar(
    rho: Array,
    psi: Array,
    lam: complex | Array,
    gamma: Array,
    nonautonomous: tuple[ReducedDynamicsHarmonic, ...],
    omega: Array,
    epsilon: Array,
) -> tuple[Array, Array, Array]:
    """Evaluate the 2D polar reduced dynamics over a radius/phase grid.

    This ports the functional core of
    ``SSMTool/src/@SSM/private/compute_reduced_dynamics_2D_polar.m`` for
    explicit non-autonomous coefficient data. The mutable MATLAB behavior that
    recomputes coefficients from an ``SSM`` object is intentionally outside this
    kernel.

    Differentiability
    -----------------
    Differentiable for fixed harmonic/multi-index structure. Harmonic sign
    normalization and missing-forcing cases are discrete Python decisions.
    Representative tests exercise ``jax.grad``.
    """

    rho = jnp.asarray(rho)
    psi = jnp.asarray(psi)
    if not nonautonomous:
        a, b = frc_ab(rho, omega, gamma, lam)
        return a, b, jnp.asarray(1, dtype=jnp.int32)

    first = nonautonomous[0]
    kappa_raw = int(jnp.asarray(first.kappa))
    kappa0 = abs(kappa_raw) if kappa_raw != 0 else 1
    f = epsilon * _first_row_sum(first.terms[0])
    if kappa_raw < 0:
        f = jnp.conj(f)

    a, b = frc_ab(rho, kappa0 * omega, gamma, lam)
    rhodot = a + jnp.cos(psi) * jnp.real(f) + jnp.sin(psi) * jnp.imag(f)
    rhopsidot = b + jnp.cos(psi) * jnp.imag(f) - jnp.sin(psi) * jnp.real(f)

    for harmonic in nonautonomous:
        kappa = jnp.asarray(harmonic.kappa) / kappa0
        cos_k = jnp.cos(kappa * psi)
        sin_k = jnp.sin(kappa * psi)
        for poly in harmonic.terms[1:]:
            coeffs = jnp.asarray(poly.coeffs)
            ind = jnp.asarray(poly.ind)
            for col in range(coeffs.shape[1]):
                rj = coeffs[0, col]
                degree = jnp.sum(ind[col, :])
                spatial = rho**degree
                rhodot = rhodot + epsilon * spatial * (jnp.real(rj) * cos_k + jnp.imag(rj) * sin_k)
                rhopsidot = rhopsidot + epsilon * spatial * (jnp.imag(rj) * cos_k - jnp.real(rj) * sin_k)
    return rhodot, rhopsidot, jnp.asarray(kappa0, dtype=jnp.int32)


def ode_2d_ssm_cartesian(
    t: Array,
    x: Array,
    params: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    nonautonomous: tuple[ReducedDynamicsHarmonic, ...] = (),
) -> Array:
    """Evaluate 2D SSM reduced dynamics in Cartesian normal-form coordinates.

    This ports the explicit coefficient-evaluation part of
    ``SSMTool/src/@SSM/private/ode_2DSSM_cartesian.m``. Inputs are
    ``params = [Omega, epsilon]`` and ``x = [real(q), imag(q)]``.

    Differentiability
    -----------------
    Differentiable with respect to ``x``, parameters, and coefficients for
    fixed polynomial/harmonic structure. The MATLAB object mutation/recompute
    branch is not included. Representative tests exercise ``jax.jit`` and
    ``jax.jacfwd``.
    """

    t = jnp.asarray(t)
    x = jnp.asarray(x)
    params = jnp.asarray(params)
    omega = params[0]
    epsilon = params[1] if params.shape[0] > 1 else jnp.asarray(0.0, dtype=params.dtype)
    q1 = x[0] + 1j * x[1]
    q2 = x[0] - 1j * x[1]

    r = _first_row_sum(autonomous[0])
    y_complex = r * q1
    n_gamma = (len(autonomous) - 1) // 2
    for j in range(1, n_gamma + 1):
        poly = autonomous[2 * j]
        if poly.ind.size:
            target = jnp.asarray([j + 1, j], dtype=jnp.asarray(poly.ind).dtype)
            mask = jnp.all(jnp.asarray(poly.ind) == target, axis=1)
            gamma_j = jnp.sum(jnp.asarray(poly.coeffs)[0, :] * mask)
            y_complex = y_complex + gamma_j * q1 ** (j + 1) * q2**j

    for harmonic in nonautonomous:
        exp_kap = jnp.exp(1j * jnp.asarray(harmonic.kappa) * omega * t)
        if harmonic.terms:
            y_complex = y_complex + epsilon * _first_row_sum(harmonic.terms[0]) * exp_kap
        for poly in harmonic.terms[1:]:
            coeffs = jnp.asarray(poly.coeffs)
            ind = jnp.asarray(poly.ind)
            for col in range(coeffs.shape[1]):
                m1 = ind[col, 0]
                m2 = ind[col, 1]
                y_complex = y_complex + epsilon * coeffs[0, col] * q1**m1 * q2**m2 * exp_kap
    return jnp.asarray([jnp.real(y_complex), jnp.imag(y_complex)])


def ode_2d_ssm_cartesian_jac_x(
    t: Array,
    x: Array,
    params: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    nonautonomous: tuple[ReducedDynamicsHarmonic, ...] = (),
) -> Array:
    """Jacobian of :func:`ode_2d_ssm_cartesian` with respect to ``x``.

    This ports the behavior of
    ``SSMTool/src/@SSM/private/ode_2DSSM_cartesian_DFDX.m`` using JAX automatic
    differentiation on the functional Cartesian kernel.

    Differentiability
    -----------------
    Differentiable under the same fixed-structure assumptions as
    :func:`ode_2d_ssm_cartesian`.
    """

    return jax.jacfwd(lambda state: ode_2d_ssm_cartesian(t, state, params, autonomous, nonautonomous))(x)


def ode_2d_ssm_cartesian_jac_params(
    t: Array,
    x: Array,
    params: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    nonautonomous: tuple[ReducedDynamicsHarmonic, ...] = (),
    *,
    include_higher_nonautonomous: bool = True,
) -> Array:
    """Jacobian of :func:`ode_2d_ssm_cartesian` with respect to parameters.

    This ports the explicit coefficient-evaluation behavior of
    ``ode_2DSSM_cartesian_DFDP.m`` and ``ode_2DSSM_cartesian_fixROM_DFDP.m``.
    When ``include_higher_nonautonomous`` is ``False``, only the leading
    non-autonomous forcing term contributes, matching MATLAB's
    ``lead_nonaut_J_dp`` branch.

    Differentiability
    -----------------
    Differentiable for fixed polynomial/harmonic structure. Derivatives of
    frequency-dependent recomputed coefficients are not included, matching the
    active MATLAB code where the sensitivity-coefficient blocks are commented
    out.
    """

    active = nonautonomous
    if nonautonomous and not include_higher_nonautonomous:
        first = nonautonomous[0]
        active = (ReducedDynamicsHarmonic(kappa=first.kappa, terms=first.terms[:1]),)
    return jax.jacfwd(lambda pars: ode_2d_ssm_cartesian(t, x, pars, autonomous, active))(params)


def ode_2d_ssm_cartesian_fixrom(
    t: Array,
    x: Array,
    params: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    nonautonomous: tuple[ReducedDynamicsHarmonic, ...] = (),
) -> Array:
    """Fixed-ROM alias for :func:`ode_2d_ssm_cartesian`.

    This covers ``ode_2DSSM_cartesian_fixROM.m``. In the functional Python API
    coefficient recomputation is never implicit, so fixed-ROM and explicit
    coefficient evaluation are the same operation.

    Differentiability
    -----------------
    Differentiable under the same assumptions as :func:`ode_2d_ssm_cartesian`.
    """

    return ode_2d_ssm_cartesian(t, x, params, autonomous, nonautonomous)


def ode_2d_ssm_cartesian_fixrom_jac_x(
    t: Array,
    x: Array,
    params: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    nonautonomous: tuple[ReducedDynamicsHarmonic, ...] = (),
) -> Array:
    """Fixed-ROM Jacobian with respect to Cartesian reduced coordinates.

    This covers ``ode_2DSSM_cartesian_fixROM_DFDX.m``.

    Differentiability
    -----------------
    Differentiable under the same assumptions as
    :func:`ode_2d_ssm_cartesian_jac_x`.
    """

    return ode_2d_ssm_cartesian_jac_x(t, x, params, autonomous, nonautonomous)


def ode_2d_ssm_cartesian_fixrom_jac_params(
    t: Array,
    x: Array,
    params: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    nonautonomous: tuple[ReducedDynamicsHarmonic, ...] = (),
    *,
    include_higher_nonautonomous: bool = True,
) -> Array:
    """Fixed-ROM Jacobian with respect to ``[Omega, epsilon]``.

    This covers ``ode_2DSSM_cartesian_fixROM_DFDP.m``.

    Differentiability
    -----------------
    Differentiable under the same assumptions as
    :func:`ode_2d_ssm_cartesian_jac_params`.
    """

    return ode_2d_ssm_cartesian_jac_params(
        t,
        x,
        params,
        autonomous,
        nonautonomous,
        include_higher_nonautonomous=include_higher_nonautonomous,
    )


def _params_column(params: Array, index: int) -> Array:
    params = jnp.asarray(params)
    if params.ndim == 1:
        return params
    return params[:, index]


def _ode_2md_ssm_cartesian_single(z: Array, params: Array, data: ReducedDynamics2mDData) -> Array:
    z = jnp.asarray(z)
    params = jnp.asarray(params)
    m = len(data.modal_frequencies)
    z_re = z[0::2]
    z_im = z[1::2]
    omega = params[0]
    epsilon = params[1]
    q = z_re + 1j * z_im
    q_conj = jnp.conj(q)

    modal_frequencies = jnp.asarray(data.modal_frequencies)
    y_re = jnp.asarray(data.lambda_real) * z_re - jnp.asarray(data.lambda_imag) * z_im
    y_re = y_re + z_im * modal_frequencies * omega
    y_im = jnp.asarray(data.lambda_real) * z_im + jnp.asarray(data.lambda_imag) * z_re
    y_im = y_im - z_re * modal_frequencies * omega

    for mode in range(m):
        kappai = jnp.asarray(data.kappa[mode])
        betai = jnp.asarray(data.beta[mode])
        for term in range(betai.shape[0]):
            exponents = kappai[term, :]
            left = exponents[0::2]
            right = exponents[1::2]
            value = betai[term] * jnp.prod(q**left * q_conj**right)
            y_re = y_re.at[mode].add(jnp.real(value))
            y_im = y_im.at[mode].add(jnp.imag(value))

    nonauto_coefficients = jnp.asarray(data.nonauto_coefficients)
    for item, mode in enumerate(data.nonauto_indices):
        mode = int(mode)
        value = epsilon * nonauto_coefficients[item]
        if data.is_base_force:
            value = value * omega**2
        y_re = y_re.at[mode].add(jnp.real(value))
        y_im = y_im.at[mode].add(jnp.imag(value))

    y = jnp.zeros((2 * m,), dtype=jnp.result_type(z, params, data.lambda_real))
    y = y.at[0::2].set(y_re)
    y = y.at[1::2].set(y_im)
    return y


def ode_2md_ssm_cartesian(z: Array, params: Array, data: ReducedDynamics2mDData) -> Array:
    """Evaluate 2m-dimensional Cartesian SSM reduced dynamics.

    This ports ``SSMTool/src/@SSM/private/ode_2mDSSM_cartesian.m`` for an
    explicit immutable coefficient container.

    Differentiability
    -----------------
    Differentiable for fixed polynomial term structure and non-autonomous
    index sets. Representative tests exercise ``jax.jit`` and Jacobian
    transforms.
    """

    z = jnp.asarray(z)
    if z.ndim == 1:
        return _ode_2md_ssm_cartesian_single(z, params, data)
    columns = []
    for index in range(z.shape[1]):
        columns.append(_ode_2md_ssm_cartesian_single(z[:, index], _params_column(params, index), data))
    return jnp.stack(columns, axis=1)


def ode_2md_ssm_cartesian_jac_x(z: Array, params: Array, data: ReducedDynamics2mDData) -> Array:
    """State Jacobian of :func:`ode_2md_ssm_cartesian`.

    This covers ``ode_2mDSSM_cartesian_DFDX.m`` using JAX AD on the functional
    coefficient evaluator.

    Differentiability
    -----------------
    Differentiable under the same fixed-structure assumptions as
    :func:`ode_2md_ssm_cartesian`.
    """

    z = jnp.asarray(z)
    if z.ndim == 1:
        return jax.jacfwd(lambda state: _ode_2md_ssm_cartesian_single(state, params, data))(z)
    blocks = []
    for index in range(z.shape[1]):
        blocks.append(
            jax.jacfwd(lambda state, idx=index: _ode_2md_ssm_cartesian_single(state, _params_column(params, idx), data))(
                z[:, index]
            )
        )
    return jnp.stack(blocks, axis=2)


def ode_2md_ssm_cartesian_jac_params(z: Array, params: Array, data: ReducedDynamics2mDData) -> Array:
    """Parameter Jacobian of :func:`ode_2md_ssm_cartesian`.

    This covers ``ode_2mDSSM_cartesian_DFDP.m``.

    Differentiability
    -----------------
    Differentiable under the same fixed-structure assumptions as
    :func:`ode_2md_ssm_cartesian`.
    """

    z = jnp.asarray(z)
    if z.ndim == 1:
        return jax.jacfwd(lambda pars: _ode_2md_ssm_cartesian_single(z, pars, data))(params)
    blocks = []
    for index in range(z.shape[1]):
        blocks.append(
            jax.jacfwd(lambda pars, idx=index: _ode_2md_ssm_cartesian_single(z[:, idx], pars, data))(
                _params_column(params, index)
            )
        )
    return jnp.stack(blocks, axis=2)


def _ode_2md_ssm_polar_single(z: Array, params: Array, data: ReducedDynamics2mDData) -> Array:
    z = jnp.asarray(z)
    params = jnp.asarray(params)
    m = len(data.modal_frequencies)
    rho = z[0::2]
    theta = z[1::2]
    omega = params[0]
    epsilon = params[1]

    modal_frequencies = jnp.asarray(data.modal_frequencies)
    y_rho = jnp.asarray(data.lambda_real) * rho
    y_theta = jnp.asarray(data.lambda_imag) - modal_frequencies * omega
    eye = jnp.eye(m, dtype=rho.dtype)

    for mode in range(m):
        kappai = jnp.asarray(data.kappa[mode])
        betai = jnp.asarray(data.beta[mode])
        e_i = eye[mode, :]
        for term in range(betai.shape[0]):
            exponents = kappai[term, :]
            left = exponents[0::2]
            right = exponents[1::2]
            angle = jnp.sum((left - right - e_i) * theta)
            rho_power = jnp.prod(rho ** (left + right))
            be = betai[term]
            y_rho = y_rho.at[mode].add(rho_power * (jnp.real(be) * jnp.cos(angle) - jnp.imag(be) * jnp.sin(angle)))
            y_theta = y_theta.at[mode].add(
                rho_power / rho[mode] * (jnp.real(be) * jnp.sin(angle) + jnp.imag(be) * jnp.cos(angle))
            )

    nonauto_coefficients = jnp.asarray(data.nonauto_coefficients)
    for item, mode in enumerate(data.nonauto_indices):
        mode = int(mode)
        value = epsilon * nonauto_coefficients[item]
        if data.is_base_force:
            value = value * omega**2
        r_re = jnp.real(value)
        r_im = jnp.imag(value)
        y_rho = y_rho.at[mode].add(r_re * jnp.cos(theta[mode]) + r_im * jnp.sin(theta[mode]))
        y_theta = y_theta.at[mode].add(
            -r_re * jnp.sin(theta[mode]) / rho[mode] + r_im * jnp.cos(theta[mode]) / rho[mode]
        )

    y = jnp.zeros((2 * m,), dtype=jnp.result_type(z, params, data.lambda_real))
    y = y.at[0::2].set(y_rho)
    y = y.at[1::2].set(y_theta)
    return y


def ode_2md_ssm_polar(z: Array, params: Array, data: ReducedDynamics2mDData) -> Array:
    """Evaluate 2m-dimensional polar SSM reduced dynamics.

    This ports ``SSMTool/src/@SSM/private/ode_2mDSSM_polar.m``.

    Differentiability
    -----------------
    Differentiable for fixed structure and positive radii. The polar phase
    equation contains divisions by ``rho``.
    """

    z = jnp.asarray(z)
    if z.ndim == 1:
        return _ode_2md_ssm_polar_single(z, params, data)
    columns = []
    for index in range(z.shape[1]):
        columns.append(_ode_2md_ssm_polar_single(z[:, index], _params_column(params, index), data))
    return jnp.stack(columns, axis=1)


def ode_2md_ssm_polar_jac_x(z: Array, params: Array, data: ReducedDynamics2mDData) -> Array:
    """State Jacobian of :func:`ode_2md_ssm_polar`.

    This covers ``ode_2mDSSM_polar_DFDX.m`` using JAX AD.

    Differentiability
    -----------------
    Differentiable for fixed structure and positive radii.
    """

    z = jnp.asarray(z)
    if z.ndim == 1:
        return jax.jacfwd(lambda state: _ode_2md_ssm_polar_single(state, params, data))(z)
    blocks = []
    for index in range(z.shape[1]):
        blocks.append(
            jax.jacfwd(lambda state, idx=index: _ode_2md_ssm_polar_single(state, _params_column(params, idx), data))(
                z[:, index]
            )
        )
    return jnp.stack(blocks, axis=2)


def ode_2md_ssm_polar_jac_params(z: Array, params: Array, data: ReducedDynamics2mDData) -> Array:
    """Parameter Jacobian of :func:`ode_2md_ssm_polar`.

    This covers ``ode_2mDSSM_polar_DFDP.m``.

    Differentiability
    -----------------
    Differentiable for fixed structure and positive radii.
    """

    z = jnp.asarray(z)
    if z.ndim == 1:
        return jax.jacfwd(lambda pars: _ode_2md_ssm_polar_single(z, pars, data))(params)
    blocks = []
    for index in range(z.shape[1]):
        blocks.append(
            jax.jacfwd(lambda pars, idx=index: _ode_2md_ssm_polar_single(z[:, idx], pars, data))(
                _params_column(params, index)
            )
        )
    return jnp.stack(blocks, axis=2)


def compute_gamma(reduced_dynamics: tuple[MultiIndexPolynomial, ...]) -> Array:
    """Extract resonant normal-form coefficients from reduced dynamics.

    This ports ``SSMTool/src/frc/compute_gamma.m``. For each ``j``, the
    coefficient at order ``2*j + 1`` and multi-index ``[j + 1, j]`` is selected
    from the first reduced-dynamics row.

    Differentiability
    -----------------
    Not differentiable as a full operation because it performs discrete
    multi-index lookup. The selected coefficient values remain JAX arrays.
    """

    order = len(reduced_dynamics)
    n_gamma = (order - 1) // 2
    gamma = []
    for j in range(1, n_gamma + 1):
        poly = reduced_dynamics[2 * j]
        coeffs = jnp.asarray(poly.coeffs)
        ind = jnp.asarray(poly.ind, dtype=jnp.int32)
        value = jnp.asarray(0, dtype=coeffs.dtype if coeffs.size else jnp.float32)
        if ind.size:
            target = jnp.asarray([j + 1, j], dtype=jnp.int32)
            matches = jnp.all(ind == target, axis=1)
            if bool(jnp.any(matches)):
                loc = int(jnp.argmax(matches))
                value = coeffs[0, loc]
        gamma.append(value)
    return jnp.asarray(gamma)


def frc_psi(rho: Array, omega: Array, gamma: Array, lam: complex | Array, forcing: complex | Array) -> Array:
    """Compute the FRC phase angle ``psi``.

    Differentiability
    -----------------
    Differentiable away from the branch cut and undefined point of ``atan2``.
    Representative tests exercise ``jax.grad`` and ``jax.vmap``.
    """

    a, b = frc_ab(rho, omega, gamma, lam)
    forcing = jnp.asarray(forcing)
    numerator = rho * b * jnp.real(forcing) - a * jnp.imag(forcing)
    denominator = -a * jnp.real(forcing) - rho * b * jnp.imag(forcing)
    return jnp.atan2(numerator, denominator)


def frc_jacobian(
    rho: Array,
    psi: Array,
    gamma: Array,
    lam: complex | Array,
    epsilon: Array,
    forcing: complex | Array,
) -> Array:
    """Evaluate the 2D polar reduced-dynamics Jacobian used for FRC stability.

    Differentiability
    -----------------
    Differentiable for ``rho != 0`` and fixed polynomial order. Representative
    tests exercise ``jax.jacfwd``.
    """

    rho = jnp.asarray(rho)
    psi = jnp.asarray(psi)
    gamma = jnp.asarray(gamma)
    lam = jnp.asarray(lam)
    epsilon = jnp.asarray(epsilon)
    forcing = jnp.asarray(forcing)
    c = epsilon * (jnp.real(forcing) * jnp.cos(psi) + jnp.imag(forcing) * jnp.sin(psi))
    d = epsilon * (-jnp.real(forcing) * jnp.sin(psi) + jnp.imag(forcing) * jnp.cos(psi))

    ell = jnp.arange(1, gamma.shape[0] + 1)
    j11 = jnp.real(lam) + jnp.sum(jnp.real(gamma) * (2 * ell + 1) * rho ** (2 * ell))
    j21 = -d / (rho**2) + jnp.sum(jnp.imag(gamma) * (2 * ell) * rho ** (2 * ell - 1))
    return jnp.asarray([[j11, d], [j21, -c / rho]])


def check_stability(
    rho: Array,
    psi: Array,
    gamma: Array,
    lam: complex | Array,
    epsilon: Array,
    forcing: complex | Array,
) -> Array:
    """Classify FRC fixed points by the Routh stability criterion.

    Differentiability
    -----------------
    Not differentiable. This routine thresholds trace and determinant signs.
    """

    rho_array = jnp.asarray(rho)
    original_shape = rho_array.shape
    rho = jnp.ravel(rho_array)
    psi = jnp.ravel(jnp.asarray(psi))
    values = []
    for index in range(rho.shape[0]):
        jac = frc_jacobian(rho[index], psi[index], gamma, lam, epsilon, forcing)
        trace = jnp.trace(jac)
        det = jnp.linalg.det(jac)
        values.append((det > 0) & (trace < 0))
    return jnp.asarray(values, dtype=bool).reshape(original_shape)


def get_contour_xy(contour_matrix: Array) -> tuple[Array, Array, int]:
    """Extract x/y contour coordinates from a MATLAB ``contourc`` matrix.

    Differentiability
    -----------------
    Not differentiable. This parses a plotting-oriented encoded contour matrix.
    """

    contour_matrix = jnp.asarray(contour_matrix)
    x_values = []
    y_values = []
    column = 0
    while column < contour_matrix.shape[1]:
        n_points = int(contour_matrix[1, column])
        next_column = column + n_points
        x_values.extend([jnp.nan, *[contour_matrix[0, idx] for idx in range(column + 1, next_column + 1)]])
        y_values.extend([jnp.nan, *[contour_matrix[1, idx] for idx in range(column + 1, next_column + 1)]])
        column = next_column + 1
    n_components = sum(1 for value in x_values if bool(jnp.isnan(value)))
    return jnp.asarray(x_values, dtype=contour_matrix.dtype), jnp.asarray(y_values, dtype=contour_matrix.dtype), n_components


def _zero_segments(x_grid: Array, y_grid: Array, values: Array) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    x_grid = jnp.asarray(x_grid)
    y_grid = jnp.asarray(y_grid)
    values = jnp.asarray(values)
    if x_grid.ndim == 1 and y_grid.ndim == 1:
        xs = x_grid
        ys = y_grid
    else:
        xs = x_grid[0, :]
        ys = y_grid[:, 0]
    segments = []
    for row in range(len(ys) - 1):
        for col in range(len(xs) - 1):
            corners = [
                (float(xs[col]), float(ys[row]), float(values[row, col])),
                (float(xs[col + 1]), float(ys[row]), float(values[row, col + 1])),
                (float(xs[col + 1]), float(ys[row + 1]), float(values[row + 1, col + 1])),
                (float(xs[col]), float(ys[row + 1]), float(values[row + 1, col])),
            ]
            points = []
            for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
                x1, y1, z1 = corners[first]
                x2, y2, z2 = corners[second]
                if z1 == 0.0 and z2 == 0.0:
                    points.extend([(x1, y1), (x2, y2)])
                elif z1 == 0.0:
                    points.append((x1, y1))
                elif z2 == 0.0:
                    points.append((x2, y2))
                elif z1 * z2 < 0.0:
                    t = -z1 / (z2 - z1)
                    points.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            unique_points = []
            for point in points:
                if not any(abs(point[0] - old[0]) < 1e-12 and abs(point[1] - old[1]) < 1e-12 for old in unique_points):
                    unique_points.append(point)
            if len(unique_points) >= 2:
                segments.append((unique_points[0], unique_points[1]))
    return segments


def _segment_intersection(
    segment_a: tuple[tuple[float, float], tuple[float, float]],
    segment_b: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    (x1, y1), (x2, y2) = segment_a
    (x3, y3), (x4, y4) = segment_b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-14:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den

    def within(value: float, a: float, b: float) -> bool:
        return min(a, b) - 1e-12 <= value <= max(a, b) + 1e-12

    if within(px, x1, x2) and within(py, y1, y2) and within(px, x3, x4) and within(py, y3, y4):
        return px, py
    return None


def compute_fixed_points_2d(x: Array, y: Array, x_dot: Array, y_dot: Array) -> tuple[Array, Array]:
    """Approximate fixed points of a 2D vector field sampled on a grid.

    This is a lightweight marching-squares replacement for MATLAB's
    ``contourc``/``polyxpoly`` path in ``compute_fixed_points_2D.m``.

    Differentiability
    -----------------
    Not differentiable. It performs sign tests, segment construction, and
    geometric intersection selection.
    """

    x_segments = _zero_segments(x, y, x_dot)
    y_segments = _zero_segments(x, y, y_dot)
    points = []
    for x_segment in x_segments:
        for y_segment in y_segments:
            point = _segment_intersection(x_segment, y_segment)
            if point is not None and not any(abs(point[0] - old[0]) < 1e-10 and abs(point[1] - old[1]) < 1e-10 for old in points):
                points.append(point)
    if not points:
        dtype = jnp.asarray(x).dtype
        return jnp.asarray([], dtype=dtype), jnp.asarray([], dtype=dtype)
    return jnp.asarray([point[0] for point in points]), jnp.asarray([point[1] for point in points])
