"""Functional helpers for SSM reduced-dynamics workflows."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from ssmtoolpy.frc import ReducedDynamics2mDData
from ssmtoolpy.multi_index import MultiIndexPolynomial


Array = jnp.ndarray


class ReducedDynamicsData(NamedTuple):
    """Data bundle for 2mD SSM reduced dynamics.

    Differentiability
    -----------------
    Not differentiable as a container.
    """

    beta: tuple[Array, ...]
    kappa: tuple[Array, ...]
    lambda_real: Array
    lambda_imag: Array
    modal_frequencies: Array
    nonauto_indices: tuple[int, ...]
    nonauto_coefficients: Array
    nonauto_kappas: Array
    order: int
    modes: Array


class POAmplitudeData(NamedTuple):
    """Data bundle for periodic-orbit amplitude objectives.

    Differentiability
    -----------------
    Not differentiable as a container. Numeric fields can be used in
    differentiable downstream objectives for fixed structure.
    """

    hatr_values: Array
    hatr_positions: tuple[Array, ...]
    cind: Array
    dind: Array
    coeffs: Array
    optdof: Array
    q: Array
    qbar: Array
    uidxpo: Array
    isnonauto: bool
    coordinates: str
    is_l2_norm: bool = False


def cal_rhos(state: Array, scale: Array) -> Array:
    """Convert interleaved Cartesian reduced coordinates to scaled radii.

    Ports ``@SSM/private/cal_rhos.m``.

    Differentiability
    -----------------
    Differentiable except at zero radius.
    """

    state = jnp.asarray(state)
    re = state[0::2]
    im = state[1::2]
    return jnp.asarray(scale).reshape((-1,)) * jnp.sqrt(re**2 + im**2)


def monitor_state_names(is_polar: bool, m: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return state-monitor names used by MATLAB continuation setup.

    Ports the naming part of ``monitor_states.m`` and
    ``monitor_scaled_states.m``. COCO problem mutation is intentionally outside
    this helper.

    Differentiability
    -----------------
    Not differentiable. This is string metadata.
    """

    if is_polar:
        return tuple(f"rho{k}" for k in range(1, m + 1)), tuple(f"th{k}" for k in range(1, m + 1))
    return tuple(f"Rez{k}" for k in range(1, m + 1)), tuple(f"Imz{k}" for k in range(1, m + 1))


def scale_parameters(values: Array, scales: Array) -> Array:
    """Scale continuation parameters elementwise.

    Ports the nested ``scale_pars`` helper in ``monitor_scaled_states.m``.

    Differentiability
    -----------------
    Differentiable.
    """

    return jnp.asarray(scales).reshape((-1,)) * jnp.asarray(values).reshape((-1,))


def initial_fixed_point_guess(
    p0: Array,
    *,
    is_polar: bool,
    m: int,
    provided: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    """Construct the initial fixed-point guess before optional solvers.

    Ports the deterministic setup and polar regularization part of
    ``initial_fixed_point.m``. MATLAB's optional `fsolve` and forward `ode45`
    refinement are not included in this JAX core helper.

    Differentiability
    -----------------
    Piecewise differentiable. Polar regularization branches on signs and wraps
    phases modulo ``2*pi``.
    """

    if provided is not None:
        p0 = jnp.asarray(provided[0])
        z0 = jnp.asarray(provided[1]).reshape((-1,))
    elif is_polar:
        z0 = 0.1 * jnp.ones((2 * m,), dtype=jnp.asarray(p0).dtype)
    else:
        z0 = jnp.zeros((2 * m,), dtype=jnp.asarray(p0).dtype)
    if is_polar:
        radii = z0[0::2]
        phases = jnp.mod(z0[1::2], 2 * jnp.pi)
        negative = radii < 0
        radii = jnp.where(negative, -radii, radii)
        phases = jnp.where(negative, phases + jnp.pi, phases)
        z0 = z0.at[0::2].set(radii)
        z0 = z0.at[1::2].set(jnp.mod(phases, 2 * jnp.pi))
    return jnp.asarray(p0), z0


def check_spectrum_and_internal_resonance(lambda_real: Array, lambda_imag: Array, modal_frequencies: Array) -> bool:
    """Validate conjugate-pair spectrum and requested internal resonance.

    Ports ``check_spectrum_and_internal_resonance.m``.

    Differentiability
    -----------------
    Not differentiable. This thresholds spectral metadata and raises
    ``ValueError`` on failure.
    """

    lambda_real = jnp.asarray(lambda_real)
    lambda_imag = jnp.asarray(lambda_imag)
    modal_frequencies = jnp.asarray(modal_frequencies)
    flags1 = jnp.all(jnp.abs(lambda_real[0::2] - lambda_real[1::2]) < 1e-6 * jnp.abs(lambda_real[0::2]))
    flags2 = jnp.all(jnp.abs(lambda_imag[0::2] + lambda_imag[1::2]) < 1e-6 * jnp.abs(lambda_imag[0::2]))
    freqs = lambda_imag[0::2]
    freqso = freqs - jnp.dot(freqs, modal_frequencies.reshape((-1,))) * modal_frequencies.reshape((-1,)) / jnp.sum(modal_frequencies**2)
    flags3 = jnp.linalg.norm(freqso) < 0.1 * jnp.linalg.norm(freqs)
    if not bool(flags1):
        raise ValueError("Real parts do not follow complex conjugate relation")
    if not bool(flags2):
        raise ValueError("Imaginary parts do not follow complex conjugate relation")
    if not bool(flags3):
        raise ValueError("Internal resonance is not detected for given master subspace")
    return True


def check_auto_reduced_dynamics(
    reduced_dynamics: tuple[MultiIndexPolynomial, ...],
    order: int,
    modal_frequencies: Array,
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    """Extract and validate autonomous reduced-dynamics resonant terms.

    Ports ``check_auto_reduced_dynamics.m`` using zero-based Python storage.

    Differentiability
    -----------------
    Not differentiable as a validation/extraction routine. It thresholds exact
    integer resonance conditions and returns discrete exponent rows.
    """

    modal_frequencies = jnp.asarray(modal_frequencies)
    m = modal_frequencies.shape[0]
    beta = [[] for _ in range(m)]
    kappa = [[] for _ in range(m)]
    eye = jnp.eye(m, dtype=jnp.int32)
    for degree in range(2, order + 1):
        poly = reduced_dynamics[degree - 1]
        coeffs = jnp.asarray(poly.coeffs)
        ind = jnp.asarray(poly.ind)
        if coeffs.size == 0:
            continue
        for mode in range(m):
            row = 2 * mode
            nonzero = jnp.nonzero(coeffs[row, :] != 0, size=None)[0]
            if nonzero.shape[0] == 0:
                continue
            kappai = ind[nonzero, :]
            left = kappai[:, 0::2]
            right = kappai[:, 1::2]
            flags = jnp.sum((left - right - eye[mode, :]) * modal_frequencies.reshape((1, -1)), axis=1)
            if not bool(jnp.all(flags == 0)):
                raise ValueError("Reduced dynamics is not consistent with desired internal resonances")
            beta[mode].append(coeffs[row, nonzero])
            kappa[mode].append(kappai)
    beta_out = tuple(jnp.concatenate(items) if items else jnp.zeros((0,)) for items in beta)
    kappa_out = tuple(jnp.concatenate(items, axis=0) if items else jnp.zeros((0, 2 * m), dtype=jnp.int32) for items in kappa)
    return beta_out, kappa_out


def create_reduced_dynamics_data(
    beta: tuple[Array, ...],
    kappa: tuple[Array, ...],
    lambda_real: Array,
    lambda_imag: Array,
    modal_frequencies: Array,
    nonauto_indices: tuple[int, ...],
    nonauto_coefficients: Array,
    nonauto_kappas: Array,
    order: int,
    resonant_modes: Array,
) -> ReducedDynamicsData:
    """Create a reduced-dynamics data bundle.

    Ports the pure data-assembly part of ``create_reduced_dynamics_data.m``.
    MATLAB's side effect of saving SSM coefficients to disk is not performed.

    Differentiability
    -----------------
    Not differentiable as a metadata/container constructor.
    """

    return ReducedDynamicsData(
        beta=tuple(beta),
        kappa=tuple(kappa),
        lambda_real=jnp.asarray(lambda_real)[0::2],
        lambda_imag=jnp.asarray(lambda_imag)[0::2],
        modal_frequencies=jnp.asarray(modal_frequencies),
        nonauto_indices=tuple(int(i) for i in nonauto_indices),
        nonauto_coefficients=jnp.asarray(nonauto_coefficients),
        nonauto_kappas=jnp.asarray(nonauto_kappas),
        order=int(order),
        modes=jnp.asarray(resonant_modes),
    )


def reduced_data_to_2md(data: ReducedDynamicsData, *, is_base_force: bool = False) -> ReducedDynamics2mDData:
    """Convert :class:`ReducedDynamicsData` to the FRC 2mD ODE data container.

    Differentiability
    -----------------
    Not differentiable as a container conversion.
    """

    return ReducedDynamics2mDData(
        beta=data.beta,
        kappa=data.kappa,
        lambda_real=data.lambda_real,
        lambda_imag=data.lambda_imag,
        modal_frequencies=data.modal_frequencies,
        nonauto_indices=data.nonauto_indices,
        nonauto_coefficients=data.nonauto_coefficients,
        is_base_force=is_base_force,
    )


def create_po_amplitude_data(
    cind: Array,
    dind: Array,
    modal_frequencies: Array,
    wcoeffs: Array,
    optdof: Array,
    *,
    dbc_weight: Array | None = None,
    is_nonauto: bool = False,
    coordinates: str = "polar",
    is_l2_norm: bool = False,
) -> POAmplitudeData:
    """Create data used by periodic-orbit amplitude objectives.

    Ports the pure data assembly in ``create_data_for_po_amp.m``. COCO-specific
    parameter-index mutation and object access are represented explicitly.

    Differentiability
    -----------------
    Not differentiable as a container constructor.
    """

    cind = jnp.asarray(cind)
    dind = jnp.asarray(dind)
    modal_frequencies = jnp.asarray(modal_frequencies)
    rhat = (cind - dind) @ modal_frequencies.reshape((-1,))
    values = []
    positions = []
    for item in rhat.tolist():
        if item not in values:
            values.append(item)
    for value in values:
        positions.append(jnp.asarray([idx for idx, item in enumerate(rhat.tolist()) if item == value], dtype=jnp.int32))
    q = jnp.eye(jnp.asarray(optdof).size) if dbc_weight is None else jnp.asarray(dbc_weight)
    qbar = 0.5 * (q + q.T)
    m = modal_frequencies.shape[0]
    uidxpo = jnp.arange(1, 2 * m + 1, dtype=jnp.int32)
    if is_nonauto:
        uidxpo = jnp.concatenate((uidxpo, jnp.asarray([2 * m + 1, 2 * m + 2], dtype=jnp.int32)))
    elif not is_l2_norm:
        uidxpo = jnp.concatenate((uidxpo, jnp.asarray([2 * m + 1], dtype=jnp.int32)))
    return POAmplitudeData(
        hatr_values=jnp.asarray(values),
        hatr_positions=tuple(positions),
        cind=cind,
        dind=dind,
        coeffs=jnp.asarray(wcoeffs),
        optdof=jnp.asarray(optdof),
        q=q,
        qbar=qbar,
        uidxpo=uidxpo,
        isnonauto=is_nonauto,
        coordinates=coordinates,
        is_l2_norm=is_l2_norm,
    )


def auto_ode_2md_ssm_cartesian(z: Array, data: ReducedDynamicsData) -> Array:
    """Evaluate autonomous 2mD Cartesian SSM reduced dynamics.

    Ports ``@SSM/private/auto_ode_2mDSSM_cartesian.m``.

    Differentiability
    -----------------
    Differentiable for fixed polynomial term structure. Tested with `jax.jit`.
    """

    z = jnp.asarray(z)

    def one(column: Array) -> Array:
        z_re = column[0::2]
        z_im = column[1::2]
        q = z_re + 1j * z_im
        q_conj = jnp.conj(q)
        y_re = data.lambda_real * z_re - data.lambda_imag * z_im
        y_im = data.lambda_real * z_im + data.lambda_imag * z_re
        for mode in range(len(data.lambda_real)):
            kappai = jnp.asarray(data.kappa[mode])
            betai = jnp.asarray(data.beta[mode])
            for term in range(betai.shape[0]):
                exponents = kappai[term, :]
                left = exponents[0::2]
                right = exponents[1::2]
                value = betai[term] * jnp.prod(q**left * q_conj**right)
                y_re = y_re.at[mode].add(jnp.real(value))
                y_im = y_im.at[mode].add(jnp.imag(value))
        out = jnp.zeros_like(column)
        out = out.at[0::2].set(y_re)
        out = out.at[1::2].set(y_im)
        return out

    if z.ndim == 1:
        return one(z)
    return jnp.stack([one(z[:, idx]) for idx in range(z.shape[1])], axis=1)


def detect_resonant_modes(resonant_lambda: Array, spectrum: Array, tolerance: float) -> tuple[Array, Array]:
    """Detect modes internally resonant with a selected master eigenvalue.

    Ports ``@SSM/private/detect_resonant_modes.m`` using zero-based Python
    indices.

    Differentiability
    -----------------
    Not differentiable. It rounds frequency ratios and thresholds closeness to
    integers.
    """

    resonant_frequency = jnp.abs(jnp.imag(resonant_lambda))
    frequencies = jnp.abs(jnp.imag(jnp.asarray(spectrum)))
    ratio = jnp.where(frequencies > resonant_frequency, frequencies / resonant_frequency, resonant_frequency / frequencies)
    m_freqs = jnp.where(frequencies > resonant_frequency, jnp.round(ratio), 1.0 / jnp.round(ratio))
    ratio = jnp.where(ratio > 5.0, jnp.inf, ratio)
    flag = jnp.abs(ratio - jnp.round(ratio)) < tolerance
    modes = jnp.nonzero(flag, size=None)[0].astype(jnp.int32)
    return modes, m_freqs[flag]
