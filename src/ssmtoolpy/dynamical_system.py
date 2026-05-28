"""Differentiable dynamical-system evaluation kernels.

This module ports the dependency-light numerical core of selected
``@DynamicalSystem`` methods without reproducing the full mutable MATLAB class.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp

from ssmtoolpy.multi_index import MultiIndexPolynomial, expand_multiindex, expand_multiindex_derivative
from ssmtoolpy.tensor import expand_tensor, expand_tensor_derivative


Array = jnp.ndarray


class FourierForcingTerm(NamedTuple):
    """One harmonic contribution to periodic forcing.

    ``terms[0]`` is the zeroth-order coefficient, and later entries are
    coordinate-dependent multi-index polynomials.

    Differentiability
    -----------------
    Not differentiable as a container. Evaluation is differentiable for fixed
    structure.
    """

    kappa: int
    terms: tuple[MultiIndexPolynomial, ...]


class PeriodicForcing(NamedTuple):
    """Periodic forcing data used by first- and second-order evaluators.

    Differentiability
    -----------------
    Not differentiable as a container. Evaluation is differentiable with respect
    to coordinates, time, coefficient values, ``omega``, and ``epsilon`` for
    fixed harmonic structure.
    """

    terms: tuple[FourierForcingTerm, ...]
    epsilon: Array
    omega: Array
    base_excitation: bool = False


class ResidualResult(NamedTuple):
    """Second-order residual and Jacobian result.

    Differentiability
    -----------------
    Not differentiable as a container. Numeric fields carry the
    differentiability of ``second_order_residual``.
    """

    residual: Array
    drdqdd: Array
    drdqd: Array
    drdq: Array
    c0: Array


def _as_matrix(values: Array) -> Array:
    values = jnp.asarray(values)
    if values.ndim == 1:
        return values[:, None]
    return values


def evaluate_polynomial_terms(terms: tuple[MultiIndexPolynomial | Array, ...], coordinates: Array) -> Array:
    """Evaluate and sum multi-index or dense-tensor polynomial terms.

    Differentiability
    -----------------
    Differentiable with respect to coordinates and coefficient values for fixed
    term structure.
    """

    coordinates = _as_matrix(coordinates)
    if not terms:
        return jnp.zeros((0, coordinates.shape[1]), dtype=coordinates.dtype)
    first = terms[0]
    output_dim = first.coeffs.shape[0] if isinstance(first, MultiIndexPolynomial) else jnp.asarray(first).shape[0]
    result = jnp.zeros((output_dim, coordinates.shape[1]), dtype=jnp.result_type(coordinates, first.coeffs if isinstance(first, MultiIndexPolynomial) else first))
    for term in terms:
        if isinstance(term, MultiIndexPolynomial):
            result = result + expand_multiindex(term, coordinates)
        else:
            result = result + jnp.stack([expand_tensor(term, coordinates[:, idx]) for idx in range(coordinates.shape[1])], axis=1)
    return result


def evaluate_polynomial_jacobian(terms: tuple[MultiIndexPolynomial | Array, ...], coordinates: Array) -> Array:
    """Evaluate and sum Jacobians of polynomial terms at one coordinate vector.

    Differentiability
    -----------------
    Differentiable with respect to coordinates and coefficient values for fixed
    term structure.
    """

    coordinates = jnp.asarray(coordinates)
    if not terms:
        return jnp.zeros((0, coordinates.shape[0]), dtype=coordinates.dtype)
    jac = None
    for term in terms:
        if isinstance(term, MultiIndexPolynomial):
            value = expand_multiindex_derivative(term, coordinates[:, None])[:, :, 0]
        else:
            value = expand_tensor_derivative(term, coordinates)
        jac = value if jac is None else jac + value
    return jac


def first_order_nonlinearity(
    state: Array,
    terms: tuple[MultiIndexPolynomial | Array, ...],
    nonintrusive: Callable[[Array], Array] | None = None,
) -> Array:
    """Evaluate first-order nonlinear force ``Fnl(z)``.

    Differentiability
    -----------------
    Differentiable for intrusive polynomial/tensor terms. A non-intrusive
    callable is transformable only if the callable itself is JAX-transformable.
    """

    state_matrix = _as_matrix(state)
    if nonintrusive is not None:
        values = jnp.stack([jnp.asarray(nonintrusive(state_matrix[:, idx])) for idx in range(state_matrix.shape[1])], axis=1)
    else:
        values = evaluate_polynomial_terms(terms, state_matrix)
    return values[:, 0] if jnp.asarray(state).ndim == 1 else values


def second_order_internal_force(
    x: Array,
    xd: Array | None,
    terms: tuple[MultiIndexPolynomial | Array, ...],
    *,
    n: int | None = None,
    nonintrusive: Callable[[Array], Array] | None = None,
    velocity_dependent: bool = False,
) -> Array:
    """Evaluate second-order nonlinear internal force ``fnl(x, xd)``.

    Differentiability
    -----------------
    Differentiable for intrusive polynomial/tensor terms. Non-intrusive
    callable differentiability depends on the callable.
    """

    x_matrix = _as_matrix(x)
    if xd is None:
        xd_matrix = None
    else:
        xd_matrix = _as_matrix(xd)
    coordinates = jnp.concatenate([x_matrix, xd_matrix], axis=0) if velocity_dependent and xd_matrix is not None else x_matrix
    if nonintrusive is not None:
        values = jnp.stack([jnp.asarray(nonintrusive(coordinates[:, idx])) for idx in range(coordinates.shape[1])], axis=1)
    else:
        values = evaluate_polynomial_terms(terms, coordinates)
    if n is not None and values.shape[0] == 0:
        values = jnp.zeros((n, coordinates.shape[1]), dtype=coordinates.dtype)
    return values[:, 0] if jnp.asarray(x).ndim == 1 else values


def second_order_internal_force_jacobian_x(
    x: Array,
    terms: tuple[MultiIndexPolynomial | Array, ...],
    *,
    n: int | None = None,
    nonintrusive_jacobian: Callable[[Array], Array] | None = None,
) -> Array:
    """Evaluate ``d fnl / d x`` for displacement-only nonlinearities.

    Differentiability
    -----------------
    Differentiable for intrusive polynomial/tensor terms. Non-intrusive
    callable differentiability depends on the callable.
    """

    x = jnp.asarray(x)
    if nonintrusive_jacobian is not None:
        return jnp.asarray(nonintrusive_jacobian(x))
    if not terms:
        if n is None:
            n = x.shape[0]
        return jnp.zeros((n, n), dtype=x.dtype)
    return evaluate_polynomial_jacobian(terms, x)


def second_order_internal_force_jacobian_xd(n: int, dtype: Array | None = None) -> Array:
    """Return ``d fnl / d xd`` for displacement-only nonlinearities.

    Differentiability
    -----------------
    Differentiable as a constant zero map.
    """

    return jnp.zeros((n, n), dtype=jnp.asarray(0.0 if dtype is None else dtype).dtype)


def first_order_from_second_order_nonlinearity(fnl: Array, n: int | None = None) -> Array:
    """Build first-order nonlinear term ``[-fnl; 0]`` from mechanical force.

    Differentiability
    -----------------
    Differentiable.
    """

    original_ndim = jnp.asarray(fnl).ndim
    fnl = _as_matrix(fnl)
    if n is None:
        n = fnl.shape[0]
    result = jnp.concatenate([-fnl, jnp.zeros((n, fnl.shape[1]), dtype=fnl.dtype)], axis=0)
    return result[:, 0] if original_ndim == 1 else result


def evaluate_periodic_forcing(time: Array, coordinates: Array, forcing: PeriodicForcing | None) -> Array:
    """Evaluate periodic Taylor/Fourier forcing at ``time`` and coordinates.

    Differentiability
    -----------------
    Differentiable for fixed forcing structure, away from any non-smooth
    coordinate-dependent terms supplied by callers.
    """

    coordinates = _as_matrix(coordinates)
    if forcing is None or not forcing.terms:
        return jnp.zeros((0, coordinates.shape[1]), dtype=coordinates.dtype)
    first_coeffs = forcing.terms[0].terms[0].coeffs
    result = jnp.zeros((first_coeffs.shape[0], coordinates.shape[1]), dtype=jnp.result_type(first_coeffs, coordinates))
    omega = jnp.asarray(forcing.omega)
    epsilon = jnp.asarray(forcing.epsilon)
    for term in forcing.terms:
        phase = jnp.exp(1j * term.kappa * omega * time)
        zeroth = term.terms[0]
        if zeroth.coeffs.size:
            result = result + jnp.real(zeroth.coeffs @ phase[None, :] if phase.ndim else zeroth.coeffs * phase)
        for poly in term.terms[1:]:
            result = result + jnp.real(expand_multiindex(poly, coordinates) * phase)
    result = result * epsilon
    if forcing.base_excitation:
        result = result * (omega**2)
    return result


def evaluate_first_order_vector_field(
    a_matrix: Array,
    b_matrix: Array,
    state: Array,
    nonlinear_terms: tuple[MultiIndexPolynomial | Array, ...] = (),
    forcing: PeriodicForcing | None = None,
    time: Array = 0.0,
) -> Array:
    """Evaluate ``B \\ (A z + Fnl(z) + Fext(t,z))``.

    Differentiability
    -----------------
    Differentiable under nonsingularity of ``b_matrix`` and differentiability
    of the nonlinear/forcing terms.
    """

    state_matrix = _as_matrix(state)
    linear = jnp.asarray(a_matrix) @ state_matrix
    fnl = first_order_nonlinearity(state_matrix, nonlinear_terms)
    if fnl.size == 0:
        fnl = jnp.zeros_like(linear)
    fext = evaluate_periodic_forcing(time, state_matrix, forcing)
    if fext.size == 0:
        fext = jnp.zeros_like(linear)
    rhs = linear + _as_matrix(fnl) + fext
    values = jnp.linalg.solve(jnp.asarray(b_matrix), rhs)
    return values[:, 0] if jnp.asarray(state).ndim == 1 else values


def second_order_residual(
    m_matrix: Array,
    c_matrix: Array,
    k_matrix: Array,
    q: Array,
    qd: Array,
    qdd: Array,
    nonlinear_terms: tuple[MultiIndexPolynomial | Array, ...] = (),
    external_force: Array | Callable[[Array, Array, Array], Array] | None = None,
    time: Array = 0.0,
) -> ResidualResult:
    """Evaluate second-order residual and tangent matrices.

    Differentiability
    -----------------
    Differentiable under the nonlinear term assumptions. The norm-based ``c0``
    is non-smooth at zero component norms.
    """

    q = jnp.asarray(q)
    qd = jnp.asarray(qd)
    qdd = jnp.asarray(qdd)
    m_matrix = jnp.asarray(m_matrix)
    c_matrix = jnp.asarray(c_matrix)
    k_matrix = jnp.asarray(k_matrix)
    fnl = second_order_internal_force(q, qd, nonlinear_terms, n=q.shape[0])
    if external_force is None:
        fext = jnp.zeros_like(q)
    elif callable(external_force):
        fext = jnp.asarray(external_force(time, q, qd))
    else:
        fext = jnp.asarray(external_force)
    inertial = m_matrix @ qdd
    damping = c_matrix @ qd
    elastic = k_matrix @ q + fnl
    residual = inertial + damping + elastic - fext
    drdqdd = m_matrix
    drdqd = c_matrix + second_order_internal_force_jacobian_xd(q.shape[0], q)
    drdq = k_matrix + second_order_internal_force_jacobian_x(q, nonlinear_terms, n=q.shape[0])
    c0 = jnp.linalg.norm(inertial) + jnp.linalg.norm(damping) + jnp.linalg.norm(elastic) + jnp.linalg.norm(fext)
    return ResidualResult(residual, drdqdd, drdqd, drdq, c0)


def mechanical_binv_a(m_matrix: Array, c_matrix: Array, k_matrix: Array) -> Array:
    """Return the first-order mechanical matrix ``B^{-1} A``.

    Differentiability
    -----------------
    Differentiable under nonsingularity of ``m_matrix``.
    """

    m_matrix = jnp.asarray(m_matrix)
    n = m_matrix.shape[0]
    top = jnp.concatenate([jnp.zeros((n, n), dtype=m_matrix.dtype), jnp.eye(n, dtype=m_matrix.dtype)], axis=1)
    bottom = jnp.concatenate([-jnp.linalg.solve(m_matrix, k_matrix), -jnp.linalg.solve(m_matrix, c_matrix)], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def mechanical_a_matrix(m_matrix: Array, k_matrix: Array) -> Array:
    """Return the first-order mechanical ``A`` matrix from ``get_A.m``.

    ``A = [[-K, 0], [0, M]]``.

    Differentiability
    -----------------
    Differentiable with respect to ``m_matrix`` and ``k_matrix``.
    """

    m_matrix = jnp.asarray(m_matrix)
    k_matrix = jnp.asarray(k_matrix)
    n = m_matrix.shape[0]
    zeros = jnp.zeros((n, n), dtype=jnp.result_type(m_matrix, k_matrix))
    top = jnp.concatenate([-k_matrix, zeros], axis=1)
    bottom = jnp.concatenate([zeros, m_matrix], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def mechanical_b_matrix(m_matrix: Array, c_matrix: Array) -> Array:
    """Return the first-order mechanical ``B`` matrix from ``get_B.m``.

    ``B = [[C, M], [M, 0]]``.

    Differentiability
    -----------------
    Differentiable with respect to ``m_matrix`` and ``c_matrix``.
    """

    m_matrix = jnp.asarray(m_matrix)
    c_matrix = jnp.asarray(c_matrix)
    n = m_matrix.shape[0]
    zeros = jnp.zeros((n, n), dtype=jnp.result_type(m_matrix, c_matrix))
    top = jnp.concatenate([c_matrix, m_matrix], axis=1)
    bottom = jnp.concatenate([m_matrix, zeros], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def polynomial_input_dim(terms: tuple[MultiIndexPolynomial | Array | None, ...]) -> int:
    """Infer nonlinear input dimension from the first nonempty polynomial/tensor.

    This ports the intrusive parts of ``get_F_input_dim.m`` and
    ``get_fnl_input_dim.m``.

    Differentiability
    -----------------
    Not differentiable. This is shape metadata inference.
    """

    for term in terms:
        if term is None:
            continue
        if isinstance(term, MultiIndexPolynomial):
            if term.coeffs.size:
                return int(term.ind.shape[1])
        else:
            array = jnp.asarray(term)
            if array.size and bool(jnp.any(array != 0)):
                return int(array.shape[1])
    raise ValueError("Failed to set input dimension of nonlinearity")


def polynomial_degree(terms: tuple[MultiIndexPolynomial | Array | None, ...]) -> int:
    """Return the number of polynomial term slots.

    This mirrors ``get_degree.m`` for an already assembled term sequence.

    Differentiability
    -----------------
    Not differentiable. This is metadata.
    """

    return len(terms)


def first_order_polynomial_terms_from_second_order(
    fnl_terms: tuple[MultiIndexPolynomial | None, ...],
    *,
    n: int,
    total_dim: int | None = None,
    system_order: int = 2,
) -> tuple[MultiIndexPolynomial, ...]:
    """Embed second-order internal-force multi-index terms in first-order form.

    For mechanical second-order systems MATLAB uses ``[-fnl; 0]`` in the
    first-order nonlinear term. If ``system_order == 1`` the sign is positive,
    matching ``set_Ftens_from_fnlmulti.m``.

    Differentiability
    -----------------
    Differentiable with respect to coefficient values for fixed term structure.
    """

    if total_dim is None:
        total_dim = 2 * n
    sign = 1.0 if system_order == 1 else -1.0
    converted = []
    for term in fnl_terms:
        if term is None or term.coeffs.size == 0:
            converted.append(
                MultiIndexPolynomial(
                    coeffs=jnp.zeros((total_dim, 0)),
                    ind=jnp.zeros((0, n), dtype=jnp.int32),
                )
            )
            continue
        coeffs = jnp.concatenate([sign * term.coeffs, jnp.zeros((total_dim - n, term.coeffs.shape[1]), dtype=term.coeffs.dtype)], axis=0)
        converted.append(MultiIndexPolynomial(coeffs=coeffs, ind=term.ind))
    return tuple(converted)


def first_order_tensor_terms_from_second_order(
    fnl_tensors: tuple[Array | None, ...],
    *,
    n: int,
    total_dim: int | None = None,
    system_order: int = 2,
) -> tuple[Array, ...]:
    """Embed second-order dense tensor nonlinearities in first-order form.

    This is the dense-JAX counterpart of ``set_Ftens_from_fnltens.m``. Input
    tensor modes are padded to ``total_dim``; existing second-order coordinates
    occupy the leading modes.

    Differentiability
    -----------------
    Differentiable with respect to tensor values for fixed shapes.
    """

    if total_dim is None:
        total_dim = 2 * n
    sign = 1.0 if system_order == 1 else -1.0
    converted = []
    for tensor in fnl_tensors:
        if tensor is None:
            converted.append(jnp.zeros((total_dim, total_dim)))
            continue
        tensor = jnp.asarray(tensor)
        degree = tensor.ndim - 1
        out = jnp.zeros((total_dim, *([total_dim] * degree)), dtype=tensor.dtype)
        slices = (slice(0, n), *[slice(0, tensor.shape[axis]) for axis in range(1, tensor.ndim)])
        converted.append(out.at[slices].set(sign * tensor))
    return tuple(converted)


def first_order_terms_from_second_order(
    fnl_terms: tuple[MultiIndexPolynomial | Array | None, ...],
    *,
    n: int,
    total_dim: int | None = None,
    system_order: int = 2,
) -> tuple[MultiIndexPolynomial | Array, ...]:
    """Embed second-order nonlinear terms, preserving multi-index or tensor form.

    Differentiability
    -----------------
    Differentiable with respect to coefficient/tensor values for fixed term
    structure.
    """

    if all(term is None or isinstance(term, MultiIndexPolynomial) for term in fnl_terms):
        return first_order_polynomial_terms_from_second_order(fnl_terms, n=n, total_dim=total_dim, system_order=system_order)
    return first_order_tensor_terms_from_second_order(fnl_terms, n=n, total_dim=total_dim, system_order=system_order)
