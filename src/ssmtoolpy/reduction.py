"""Reduced-coordinate to full-coordinate polynomial maps."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from ssmtoolpy.multi_index import MultiIndexPolynomial, expand_multiindex


Array = jnp.ndarray


class NonAutonomousTerm(NamedTuple):
    """Non-autonomous contribution for ``reduced_to_full``.

    Differentiability
    -----------------
    Not differentiable as a container. Evaluation is differentiable with respect
    to polynomial coefficients and reduced coordinates.
    """

    kappa: int
    terms: tuple[MultiIndexPolynomial, ...]


def reduced_to_full(
    points: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    nonautonomous: tuple[NonAutonomousTerm, ...] = (),
    epsilon: float | Array = 0.0,
) -> Array:
    """Map reduced coordinates to real full coordinates.

    This ports the low-level behavior of ``reduced_to_full.m`` for tuple-based
    Python containers.

    Differentiability
    -----------------
    Differentiable with respect to ``points`` and coefficients for fixed
    non-autonomous structure and fixed ``epsilon``.
    """

    points = jnp.asarray(points)
    output_dim = autonomous[0].coeffs.shape[0]
    z = jnp.zeros((output_dim, points.shape[1]), dtype=jnp.result_type(points, autonomous[0].coeffs))
    if nonautonomous and not bool(jnp.asarray(epsilon == 0)):
        phi = jnp.linspace(0.0, 2.0 * jnp.pi, points.shape[1])
        for item in nonautonomous:
            zeroth = item.terms[0]
            z = z + epsilon * jnp.real(zeroth.coeffs @ jnp.exp(1j * item.kappa * phi)[None, :])
            for poly in item.terms[1:]:
                z = z + epsilon * jnp.real(expand_multiindex(poly, points) * jnp.exp(1j * item.kappa * phi))
    for poly in autonomous:
        z = z + jnp.real(expand_multiindex(poly, points))
    return z


def reduced_to_full_complex(
    points: Array,
    autonomous: tuple[MultiIndexPolynomial, ...],
    forcing: MultiIndexPolynomial | None = None,
    kappas: Array | None = None,
    epsilon: float | Array = 0.0,
) -> Array:
    """Complex-valued variant of ``reduced_to_full``.

    Differentiability
    -----------------
    Differentiable with respect to ``points`` and coefficients for fixed
    forcing structure and fixed ``epsilon``.
    """

    points = jnp.asarray(points)
    output_dim = autonomous[0].coeffs.shape[0]
    z = jnp.zeros((output_dim, points.shape[1]), dtype=jnp.result_type(points, autonomous[0].coeffs))
    if forcing is not None and kappas is not None and not bool(jnp.asarray(epsilon == 0)):
        phi = jnp.linspace(0.0, 2.0 * jnp.pi, points.shape[1])
        z = z + epsilon * (forcing.coeffs @ jnp.exp(1j * jnp.asarray(kappas)[:, None] * phi))
    for poly in autonomous:
        z = z + expand_multiindex(poly, points)
    return z
