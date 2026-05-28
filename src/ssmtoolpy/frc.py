"""Forced-response curve helper kernels."""

from __future__ import annotations

import jax.numpy as jnp


Array = jnp.ndarray


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

