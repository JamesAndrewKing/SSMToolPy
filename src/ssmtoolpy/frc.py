"""Forced-response curve helper kernels."""

from __future__ import annotations

import jax.numpy as jnp

from ssmtoolpy.multi_index import MultiIndexPolynomial


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
