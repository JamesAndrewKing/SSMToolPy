"""Trajectory assembly helpers used by example workflows."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp


Array = jnp.ndarray


def integrate_two_sided_branches(
    trajectory_function: Callable[[Array], Array],
    initial_state: Array,
) -> Array:
    """Integrate positive and negative branches and concatenate them.

    ``trajectory_function`` receives one initial state and returns an array with
    time along the first axis. The negative branch is reversed before
    concatenation, matching SSMTool plotting workflows such as
    ``Lorenz1stOrder/demo.mlx``.

    Differentiability: differentiable with respect to ``initial_state`` and any
    continuous closed-over parameters for fixed trajectory function structure.
    """

    initial_state = jnp.asarray(initial_state)
    positive = trajectory_function(initial_state)
    negative = trajectory_function(-initial_state)
    return jnp.concatenate([negative[::-1], positive], axis=0)
