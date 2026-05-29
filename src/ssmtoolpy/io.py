"""MATLAB/COCO solution readers for SSMToolPy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat


Array = jnp.ndarray


class SSMContinuationInfo(NamedTuple):
    """Metadata attached to SSM continuation solution files.

    Differentiability
    -----------------
    Not differentiable. This is file metadata loaded from MATLAB output.
    """

    order: Any
    nonauto: Any
    ispolar: Any


class NumericalIntegrationSummary(NamedTuple):
    """Summary of numerical integration periodic-orbit files.

    Differentiability
    -----------------
    Not differentiable. This is disk I/O and norm post-processing.
    """

    amplitudes: Array
    norms: Array
    omegas: Array


class PeriodicOrbitInitialConditions(NamedTuple):
    """Initial-condition columns extracted from ``po_ssm*.mat`` files.

    Differentiability
    -----------------
    Not differentiable. This is disk I/O and array concatenation.
    """

    z_init: Array
    omega: Array


def coco_fname(runid: str | tuple[str, ...] | list[str], filename: str, *, root: str | Path = ".") -> Path:
    """Resolve a COCO output filename using MATLAB ``coco_fname`` rules.

    The direct ``runid/filename`` path is tried first, followed by
    ``data/runid/filename``.

    Differentiability
    -----------------
    Not differentiable. This resolves filesystem paths.
    """

    root_path = Path(root)
    run_path = Path(*runid) if isinstance(runid, (tuple, list)) else Path(runid)
    direct = root_path / run_path / filename
    if direct.exists():
        return direct
    data_path = root_path / "data" / run_path / filename
    if data_path.exists():
        return data_path
    raise FileNotFoundError(f"could not find {filename!r} in {run_path!s} or data/{run_path!s}")


def _load_frc(runid: str, suffix: str, filename: str, *, root: str | Path = ".") -> dict[str, Any]:
    path = coco_fname(f"{runid}.{suffix}", filename, root=root)
    loaded = loadmat(path, simplify_cells=True)
    frc = loaded.get("FRC")
    if frc is None:
        raise KeyError(f"{path} does not contain an FRC variable")
    if not isinstance(frc, dict):
        raise TypeError("FRC variable must load as a MATLAB struct/dict")
    return frc


def _info_from_frc(frc: dict[str, Any]) -> SSMContinuationInfo:
    return SSMContinuationInfo(
        frc.get("SSMorder"),
        bool(frc.get("SSMnonAuto")) if "SSMnonAuto" in frc else None,
        bool(frc.get("SSMispolar")) if "SSMispolar" in frc else None,
    )


def _column(array: Any, idx: int) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 1:
        return values
    return values[:, idx]


def read_ssm_ep_solution(runid: str, *, root: str | Path = ".") -> dict[str, Any]:
    """Read equilibrium-point SSM continuation output.

    Ports ``misc/SSM_ep_read_solution.m``. The returned dictionary is the
    MATLAB ``FRC`` struct loaded from ``SSMep.mat``.

    Differentiability
    -----------------
    Not differentiable. This is MATLAB file I/O.
    """

    return _load_frc(runid, "ep", "SSMep.mat", root=root)


def read_ssm_po_solution(runid: str, label: int | None = None, *, root: str | Path = ".") -> dict[str, Any]:
    """Read periodic-orbit SSM continuation output.

    Ports the SSM-specific part of ``misc/SSM_po_read_solution.m``. COCO's
    separate ``po_read_solution`` payload is not loaded; callers can combine it
    later when a Python COCO reader exists.

    Differentiability
    -----------------
    Not differentiable. This is MATLAB file I/O and label selection.
    """

    frc = _load_frc(runid, "po", "SSMpo.mat", root=root)
    if label is None:
        sol = {key: frc.get(key) for key in ("om", "ep", "st", "lab")}
    else:
        labs = np.atleast_1d(np.asarray(frc["lab"]))
        matches = np.nonzero(labs == label)[0]
        if matches.size == 0:
            raise KeyError(f"label {label} not found")
        idx = int(matches[0])
        sol = {
            "om": np.atleast_1d(np.asarray(frc["om"]))[idx],
            "ep": np.atleast_1d(np.asarray(frc["ep"]))[idx],
            "st": np.atleast_1d(np.asarray(frc["st"]))[idx],
            "tTr": frc["tTr"][idx],
            "xTr": frc["zTr"][idx],
            "z0": _column(frc["Z0_frc"], idx),
        }
    sol["info"] = _info_from_frc(frc)
    return sol


def read_ssm_tor_solution(runid: str, label: int | None = None, *, root: str | Path = ".") -> dict[str, Any]:
    """Read torus SSM continuation output.

    Ports the SSM-specific part of ``misc/SSM_tor_read_solution.m``. COCO's
    separate ``tor_read_solution`` payload is intentionally left for a future
    COCO-reader layer.

    Differentiability
    -----------------
    Not differentiable. This is MATLAB file I/O and label selection.
    """

    frc = _load_frc(runid, "tor", "SSMtor.mat", root=root)
    if label is None:
        sol = {key: frc.get(key) for key in ("om", "ep", "lab")}
    else:
        labs = np.atleast_1d(np.asarray(frc["lab"]))
        matches = np.nonzero(labs == label)[0]
        if matches.size == 0:
            raise KeyError(f"label {label} not found")
        idx = int(matches[0])
        sol = {
            "om": np.atleast_1d(np.asarray(frc["om"]))[idx],
            "ep": np.atleast_1d(np.asarray(frc["ep"]))[idx],
            "tTr": frc["tTr"][idx],
            "xTr": frc["zTr"][idx],
            "z0": _column(frc["Z0_frc"], idx),
        }
    sol["info"] = _info_from_frc(frc)
    return sol


def read_num_int_sol(labels: tuple[int, ...] | list[int] | Array, outdof: Array | None = None, *, root: str | Path = ".") -> NumericalIntegrationSummary:
    """Read numerical-integration periodic-orbit summaries.

    Ports ``misc/read_num_int_sol.m``. Files are named ``po<label>.mat`` and
    are expected to contain ``x0``, ``Omega`` and ``t0``.

    Differentiability
    -----------------
    Not differentiable. This is MATLAB file I/O and discrete label indexing.
    """

    labels_array = [int(value) for value in np.asarray(labels).ravel()]
    n = max(labels_array) + 1 if labels_array else 0
    omegas = np.full((n,), np.nan)
    norms = np.zeros((n,))
    amplitudes = np.zeros((n,))
    root_path = Path(root)
    for label in labels_array:
        loaded = loadmat(root_path / f"po{label}.mat", simplify_cells=True)
        x0 = np.asarray(loaded["x0"])
        omega = loaded["Omega"]
        t0 = np.asarray(loaded["t0"]).ravel()
        omegas[label] = omega
        norms[label] = np.linalg.norm(x0, ord="fro") / np.sqrt(t0.size - 1)
        if outdof is not None:
            amplitudes[label] = np.linalg.norm(x0[:, np.asarray(outdof, dtype=int)], ord=np.inf)
    return NumericalIntegrationSummary(jnp.asarray(amplitudes), jnp.asarray(norms), jnp.asarray(omegas))


def read_po_ssm_init(labels: tuple[int, ...] | list[int] | Array, *, root: str | Path = ".") -> PeriodicOrbitInitialConditions:
    """Read full initial conditions saved during SSM FRC extraction.

    Ports ``misc/read_po_ssm_init.m``. Files are named ``po_ssm<label>.mat`` and
    contain a cell-like ``Z`` plus ``Omega_0``.

    Differentiability
    -----------------
    Not differentiable. This is MATLAB file I/O and concatenation.
    """

    columns = []
    omegas = []
    root_path = Path(root)
    for label in [int(value) for value in np.asarray(labels).ravel()]:
        loaded = loadmat(root_path / f"po_ssm{label}.mat", simplify_cells=True)
        z_items = loaded["Z"]
        omega_items = np.asarray(loaded["Omega_0"]).ravel()
        for idx, omega in enumerate(omega_items):
            columns.append(np.asarray(z_items[idx])[:, 0])
            omegas.append(omega)
    if columns:
        z_init = np.stack(columns, axis=1)
    else:
        z_init = np.zeros((0, 0))
    return PeriodicOrbitInitialConditions(jnp.asarray(z_init), jnp.asarray(omegas))
