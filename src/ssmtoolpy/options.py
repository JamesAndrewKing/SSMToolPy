"""Configuration containers ported from SSMTool option classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _canonical(value: str, mapping: dict[str, str], field_name: str) -> str:
    key = value.lower()
    if key not in mapping:
        allowed = ", ".join(sorted(mapping.values()))
        raise ValueError(f"Unknown {field_name}: expected one of {allowed}")
    return mapping[key]


def _require_bool(value: bool, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


@dataclass
class DSOptions:
    """Options for dynamical-system setup.

    Ports defaults and validation from ``SSMTool/src/DSOptions.m``.

    Differentiability
    -----------------
    Not differentiable. This is a configuration container.
    """

    notation: str = "tensor"
    choose_complex_comp: bool = False
    ds_type: str = "real"
    n_max: int = 100
    e_max: int = 10
    out_dof: Any = None
    rayleigh_damping: bool = True
    harmonic_force: bool = True
    lambda_threshold: float = 1e16
    base_excitation: bool = False
    sigma: float = 0.0
    remove_zeros: bool = True
    intrusion: str = "full"

    def __post_init__(self) -> None:
        self.notation = _canonical(self.notation, {"tensor": "tensor", "multiindex": "multiindex"}, "notation")
        self.choose_complex_comp = _require_bool(self.choose_complex_comp, "choose_complex_comp")
        self.ds_type = _canonical(self.ds_type, {"real": "real", "complex": "complex"}, "ds_type")
        self.rayleigh_damping = _require_bool(self.rayleigh_damping, "rayleigh_damping")
        self.harmonic_force = _require_bool(self.harmonic_force, "harmonic_force")
        self.base_excitation = _require_bool(self.base_excitation, "base_excitation")
        self.remove_zeros = _require_bool(self.remove_zeros, "remove_zeros")
        self.intrusion = _canonical(self.intrusion, {"full": "full", "semi": "semi", "none": "none"}, "intrusion")


@dataclass
class ManifoldOptions:
    """Options for manifold computation.

    Ports defaults and validation from ``SSMTool/src/ManifoldOptions.m``.

    Differentiability
    -----------------
    Not differentiable. This is a configuration container.
    """

    notation: str = "tensor"
    param_style: str = "normalform"
    reltol: float = 0.5
    ir_tol: float = 0.05
    comp_type: str = "first"
    contrib_non_auto: bool = True
    solver: str = "lsqminnorm"

    def __post_init__(self) -> None:
        self.notation = _canonical(self.notation, {"tensor": "tensor", "multiindex": "multiindex"}, "notation")
        self.param_style = _canonical(self.param_style, {"normalform": "normalform", "graph": "graph"}, "param_style")
        self.comp_type = _canonical(self.comp_type, {"first": "first", "second": "second"}, "comp_type")
        self.contrib_non_auto = _require_bool(self.contrib_non_auto, "contrib_non_auto")


@dataclass
class FRCOptions:
    """Options for forced-response-curve computation.

    Ports defaults and string canonicalization from ``SSMTool/src/FRCOptions.m``.

    Differentiability
    -----------------
    Not differentiable. This is a configuration container.
    """

    n_rho: int = 100
    n_par: int = 100
    n_psi: int = 100
    nt: int = 128
    rho_scale: float = 1.0
    n_cycle: int = 200
    omega_samp_style: str = "uniform"
    initial_solver: str = "forward"
    coordinates: str = "polar"
    samp_style: str = "cocoBD"
    method: str = "level set"
    save_ic: bool = True
    init: Any = None
    frac: tuple[float, float] = (1.0, 1.0)
    out_dof: Any = None
    res_type: str = "1:1"
    periods_ratio: int = 1
    branch_switch: bool = False
    p0: Any = None
    z0: Any = None
    non_auto_par_red_com: bool = False
    om_dep_non_auto: bool = True
    om_dep_non_auto_val: Any = None
    par_samps: Any = None
    tor_rot_diret: str = "pos"
    tor_num_segs: int = 10
    tor_purtb: float = 1e-4
    dbc_obj_norm: str = "l2"
    dbc_obj_weight: Any = None
    dbc_step_factor: tuple[float, float] = (10.0, 10.0)
    pt_mxbc_run: Any = None

    def __post_init__(self) -> None:
        self.omega_samp_style = _canonical(
            self.omega_samp_style,
            {"uniform": "uniform", "cocoout": "cocoOut", "cocobd": "cocoBD"},
            "omega_samp_style",
        )
        self.initial_solver = _canonical(self.initial_solver, {"forward": "forward", "fsolve": "fsolve"}, "initial_solver")
        self.coordinates = _canonical(self.coordinates, {"polar": "polar", "cartesian": "cartesian"}, "coordinates")
        self.samp_style = _canonical(
            self.samp_style,
            {"uniform": "uniform", "cocoout": "cocoOut", "cocobd": "cocoBD"},
            "samp_style",
        )
        self.method = _canonical(
            self.method,
            {"level set": "level set", "continuation ep": "continuation ep", "continuation po": "continuation po"},
            "method",
        )
        self.save_ic = _require_bool(self.save_ic, "save_ic")
        self.branch_switch = _require_bool(self.branch_switch, "branch_switch")
        self.non_auto_par_red_com = _require_bool(self.non_auto_par_red_com, "non_auto_par_red_com")
        self.om_dep_non_auto = _require_bool(self.om_dep_non_auto, "om_dep_non_auto")
        self.tor_rot_diret = _canonical(self.tor_rot_diret, {"pos": "pos", "neg": "neg"}, "tor_rot_diret")
        self.dbc_obj_norm = _canonical(self.dbc_obj_norm, {"l2": "l2", "linf": "linf"}, "dbc_obj_norm")


@dataclass
class FRSOptions:
    """Options for forced-response-surface computation.

    Ports defaults and validation from ``SSMTool/src/FRSOptions.m``.

    Differentiability
    -----------------
    Not differentiable. This is a configuration container.
    """

    rho_max: float = 1.0
    mesh_dens: int = 100
    method: str = "level set"
    cal_frs: bool = False

    def __post_init__(self) -> None:
        self.method = _canonical(self.method, {"level set": "level set", "continuation": "continuation"}, "method")
        self.cal_frs = _require_bool(self.cal_frs, "cal_frs")


def matlab_style_options(options: DSOptions | ManifoldOptions | FRCOptions | FRSOptions) -> dict[str, Any]:
    """Return option data using MATLAB-inspired field names.

    This helps bridge future class-style ports that still refer to MATLAB field
    names in the migration notes.

    Differentiability
    -----------------
    Not differentiable. This is metadata conversion.
    """

    if isinstance(options, DSOptions):
        return {
            "notation": options.notation,
            "ChooseComplexComp": options.choose_complex_comp,
            "DStype": options.ds_type,
            "Nmax": options.n_max,
            "Emax": options.e_max,
            "outDOF": options.out_dof,
            "RayleighDamping": options.rayleigh_damping,
            "HarmonicForce": options.harmonic_force,
            "lambdaThreshold": options.lambda_threshold,
            "BaseExcitation": options.base_excitation,
            "sigma": options.sigma,
            "RemoveZeros": options.remove_zeros,
            "Intrusion": options.intrusion,
        }
    if isinstance(options, ManifoldOptions):
        return {
            "notation": options.notation,
            "paramStyle": options.param_style,
            "reltol": options.reltol,
            "IRtol": options.ir_tol,
            "COMPtype": options.comp_type,
            "contribNonAuto": options.contrib_non_auto,
            "solver": options.solver,
        }
    if isinstance(options, FRCOptions):
        return {
            "nRho": options.n_rho,
            "nPar": options.n_par,
            "nPsi": options.n_psi,
            "nt": options.nt,
            "rhoScale": options.rho_scale,
            "nCycle": options.n_cycle,
            "omegaSampStyle": options.omega_samp_style,
            "initialSolver": options.initial_solver,
            "coordinates": options.coordinates,
            "sampStyle": options.samp_style,
            "method": options.method,
            "saveIC": options.save_ic,
            "init": options.init,
            "frac": options.frac,
            "outdof": options.out_dof,
            "resType": options.res_type,
            "periodsRatio": options.periods_ratio,
            "branchSwitch": options.branch_switch,
            "p0": options.p0,
            "z0": options.z0,
            "nonAutoParRedCom": options.non_auto_par_red_com,
            "omDepNonAuto": options.om_dep_non_auto,
            "omDepNonAutoVal": options.om_dep_non_auto_val,
            "parSamps": options.par_samps,
            "torRotDiret": options.tor_rot_diret,
            "torNumSegs": options.tor_num_segs,
            "torPurtb": options.tor_purtb,
            "DBCobjnorm": options.dbc_obj_norm,
            "DBCobjweight": options.dbc_obj_weight,
            "DBCstepFactor": options.dbc_step_factor,
            "PtMXBCrun": options.pt_mxbc_run,
        }
    return {
        "rhoMax": options.rho_max,
        "meshDens": options.mesh_dens,
        "method": options.method,
        "calFRS": options.cal_frs,
    }
