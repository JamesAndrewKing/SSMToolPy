import pytest

from ssmtoolpy import DSOptions, FRCOptions, FRSOptions, ManifoldOptions, matlab_style_options


def test_ds_options_defaults_and_canonicalization():
    opts = DSOptions(notation="MULTIINDEX", ds_type="COMPLEX", intrusion="NONE")
    assert opts.notation == "multiindex"
    assert opts.ds_type == "complex"
    assert opts.intrusion == "none"
    assert opts.n_max == 100
    assert opts.e_max == 10
    assert opts.lambda_threshold == 1e16
    fields = matlab_style_options(opts)
    assert fields["ChooseComplexComp"] is False
    assert fields["DStype"] == "complex"
    assert fields["Intrusion"] == "none"
    assert fields["BaseExcitation"] is False


def test_manifold_options_defaults_validation_and_matlab_names():
    opts = ManifoldOptions(notation="tensor", param_style="GRAPH", comp_type="SECOND", contrib_non_auto=False)
    assert opts.notation == "tensor"
    assert opts.param_style == "graph"
    assert opts.comp_type == "second"
    assert opts.contrib_non_auto is False
    fields = matlab_style_options(opts)
    assert fields["paramStyle"] == "graph"
    assert fields["IRtol"] == 0.05
    assert fields["COMPtype"] == "second"
    assert fields["solver"] == "lsqminnorm"
    with pytest.raises(TypeError):
        ManifoldOptions(contrib_non_auto=1)


def test_frc_options_defaults_and_string_canonicalization():
    opts = FRCOptions(
        omega_samp_style="COCOBD",
        initial_solver="FSOLVE",
        coordinates="CARTESIAN",
        samp_style="COCOOUT",
        method="CONTINUATION PO",
        tor_rot_diret="NEG",
        dbc_obj_norm="LINF",
    )
    assert opts.omega_samp_style == "cocoBD"
    assert opts.initial_solver == "fsolve"
    assert opts.coordinates == "cartesian"
    assert opts.samp_style == "cocoOut"
    assert opts.method == "continuation po"
    assert opts.tor_rot_diret == "neg"
    assert opts.dbc_obj_norm == "linf"
    fields = matlab_style_options(opts)
    assert fields["nRho"] == 100
    assert fields["nt"] == 128
    assert fields["rhoScale"] == 1.0
    assert fields["saveIC"] is True
    assert fields["DBCstepFactor"] == (10.0, 10.0)
    with pytest.raises(ValueError):
        FRCOptions(method="shooting")


def test_frs_options_defaults_and_validation():
    opts = FRSOptions(method="CONTINUATION", cal_frs=True)
    assert opts.rho_max == 1.0
    assert opts.mesh_dens == 100
    assert opts.method == "continuation"
    assert opts.cal_frs is True
    fields = matlab_style_options(opts)
    assert fields == {"rhoMax": 1.0, "meshDens": 100, "method": "continuation", "calFRS": True}
    with pytest.raises(ValueError):
        FRSOptions(method="grid")
