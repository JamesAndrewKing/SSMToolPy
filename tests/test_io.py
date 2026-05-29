import jax.numpy as jnp
import numpy as np
from scipy.io import savemat

from ssmtoolpy import (
    SSMContinuationInfo,
    coco_fname,
    read_num_int_sol,
    read_po_ssm_init,
    read_ssm_ep_solution,
    read_ssm_po_solution,
    read_ssm_tor_solution,
)


def test_coco_fname_resolves_direct_and_data_paths(tmp_path):
    direct = tmp_path / "run.ep"
    direct.mkdir()
    (direct / "SSMep.mat").write_text("x")
    assert coco_fname("run.ep", "SSMep.mat", root=tmp_path) == direct / "SSMep.mat"

    data = tmp_path / "data" / "other.ep"
    data.mkdir(parents=True)
    (data / "SSMep.mat").write_text("x")
    assert coco_fname("other.ep", "SSMep.mat", root=tmp_path) == data / "SSMep.mat"


def test_read_ssm_ep_po_and_tor_solutions(tmp_path):
    ep_dir = tmp_path / "run.ep"
    ep_dir.mkdir()
    savemat(ep_dir / "SSMep.mat", {"FRC": {"om": np.array([1.0, 2.0]), "SSMorder": 3}})
    ep = read_ssm_ep_solution("run", root=tmp_path)
    np.testing.assert_allclose(ep["om"], np.array([1.0, 2.0]))

    po_dir = tmp_path / "run.po"
    po_dir.mkdir()
    savemat(
        po_dir / "SSMpo.mat",
        {
            "FRC": {
                "om": np.array([1.0, 2.0]),
                "ep": np.array([0.1, 0.2]),
                "st": np.array([1, 0]),
                "lab": np.array([5, 6]),
                "tTr": np.array([np.array([0.0, 1.0]), np.array([0.0, 2.0])], dtype=object),
                "zTr": np.array([np.eye(2), 2 * np.eye(2)], dtype=object),
                "Z0_frc": np.array([[10.0, 20.0], [30.0, 40.0]]),
                "SSMorder": 4,
                "SSMnonAuto": True,
                "SSMispolar": False,
            }
        },
    )
    po_all = read_ssm_po_solution("run", root=tmp_path)
    np.testing.assert_array_equal(po_all["lab"], np.array([5, 6]))
    po = read_ssm_po_solution("run", label=6, root=tmp_path)
    assert isinstance(po["info"], SSMContinuationInfo)
    assert po["info"].order == 4
    np.testing.assert_allclose(po["z0"], np.array([20.0, 40.0]))

    tor_dir = tmp_path / "run.tor"
    tor_dir.mkdir()
    savemat(
        tor_dir / "SSMtor.mat",
        {
            "FRC": {
                "om": np.array([3.0]),
                "ep": np.array([0.3]),
                "lab": np.array([9]),
                "tTr": np.array([np.array([0.0, 1.0])], dtype=object),
                "zTr": np.array([np.eye(2)], dtype=object),
                "Z0_frc": np.array([[7.0], [8.0]]),
                "SSMorder": 2,
                "SSMnonAuto": False,
                "SSMispolar": True,
            }
        },
    )
    tor = read_ssm_tor_solution("run", label=9, root=tmp_path)
    np.testing.assert_allclose(tor["z0"], np.array([7.0, 8.0]))
    assert tor["info"].ispolar is True


def test_read_num_int_sol_and_po_ssm_init(tmp_path):
    savemat(tmp_path / "po1.mat", {"x0": np.array([[1.0, -2.0], [3.0, 4.0]]), "Omega": 2.5, "t0": np.array([0.0, 1.0, 2.0])})
    summary = read_num_int_sol([1], outdof=jnp.array([0]), root=tmp_path)
    np.testing.assert_allclose(np.asarray(summary.omegas), np.array([np.nan, 2.5]), equal_nan=True)
    np.testing.assert_allclose(np.asarray(summary.norms[1]), np.linalg.norm(np.array([[1.0, -2.0], [3.0, 4.0]]), "fro") / np.sqrt(2))
    np.testing.assert_allclose(np.asarray(summary.amplitudes[1]), 3.0)

    z_items = np.empty((2,), dtype=object)
    z_items[0] = np.array([[1.0, 2.0], [3.0, 4.0]])
    z_items[1] = np.array([[5.0, 6.0], [7.0, 8.0]])
    savemat(tmp_path / "po_ssm2.mat", {"Z": z_items, "Omega_0": np.array([0.5, 0.75])})
    init = read_po_ssm_init([2], root=tmp_path)
    np.testing.assert_allclose(np.asarray(init.z_init), np.array([[1.0, 5.0], [3.0, 7.0]]))
    np.testing.assert_allclose(np.asarray(init.omega), np.array([0.5, 0.75]))
