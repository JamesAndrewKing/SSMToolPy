from __future__ import annotations

from pathlib import Path
import re
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

from ssmtoolpy.systems.planar import planar_ssm_graph_coefficients


ROOT = Path(__file__).resolve().parents[1]
PLANAR_DIR = ROOT / "SSMTool" / "examples" / "PlanarSystem"
BENCHMARK_DIR = ROOT / "SSMTool" / "examples" / "BenchamrkSSM1stOrder"


def _normalized_matlab_source(path: Path) -> str:
    return re.sub(r"\s+", "", path.read_text())


def _mlx_document_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return archive.read("matlab/document.xml").decode("utf-8")


def test_benchmark_build_model_matches_planar_system_source() -> None:
    assert (BENCHMARK_DIR / "build_model.m").exists()
    assert _normalized_matlab_source(BENCHMARK_DIR / "build_model.m") == _normalized_matlab_source(
        PLANAR_DIR / "build_model.m"
    )


def test_benchmark_mlx_documents_same_analytical_coefficients() -> None:
    document = _mlx_document_text(BENCHMARK_DIR / "demo.mlx")

    assert r"\dot{x}=-x" in document
    assert r"\dot{y}=-\sqrt{24}y+x^2+x^3+x^4+x^5" in document
    assert "coeffs(2,2:5)" in document
    for degree in range(2, 6):
        assert rf"\sqrt{{24}}-{degree}" in document


def test_benchmark_coefficients_match_planar_solver_regression() -> None:
    coefficients = planar_ssm_graph_coefficients(order=8)
    expected = np.zeros(9)
    for degree in range(2, 6):
        expected[degree] = 1.0 / (np.sqrt(24.0) - degree)

    np.testing.assert_allclose(np.asarray(coefficients), expected, rtol=1e-12)


def test_benchmark_named_regression_keeps_jax_transform_path() -> None:
    gradient = jax.grad(
        lambda decay: jnp.sum(planar_ssm_graph_coefficients(5, decay))
    )(jnp.sqrt(24.0))
    expected = -sum(1.0 / (np.sqrt(24.0) - degree) ** 2 for degree in range(2, 6))

    np.testing.assert_allclose(np.asarray(gradient), expected, rtol=1e-12)
