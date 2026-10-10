"""Regression tests for the 2026-10-05 notebook review findings (SWD-M1..M3, SWD-m1, SWD-m3).

They need only CI's lightweight dependencies (NumPy, Pillow, pytest): the generated notebook is checked statically and
its own BYOD cell source is executed with stand-ins (a fake Colab upload). No OpenMMLab, no weights; none of this is
model evidence. The default COCO8 path itself is executed by `.github/workflows/verify-task-tutorial.yml`.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"swd_fix_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[str]:
    return [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# ---- SWD-M1 / SWD-m3: an isolated, uv-managed CPython 3.10; no kernel install, no restart ------------------------


def test_swd_M1_first_code_cell_provisions_and_checks_python_310_before_any_install(notebook: dict) -> None:
    first = _code(notebook)[0]
    assert first.startswith("# @title Infrastructure: install the locked runtime into an isolated environment")
    assert "MANAGED_PYTHON = '3.10.18'" in first
    assert first.index("isolated_version != MANAGED_PYTHON") < first.index('"pip", "install"'), "the interpreter is checked before anything is installed"
    install = next(line for line in first.splitlines() if '"pip", "install"' in line)
    assert '"--managed-python"' in first and '"--require-hashes", "--only-binary", ":all:"' in install
    assert '"--index-strategy", "unsafe-best-match"' in install and '"--find-links"' in install
    pins = first[first.index("PINS = [") : first.index("]", first.index("PINS = ["))]
    assert "'torch==2.1.2+cpu'" in pins and "'mmcv==2.1.0'" in pins and "'--find-links'" in pins


def test_swd_m3_no_kernel_install_and_no_restart_instruction(notebook: dict) -> None:
    assert "Restart the runtime" not in NOTEBOOK.read_text(encoding="utf-8")
    sources = _code(notebook)
    assert not any("[sys.executable, '-m', 'pip'" in s for s in sources)
    assert sum("# dimer: kernel cell" in s for s in sources) == 2
    runtime_check = next(s for s in sources if "if sys.version_info[:2] != (3, 10):" in s)
    assert "Run the Section 1 cells first" in runtime_check


def test_swd_M1_no_entry_point_presents_colab_as_verified() -> None:
    for path in (NOTEBOOK, ROOT / "README.md", ROOT / "tutorials" / "README.md"):
        text = path.read_text(encoding="utf-8")
        assert "colab-badge.svg" not in text, path.name
        assert "verification pending" in text or "verification%20pending" in text, path.name


# ---- SWD-M2: the default path measures --------------------------------------------------------------------------


def test_swd_M2_default_sample_is_the_labelled_coco8_subset(notebook: dict) -> None:
    sample = next(s for s in _code(notebook) if "SAMPLE = 'coco8'" in s)
    assert "USE_COCO8" not in sample
    assert "elif SAMPLE == 'coco8':" in sample and "boxes_from_yolo_labels(label_text, width, height)" in sample
    workflow = (ROOT / ".github" / "workflows" / "verify-task-tutorial.yml").read_text(encoding="utf-8")
    assert "report['verdict']=='sample-sanity'" in workflow and "ground_truth_boxes']==17" in workflow
    assert "not-measurable" in _markdown(notebook) and "SAMPLE = 'synthetic'" in _markdown(notebook)


# ---- SWD-M3: the guided layer ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    ["## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model/System → Output", "## Roadmap", "<strong>Glossary</strong>", "**IoU (intersection over union)**", "**YOLO label format**", "**`.pth` trust boundary**", "**Predict before running:**", "<summary>Check your reasoning</summary>", "## 10. Activity: change one thing", "## Troubleshooting", "## Conclusion (your notes)"],
)
def test_swd_M3_guided_layer_marker_is_present(notebook: dict, marker: str) -> None:
    assert marker in _markdown(notebook)


def test_swd_M3_infrastructure_cells_are_collapsed(notebook: dict) -> None:
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    learner_start = next(i for i, c in enumerate(code) if c["source"].startswith("import sys"))
    for cell in code[:learner_start]:
        assert cell["metadata"].get("cellView") == "form", cell["source"][:60]
    assert _markdown(notebook).count("**Predict before running:**") >= 4


# ---- SWD-m1: BYOD by path, and a guarded upload ------------------------------------------------------------------


def _sample_cell(notebook: dict) -> str:
    return next(s for s in _code(notebook) if "SAMPLE = 'coco8'" in s)


def _run_byod(notebook: dict, tmp_path: Path, monkeypatch, path: str = "", uploaded: dict | None = None) -> dict:
    import numpy

    monkeypatch.chdir(tmp_path)
    if uploaded is not None:
        files = types.SimpleNamespace(upload=lambda: uploaded)
        colab = types.ModuleType("google.colab")
        colab.files = files
        google = types.ModuleType("google")
        google.colab = colab
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab)
    else:
        monkeypatch.setitem(sys.modules, "google.colab", None)
    source = _sample_cell(notebook).replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_IMAGE_PATH = ''", f"BYOD_IMAGE_PATH = {path!r}")
    namespace = {"numpy": numpy}
    exec(source, namespace)  # noqa: S102 - the notebook's own cell
    return namespace


def test_swd_m1_byod_by_path_works_without_colab(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    image = tmp_path / "mine.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(image)
    namespace = _run_byod(notebook, tmp_path, monkeypatch, path=str(image))
    assert namespace["sample_kind"] == "BYOD" and namespace["image_paths"] == [image] and namespace["ground_truth"] is None


@pytest.mark.parametrize("uploaded", [{}, {"a.png": b"x", "b.png": b"y"}])
def test_swd_m1_cancelled_or_multi_file_upload_names_the_rule(notebook: dict, tmp_path: Path, monkeypatch, uploaded: dict) -> None:
    with pytest.raises(ValueError, match=r"Upload exactly one image file \(received \d\)"):
        _run_byod(notebook, tmp_path, monkeypatch, uploaded=uploaded)


def test_swd_m1_no_dialog_outside_colab_says_to_set_the_path(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="set BYOD_IMAGE_PATH"):
        _run_byod(notebook, tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="is not a file in this runtime"):
        _run_byod(notebook, tmp_path, monkeypatch, path=str(tmp_path / "absent.png"))
