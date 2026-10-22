"""NOTEBOOK_SPEC 3.0 migration (2026-10-22): declaration, §7.2 section order, and the optional sections.

Offline, with CI's dependency budget (NumPy, Pillow, pytest). The optional-section cells are executed with the four
stage functions replaced by recorders, so no model runs; none of this is execution evidence.
"""
# ruff: noqa: E501
from __future__ import annotations

import copy
import importlib.util
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "swin_detection_task_inference.ipynb"
STAGES = ("validate_stage", "detect_stage", "evaluate_stage", "export_stage")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"spec3_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _code_cell(nb: dict, marker: str) -> str:
    matches = [_source(c) for c in nb["cells"] if c["cell_type"] == "code" and marker in _source(c)]
    assert len(matches) == 1, marker
    return matches[0]


def _run(source: str, ns: dict) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        exec(source, ns)  # noqa: S102 - the notebook's own cell
    return out.getvalue()


def _optional_namespace(nb: dict, tmp_path: Path, monkeypatch) -> tuple[dict, list]:
    """Section 5's definitions (no default download) plus recording stand-ins for the four stage functions."""
    import numpy

    monkeypatch.chdir(tmp_path)
    ns = {"numpy": numpy, "validate_inputs": lambda *a, **k: {"verdict": "accepted", "findings": []}, "__name__": "notebook_cell"}
    source = _code_cell(nb, "sample = load_input('val2017')").rstrip()
    _run(source[: -len("sample = load_input('val2017')")], ns)
    calls: list[tuple[str, str]] = []

    def recorder(stage):
        def call(prepared, *args):
            calls.append((stage, prepared["kind"]))
            return ([], []) if stage == "detect_stage" else {"verdict": "not-measurable"}

        return call

    ns.update({stage: recorder(stage) for stage in STAGES})
    return ns, calls


def test_declares_spec_3_0_tutorial_kind(notebook: dict) -> None:
    dimer = notebook["metadata"]["dimer"]
    assert (dimer["notebook_spec"], dimer["notebook_kind"], dimer["notebook_profile"]) == ("3.0", "TUTORIAL", "TASK-INFERENCE")
    assert "**Notebook specification:** DIMER Notebook Specification 3.0" in _source(notebook["cells"][0])
    assert _load("build_notebook").NOTEBOOK_SPEC == _load("validate_release_assets").NOTEBOOK_SPEC == "3.0"


def test_section_order_follows_7_2_and_the_validator_enforces_it(notebook: dict) -> None:
    validator = _load("validate_release_assets")
    validator._validate_section_order(NOTEBOOK, notebook)  # the committed notebook passes
    swapped = copy.deepcopy(notebook)
    cells = swapped["cells"]
    byod = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "USE_BYOD = False" in _source(c))
    interpretation = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "## Interpretation and limits" in _source(c))
    cells.insert(interpretation, cells.pop(byod))  # the BYOD cell moved ahead of the interpretation
    with pytest.raises(validator.ValidationError, match="must follow Interpretation"):
        validator._validate_section_order(NOTEBOOK, swapped)


def test_default_path_cells_carry_no_optional_form_fields(notebook: dict) -> None:
    interpretation = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "markdown" and "## Interpretation and limits" in _source(c))
    before = [_source(c) for c in notebook["cells"][:interpretation] if c["cell_type"] == "code"]
    for name in ("USE_BYOD", "BYOD_IMAGE_PATH", "RUN_ACTIVITY", "SAMPLE"):
        assert not any(re.search(rf"^{name} = .*# @param", source, re.M) for source in before), name


@pytest.mark.parametrize("gate, marker", [("RUN_ACTIVITY", "Optional activity skipped"), ("USE_BYOD", "BYOD skipped")])
def test_optional_sections_are_off_by_default_and_run_nothing(notebook: dict, tmp_path: Path, monkeypatch, gate: str, marker: str) -> None:
    ns, calls = _optional_namespace(notebook, tmp_path, monkeypatch)
    source = _code_cell(notebook, f"{gate} = False  # @param")
    out = _run(source, ns)
    assert out.startswith(marker) and calls == []
    assert not list((tmp_path / "outputs").glob("*_exploratory_*"))


def test_activity_runs_the_four_stages_on_the_synthetic_scene(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns, calls = _optional_namespace(notebook, tmp_path, monkeypatch)
    source = _code_cell(notebook, "RUN_ACTIVITY = False  # @param").replace("RUN_ACTIVITY = False", "RUN_ACTIVITY = True", 1)
    _run(source, ns)
    assert calls == [(stage, "synthetic") for stage in STAGES]
    activity = ns["activity"]
    assert activity["ground_truth"] is None and activity["output_suffix"] == "_exploratory_1"
    assert [p.name for p in activity["image_paths"]] == ["synthetic_scene_640x480.png"]


def test_generator_refuses_the_retired_demo_kind() -> None:
    build = _load("build_notebook")
    template = dict(_load("notebook_template").TEMPLATE, kind="DEMO")
    with pytest.raises(SystemExit, match="DEMO"):
        build.render(ROOT, template)
