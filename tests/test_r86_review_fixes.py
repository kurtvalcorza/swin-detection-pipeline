"""Row-86 review fixes (2026-10-18): the val2017 default sample, DAT9 disclosure, refusals, outputs and text.

Every test here runs offline with the CI dependency budget (NumPy, Pillow, pytest): the notebook's own cells
are executed with inert stand-ins for the model and the carried package helpers. No model is run.
"""
# ruff: noqa: E501  -- long literal digests and test names are kept on one line
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "swin_detection_task_inference.ipynb"
PIN = json.loads((ROOT / "tools" / "coco_val2017_sample_50.json").read_text(encoding="utf-8"))
PERMISSIVE = {"CC BY 2.0", "CC BY-SA 2.0", "No known copyright restrictions", "United States Government Work"}


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell(nb: dict, marker: str) -> str:
    matches = [s for s in _code(nb) if marker in s]
    assert len(matches) == 1, marker
    return matches[0]


def _validate_inputs(paths, score_threshold=0.0, max_detections=1, names=None):
    from PIL import Image

    paths = list(paths) if isinstance(paths, (list, tuple)) else [paths]
    findings = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"{path.name}: file does not exist")
        try:
            with Image.open(path) as image:
                image.load()
        except Exception as exc:  # noqa: BLE001 - stand-in for the package's validate_image
            raise ValueError(f"{path.name}: Pillow cannot decode it ({type(exc).__name__}).") from None
        findings.append({"input": path.name, "verdict": "accepted"})
    return {"verdict": "accepted", "findings": findings}


def _namespace() -> dict:
    import numpy

    return {"numpy": numpy, "validate_inputs": _validate_inputs, "__name__": "notebook_cell"}


def _run(source: str, namespace: dict) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        exec(source, namespace)  # noqa: S102 - the notebook's own cell
    return out.getvalue()


def _sample(nb: dict, **params: object) -> str:
    source = _cell(nb, "SAMPLE = 'val2017'  # @param")
    for name, value in params.items():
        line = re.search(rf"^{name} = .*$", source, re.M)
        assert line, name
        source = source.replace(line.group(0), f"{name} = {value!r}  # @param", 1)
    return source


# ---- R86-M1 / D1 / D2: the pinned val2017 sample -------------------------------------------------------------------


def test_r86_D2_pin_is_the_rule_selected_50_with_the_reviewed_digest() -> None:
    assert PIN["n_images"] == 50 and PIN["n_boxes"] == 279 and PIN["n_categories"] == 59
    assert PIN["gt_digest_sha256_review_form"] == "b6057ca6502f2e80cb468bc2f96d5035efc943ca71684a380f1347818e8ed63f"
    assert PIN["annotations_zip_sha256"].startswith("113a836d")
    assert PIN["training_overlap"]["checkpoint_training_split"] == "train2017" and PIN["training_overlap"]["eval_data_in_training"] is False
    ids = [image["image_id"] for image in PIN["images"]]
    assert ids == sorted(ids) and len(set(ids)) == 50
    digest = hashlib.sha256(json.dumps([[i["file_name"], i["boxes"]] for i in PIN["images"]], separators=(",", ":")).encode()).hexdigest()
    assert digest == PIN["gt_digest_sha256"]
    for image in PIN["images"]:
        assert re.fullmatch(r"[0-9a-f]{64}", image["sha256"]) and image["format"] == "JPEG"
        assert image["release_url"].endswith("/releases/download/data-coco-val2017-50-v1/" + image["file_name"])
        assert image["coco_url"] == "http://images.cocodataset.org/val2017/" + image["file_name"]
        assert image["licence"] in {"Attribution License", "Attribution-ShareAlike License", "No known copyright restrictions", "United States Government Work"}
        for class_index, x1, y1, x2, y2 in image["boxes"]:
            assert 0 <= class_index < 80 and x2 > x1 and y2 > y1


def test_r86_M1_notebook_rows_equal_the_pin_and_the_digest_check(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns = _namespace()
    _run(_sample(notebook, SAMPLE="synthetic"), ns)
    rows = ns["VAL2017_IMAGES"]
    assert [(r[0], r[1], r[2], r[4], r[5], r[6]) for r in rows] == [(i["file_name"], i["sha256"], i["bytes"], i["licence_url"], i["flickr_url"], i["boxes"]) for i in PIN["images"]]
    assert {r[3] for r in rows} <= PERMISSIVE
    assert ns["VAL2017_GT_SHA256"] == PIN["gt_digest_sha256"]
    assert dict(ns["VAL2017_TRAINING_OVERLAP"])["eval_data_in_training"] is False
    assert ns["DEFAULT_SAMPLE_KIND"] == "COCO-val2017-50"


def test_r86_M1_val2017_path_refuses_wrong_bytes(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    import urllib.request

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=60: _Response(b"not the pinned image"))
    with pytest.raises(RuntimeError, match="refusing to use it"):
        _run(_sample(notebook), _namespace())
    assert not list((tmp_path / "sample").glob("*.jpg"))


# ---- R86-m5 (DAT20): refusals are messages, and earlier outputs move aside -----------------------------------------


def _byod(notebook: dict, tmp_path: Path, monkeypatch, path: str = "", uploaded: dict | None = None) -> tuple[dict, str]:
    monkeypatch.chdir(tmp_path)
    if uploaded is not None:
        colab = types.ModuleType("google.colab")
        colab.files = types.SimpleNamespace(upload=lambda: uploaded)
        google = types.ModuleType("google")
        google.colab = colab
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab)
    else:
        monkeypatch.setitem(sys.modules, "google.colab", None)
    ns = _namespace()
    return ns, _run(_sample(notebook, USE_BYOD=True, BYOD_IMAGE_PATH=path), ns)


@pytest.mark.parametrize(
    "case, rule",
    [
        ("missing", "is not a file in this runtime"),
        ("no-dialog", "There is no upload dialog outside Google Colab"),
        ("no-file", r"Upload exactly one image file \(received 0\)"),
        ("two-files", r"Upload exactly one image file \(received 2\)"),
        ("undecodable", "cannot be used"),
    ],
)
def test_r86_m5_byod_refusal_is_one_message_and_moves_outputs_aside(notebook: dict, tmp_path: Path, monkeypatch, case: str, rule: str) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "swin_detection_task_inference_evaluation_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "outputs" / "swin_detection_task_inference_result_exploratory_1.json").write_text("{}", encoding="utf-8")
    kwargs: dict = {}
    if case == "missing":
        kwargs = {"path": str(tmp_path / "absent.png")}
    elif case == "no-file":
        kwargs = {"uploaded": {}}
    elif case == "two-files":
        kwargs = {"uploaded": {"a.png": b"x", "b.png": b"y"}}
    elif case == "undecodable":
        kwargs = {"uploaded": {"broken.png": b"not an image"}}
    ns, out = _byod(notebook, tmp_path, monkeypatch, **kwargs)
    assert ns["sample_ready"] is False
    assert re.search(rule, out) and out.startswith("BYOD input refused:"), out
    # Peer-review F8: only exploratory files move; the default sample's report stays where the text says it is.
    assert (tmp_path / "outputs" / "earlier_1" / "swin_detection_task_inference_result_exploratory_1.json").is_file()
    assert (tmp_path / "outputs" / "swin_detection_task_inference_evaluation_report.json").is_file()
    assert not list((tmp_path / "sample").glob("*.png"))  # uploads are kept under sample/byod/
    for section, marker in (("6", "input_manifest = validate_inputs("), ("7", "pipe.predict_many("), ("8", "report = evaluation_report("), ("9", "payload = {")):
        skipped = _run(_cell(notebook, marker), ns)
        assert skipped.startswith(f"Section {section} skipped"), skipped


def test_r86_m5_byod_by_path_is_accepted_without_colab(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    image = tmp_path / "mine.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(image)
    ns, _ = _byod(notebook, tmp_path, monkeypatch, path=str(image))
    assert ns["sample_ready"] is True and ns["sample_kind"] == "BYOD" and ns["image_paths"] == [image] and ns["ground_truth"] is None
    assert ns["OUTPUT_SUFFIX"] == "_exploratory_1"
    assert ns["training_overlap"]["eval_data_in_training"] is None  # peer-review F11: never a truthy string


def _validator():
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_release_assets", ROOT / "tools" / "validate_release_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r86_F1_only_the_exact_pinned_release_assets_are_allowed(notebook: dict) -> None:
    validator = _validator()
    code = "\n".join(_code(notebook))
    masked, problems = validator.data_release_violations(code)
    assert not problems and "github.com/kurtvalcorza" not in masked.lower()
    prefix = validator.DATA_RELEASE_PREFIX
    for injected in (
        f"urllib.request.urlopen('{prefix}../../../archive/refs/heads/main.zip')",
        f"urllib.request.urlopen('{prefix}%2e%2e/%2e%2e/%2e%2e/archive/refs/heads/main.zip')",
        f"urllib.request.urlopen('{prefix}000000000285.jpg?x=1')",
        f"exec(urllib.request.urlopen('{prefix}payload.py').read())",
        "urllib.request.urlopen(VAL2017_RELEASE_URL + '../../../archive/refs/heads/main.zip')",
        "urllib.request.urlopen('https://GitHub.com/KurtValcorza/swin-detection-pipeline/archive/main.zip')",
    ):
        masked, problems = validator.data_release_violations(code + "\n" + injected)
        flagged = bool(problems) or any(p.search(masked) for label, p in validator.FORBIDDEN_PATTERNS if label == "repository clone (ST1)")
        assert flagged, injected


def test_r86_F3_bootstrap_interval_is_computed_by_the_section_8_cell(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pycocotools")  # CI's offline budget has no pycocotools; the notebook run covers it there
    import numpy

    from dimer_swin_detection.metrics import coco_box_ap

    monkeypatch.chdir(tmp_path)
    ns = _namespace()
    _run(_sample(notebook, SAMPLE="synthetic"), ns)
    gt = [{"image_id": f"im{i}.png", "boxes": [{"class_id": i % 3, "bbox_xyxy": [10.0, 10.0, 60.0 + i, 80.0]}]} for i in range(6)]
    rows = [{"image_id": e["image_id"], "class_id": e["boxes"][0]["class_id"], "score": 0.9, "bbox_xyxy": e["boxes"][0]["bbox_xyxy"]} for e in gt]
    ns.update(numpy=numpy, coco_box_ap=coco_box_ap, rows=rows, ground_truth=gt, detections=rows, sample_kind="COCO-val2017-50",
              evaluation_report=lambda d, g, sample_kind: {"verdict": "sample-sanity", "sample_kind": sample_kind})
    _run(_cell(notebook, "report = evaluation_report("), ns)
    boot = ns["report"]["bootstrap_over_images"]
    assert boot["resamples"] == 200 and boot["seed"] == 0 and boot["coco_ap_50_95_95pct_interval"] == [1.0, 1.0]


# ---- R86-S1: atomic writes; exploratory passes never overwrite the default report ----------------------------------


def _section6(notebook: dict, ns: dict) -> None:
    ns.update(MAX_PIXELS=64_000_000, INPUT_SCHEMA={"score_threshold": [0, 1], "max_detections": [1, None]}, pipe=types.SimpleNamespace(classes=list(range(80))))
    _run(_cell(notebook, "input_manifest = validate_inputs("), ns)


def test_r86_S1_exploratory_passes_get_new_names_and_keep_the_default(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    default = outputs / "swin_detection_task_inference_input_manifest.json"
    default.write_text('{"default": true}', encoding="utf-8")
    for n in (1, 2):
        ns = _namespace()
        _run(_sample(notebook, SAMPLE="synthetic"), ns)
        assert ns["OUTPUT_SUFFIX"] == f"_exploratory_{n}"
        _section6(notebook, ns)
        assert (outputs / f"swin_detection_task_inference_input_manifest_exploratory_{n}.json").is_file()
    assert default.read_text(encoding="utf-8") == '{"default": true}'
    assert not list(outputs.glob("*.part"))


def test_r86_S1_a_failed_write_never_leaves_an_empty_main_report(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns = _namespace()
    _run(_sample(notebook, SAMPLE="synthetic"), ns)
    report = tmp_path / "outputs" / "swin_detection_task_inference_evaluation_report.json"
    ns["write_json_atomic"](report, {"verdict": "sample-sanity"})
    with pytest.raises(TypeError):
        ns["write_json_atomic"](report, {"unserialisable": object()})
    import os

    def _killed(src, dst):
        raise OSError("killed between write and rename")

    monkeypatch.setattr(os, "replace", _killed)
    with pytest.raises(OSError):
        ns["write_json_atomic"](report, {"verdict": "changed"})
    assert json.loads(report.read_text(encoding="utf-8")) == {"verdict": "sample-sanity"}


# ---- acceptance checks for the text findings -----------------------------------------------------------------------


def test_r86_M1_text_states_dat9_and_drops_the_overclaims(notebook: dict) -> None:
    md = _markdown(notebook)
    for gone in ("validation subset", "measures what it detects", "easier than the full COCO", "COCO8-val"):
        assert gone not in md and all(gone not in s for s in _code(notebook)), gone
    assert "train2017" in md and "ImageNet-1K" in md and "not an independent test" in md
    assert "resubstitution" in md
    for path in (ROOT / "MODEL_CARD.md", ROOT / "spec" / "task-inference-surface.json", ROOT / "README.md", ROOT / "tutorials" / "README.md"):
        text = path.read_text(encoding="utf-8")
        assert "validation subset" not in text and "COCO8 validation" not in text, path.name
    assert "train2017" in (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")


def test_r86_m1_licences_and_attribution_are_stated(notebook: dict) -> None:
    md = _markdown(notebook)
    assert "CC BY 4.0" in md and "33 CC BY 2.0" in md and "17 CC BY-SA 2.0" in md and "ATTRIBUTION.md" in md
    assert "'licences': sample_licences" in _cell(notebook, "payload = {")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in readme and "ATTRIBUTION.md" in readme


def test_r86_m2_quoted_numbers_name_their_revision(notebook: dict) -> None:
    md = _markdown(notebook)
    assert "0.707" in md and "`ea6b456d`" in md
    assert "`966c4ad9`" in md and "`181c09c`" in md
    assert "2026-09-11" not in md


def test_r86_m3_no_stale_release_text(notebook: dict) -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "USE_COCO8" not in text and "no Colab run" not in text and "COCO8_SHA256" not in text
    rv = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "USE_COCO8 = False" not in rv and "data-coco-val2017-50-v1" in rv


def test_r86_m4_spec_2_5_use_notice_and_isolated_env(notebook: dict) -> None:
    dimer = notebook["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code")
    before = ["".join(c["source"]) for c in notebook["cells"][:first_code]]
    assert any(s.startswith("> **⚠️ Notice.** For research, training and evaluation only.") for s in before)
    assert "DIMER_ISOLATED_ENV" in _markdown(notebook)
    refs = _markdown(notebook).split("## References", 1)[1]
    for n in range(1, 9):
        assert f"[{n}] " in refs and f"[{n}]" in _markdown(notebook).split("## References", 1)[0], n


def test_r86_S3_coco_ap_cap_is_stated(notebook: dict) -> None:
    md = _markdown(notebook)
    assert "maxDets=100" in md and "ranks all of them" not in md
