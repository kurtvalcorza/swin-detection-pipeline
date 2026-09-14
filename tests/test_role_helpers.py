"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dimer_swin_detection import (
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)
from dimer_swin_detection.runtime import Detection


def _image(tmp_path: Path, name: str = "a.png", size=(32, 24)) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


def test_validate_inputs_returns_manifest_with_schema_and_identity(tmp_path: Path) -> None:
    a, b = _image(tmp_path), _image(tmp_path, "b.jpg", (48, 48))
    manifest = validate_inputs([a, b], score_threshold=0.3, max_detections=10, names=["first", "second"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert [entry["id"] for entry in manifest["inputs"]] == ["first", "second"]
    assert manifest["inputs"][0]["width"] == 32 and manifest["inputs"][0]["height"] == 24
    assert manifest["inputs"][1]["mode"] == "RGB"
    assert (manifest["score_threshold"], manifest["max_detections"]) == (0.3, 10)
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_single_path_default_ids(tmp_path: Path) -> None:
    manifest = validate_inputs(_image(tmp_path, "solo.png"))
    assert [entry["id"] for entry in manifest["inputs"]] == ["solo.png"]


def test_validate_inputs_rejects_like_predict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="score_threshold must be between 0 and 1"):
        validate_inputs(_image(tmp_path), score_threshold=1.5)
    with pytest.raises(ValueError, match="max_detections must be positive"):
        validate_inputs(_image(tmp_path), max_detections=0)
    with pytest.raises(ValueError, match="Image does not exist"):
        validate_inputs(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        validate_inputs(bad)
    with pytest.raises(ValueError, match="names must have one entry per image"):
        validate_inputs(_image(tmp_path), names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    rows = [
        Detection("a.png", 0, "person", 0.9, (0.0, 0.0, 1.0, 1.0)),
        {"image_id": "b.png", "class_id": 2, "class_name": "car", "score": 0.5, "bbox_xyxy": [0, 0, 2, 2]},
    ]
    report = evaluation_report(rows, sample_kind="synthetic")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert (report["n_images"], report["n_detections"]) == (2, 2)
    assert "COCO AP" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert evaluation_report([])["n_detections"] == 0
