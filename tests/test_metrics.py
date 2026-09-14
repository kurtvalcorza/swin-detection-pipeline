"""Offline tests for the carried detection metric helpers (pycocotools stubbed: CI has no OpenMMLab stack)."""

from __future__ import annotations

import sys
import types

import pytest

from dimer_swin_detection import (
    MODEL_ID,
    MODEL_REVISION,
    boxes_from_yolo_labels,
    coco_box_ap,
    evaluation_report,
)
from dimer_swin_detection.metrics import COCO_NUM_CLASSES


def _stub_pycocotools(monkeypatch, stats: list[float]) -> dict:
    """A minimal pycocotools stand-in that records what coco_box_ap hands it."""
    captured: dict = {}

    class COCO:
        def __init__(self) -> None:
            self.dataset: dict = {}

        def createIndex(self) -> None:
            captured["gt"] = self.dataset

        def loadRes(self, results):
            captured["dt"] = results
            return "dt-object"

    class COCOeval:
        def __init__(self, gt, dt, iou_type) -> None:
            captured["iou_type"] = iou_type
            self.params = types.SimpleNamespace(imgIds=None)
            self.stats = stats

        def evaluate(self) -> None:
            captured["evaluated"] = True

        def accumulate(self) -> None:
            captured["accumulated"] = True

        def summarize(self) -> None:
            captured["img_ids"] = list(self.params.imgIds)

    root = types.ModuleType("pycocotools")
    coco = types.ModuleType("pycocotools.coco")
    coco.COCO = COCO
    cocoeval = types.ModuleType("pycocotools.cocoeval")
    cocoeval.COCOeval = COCOeval
    root.coco, root.cocoeval = coco, cocoeval
    monkeypatch.setitem(sys.modules, "pycocotools", root)
    monkeypatch.setitem(sys.modules, "pycocotools.coco", coco)
    monkeypatch.setitem(sys.modules, "pycocotools.cocoeval", cocoeval)
    return captured


def test_boxes_from_yolo_labels_converts_normalised_rows_to_xyxy_pixels() -> None:
    boxes = boxes_from_yolo_labels("16 0.5 0.5 0.5 0.25\n\n0 0.25 0.25 0.5 0.5\n", 200, 100)
    assert boxes == [
        {"class_id": 16, "bbox_xyxy": [50.0, 37.5, 150.0, 62.5]},
        {"class_id": 0, "bbox_xyxy": [0.0, 0.0, 100.0, 50.0]},
    ]
    with pytest.raises(ValueError, match="5 fields"):
        boxes_from_yolo_labels("1 0.5 0.5", 10, 10)
    with pytest.raises(ValueError, match="outside 0..79"):
        boxes_from_yolo_labels(f"{COCO_NUM_CLASSES} 0.5 0.5 0.1 0.1", 10, 10)


def test_coco_box_ap_builds_coco_structures_and_reads_the_three_ap_stats(monkeypatch) -> None:
    captured = _stub_pycocotools(monkeypatch, [0.5, 0.8, 0.4] + [0.0] * 9)
    detections = [{"image_id": "a.jpg", "class_id": 16, "score": 0.9, "bbox_xyxy": [10.0, 20.0, 30.0, 60.0]}]
    ground_truth = [
        {"image_id": "a.jpg", "boxes": [{"class_id": 16, "bbox_xyxy": [10, 20, 30, 60]}]},
        {"image_id": "b.jpg", "boxes": []},
    ]
    out = coco_box_ap(detections, ground_truth)
    assert (out["coco_ap_50_95"], out["ap50"], out["ap75"]) == (0.5, 0.8, 0.4)
    assert (out["n_images"], out["n_ground_truth_boxes"], out["n_detections"]) == (2, 1, 1)
    assert captured["iou_type"] == "bbox" and captured["img_ids"] == [1, 2]
    expected_annotation = {"id": 1, "image_id": 1, "category_id": 17, "bbox": [10.0, 20.0, 20.0, 40.0]}
    assert captured["gt"]["annotations"] == [{**expected_annotation, "area": 800.0, "iscrowd": 0}]
    assert len(captured["gt"]["categories"]) == COCO_NUM_CLASSES
    assert captured["dt"] == [
        {"image_id": 1, "category_id": 17, "bbox": [10.0, 20.0, 20.0, 40.0], "score": 0.9}
    ]
    assert captured["evaluated"] and captured["accumulated"]


def test_coco_box_ap_empty_detector_is_zero_without_pycocotools(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pycocotools", None)  # an import would raise ImportError
    out = coco_box_ap([], [{"image_id": "a.jpg", "boxes": [{"class_id": 0, "bbox_xyxy": [0, 0, 1, 1]}]}])
    assert (out["coco_ap_50_95"], out["ap50"], out["ap75"]) == (0.0, 0.0, 0.0)
    assert out["n_detections"] == 0


def test_coco_box_ap_rejects_unknown_images_and_bad_ids() -> None:
    gt = [{"image_id": "a.jpg", "boxes": []}]
    with pytest.raises(ValueError, match="no ground-truth entry"):
        coco_box_ap([{"image_id": "zzz.jpg", "class_id": 0, "score": 1.0, "bbox_xyxy": [0, 0, 1, 1]}], gt)
    with pytest.raises(ValueError, match="at least one image"):
        coco_box_ap([], [])
    with pytest.raises(ValueError, match="outside 0..79"):
        coco_box_ap([], [{"image_id": "a.jpg", "boxes": [{"class_id": 80, "bbox_xyxy": [0, 0, 1, 1]}]}])


def test_evaluation_report_sample_sanity_with_ground_truth(monkeypatch) -> None:
    _stub_pycocotools(monkeypatch, [0.71, 0.96, 0.70] + [0.0] * 9)
    detections = [
        {"image_id": "a.jpg", "class_id": 16, "class_name": "dog", "score": 0.9, "bbox_xyxy": [1, 2, 3, 4]}
    ]
    ground_truth = [{"image_id": "a.jpg", "boxes": [{"class_id": 16, "bbox_xyxy": [1, 2, 3, 4]}]}]
    report = evaluation_report(detections, ground_truth, sample_kind="COCO8-val")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "COCO8-val"
    assert [(m["id"], m["iou"], m["value"]) for m in report["metrics"]] == [
        ("coco_box_ap", "0.50:0.95", 0.71),
        ("coco_box_ap", "0.50", 0.96),
        ("coco_box_ap", "0.75", 0.70),
    ]
    assert report["baselines"] == [
        {
            "id": "empty_detector",
            "coco_box_ap": 0.0,
            "note": "a detector returning no boxes scores AP 0 by construction",
        }
    ]
    assert report["context"]["upstream_reported_box_ap"] == 42.7
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
