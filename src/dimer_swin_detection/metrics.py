"""Detection metric helpers carried by the standalone tutorial (NOTEBOOK_SPEC 1.1 EVAL2).

`coco_box_ap` is the repository's only detection metric: COCO AP@[0.50:0.95], AP50 and AP75 over
axis-aligned boxes, computed by `pycocotools` on ground truth the caller supplies. `pycocotools` is
imported lazily so the module (and the notebook cell that carries it) loads without it; the pinned
runtime installs `pycocotools==2.0.11`. `boxes_from_yolo_labels` converts a YOLO-format label file
(normalised `class xc yc w h` rows, as the public COCO8 sample ships) into the ground-truth box shape
`coco_box_ap` consumes. No model logic lives here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

COCO_NUM_CLASSES = 80  # MMDetection COCO class ordering; COCO category id = class_id + 1 below


def boxes_from_yolo_labels(text: str, width: int, height: int) -> list[dict[str, Any]]:
    """Parse YOLO label rows (`class xc yc w h`, normalised to the image size) into xyxy pixel boxes."""
    if width < 1 or height < 1:
        raise ValueError(f"image size must be positive; got {width}x{height}")
    boxes: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"label line {line_number} must have 5 fields (class xc yc w h): {raw!r}")
        class_id = int(parts[0])
        xc, yc, bw, bh = (float(value) for value in parts[1:])
        if not 0 <= class_id < COCO_NUM_CLASSES:
            raise ValueError(
                f"label line {line_number}: class id {class_id} outside 0..{COCO_NUM_CLASSES - 1}"
            )
        w, h = bw * width, bh * height
        x1, y1 = xc * width - w / 2, yc * height - h / 2
        boxes.append({"class_id": class_id, "bbox_xyxy": [x1, y1, x1 + w, y1 + h]})
    return boxes


def _xywh(bbox_xyxy: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"bbox_xyxy must satisfy x1<=x2 and y1<=y2: {[x1, y1, x2, y2]}")
    return [x1, y1, x2 - x1, y2 - y1]


def coco_box_ap(
    detections: Iterable[Mapping[str, Any]],
    ground_truth: Iterable[Mapping[str, Any]],
    *,
    num_classes: int = COCO_NUM_CLASSES,
) -> dict[str, Any]:
    """COCO AP@[0.50:0.95], AP50 and AP75 of `detections` against `ground_truth` (pycocotools, bbox).

    `detections`: rows shaped like `Detection.to_dict()` (`image_id`, `class_id`, `score`, `bbox_xyxy`).
    `ground_truth`: one entry per evaluated image,
    `{"image_id": <name>, "boxes": [{"class_id", "bbox_xyxy"}, ...]}`; every detection must name an image
    present in the ground truth. A detector that returns no boxes scores AP 0.0 by construction (the
    empty-detector baseline) and is reported without invoking pycocotools.
    """
    gt_entries = [dict(entry) for entry in ground_truth]
    if not gt_entries:
        raise ValueError("ground_truth must name at least one image")
    image_ids: dict[str, int] = {}
    images = []
    annotations = []
    for entry in gt_entries:
        name = str(entry["image_id"])
        if name in image_ids:
            raise ValueError(f"duplicate ground-truth image_id {name!r}")
        image_ids[name] = len(image_ids) + 1
        images.append({"id": image_ids[name], "file_name": name})
        for box in entry.get("boxes", []):
            class_id = int(box["class_id"])
            if not 0 <= class_id < num_classes:
                raise ValueError(f"ground-truth class id {class_id} outside 0..{num_classes - 1}")
            x, y, w, h = _xywh(box["bbox_xyxy"])
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_ids[name],
                    "category_id": class_id + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
    results = []
    for row in detections:
        name = str(row["image_id"])
        if name not in image_ids:
            raise ValueError(f"detection names image {name!r} that has no ground-truth entry")
        class_id = int(row["class_id"])
        if not 0 <= class_id < num_classes:
            raise ValueError(f"detection class id {class_id} outside 0..{num_classes - 1}")
        results.append(
            {
                "image_id": image_ids[name],
                "category_id": class_id + 1,
                "bbox": _xywh(row["bbox_xyxy"]),
                "score": float(row["score"]),
            }
        )
    summary = {
        "n_images": len(images),
        "n_ground_truth_boxes": len(annotations),
        "n_detections": len(results),
        "iou_type": "bbox",
    }
    if not results:
        empty = {"coco_ap_50_95": 0.0, "ap50": 0.0, "ap75": 0.0, "note": "no detections: empty-detector AP"}
        return {**summary, **empty}
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO()
    gt.dataset = {
        "info": {"description": "tutorial ground truth"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": str(index)} for index in range(num_classes)],
    }
    gt.createIndex()
    dt = gt.loadRes(results)
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = sorted(image_ids.values())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    return {**summary, "coco_ap_50_95": stats[0], "ap50": stats[1], "ap75": stats[2]}
