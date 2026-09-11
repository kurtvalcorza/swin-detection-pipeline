from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import mmdet
import torch
from mmdet.apis import DetInferencer

MODEL_CONFIG = "mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py"
CHECKPOINT_URL = "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth"
SAMPLE_URL = "https://raw.githubusercontent.com/open-mmlab/mmdetection/v3.3.0/demo/demo.jpg"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


root = Path(".runtime-probe")
root.mkdir(exist_ok=True)
checkpoint = root / "model.pth"
sample = root / "demo.jpg"
if not checkpoint.exists():
    urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
if not sample.exists():
    urllib.request.urlretrieve(SAMPLE_URL, sample)

config = Path(mmdet.__file__).resolve().parent / ".mim" / "configs" / "swin" / MODEL_CONFIG
if not config.is_file():
    raise RuntimeError(f"Packaged MMDetection config not found: {config}")

digest = sha256(checkpoint)
print(json.dumps({
    "torch": torch.__version__,
    "mmdet": mmdet.__version__,
    "config": str(config),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "checkpoint_sha256": digest,
}, indent=2))

inferencer = DetInferencer(model=str(config), weights=str(checkpoint), device="cpu")
result = inferencer(str(sample), pred_score_thr=0.5, no_save_pred=True, return_vis=False)
pred = result["predictions"][0]
print(json.dumps({
    "num_detections_ge_0_5": len(pred["scores"]),
    "top_score": max(pred["scores"]) if pred["scores"] else None,
    "labels_head": pred["labels"][:5],
    "boxes_head": pred["bboxes"][:2],
}, indent=2))
assert pred["scores"], "Expected at least one detection on MMDetection demo image"
print("TASK-INFERENCE RUNTIME PROBE: PASS")
