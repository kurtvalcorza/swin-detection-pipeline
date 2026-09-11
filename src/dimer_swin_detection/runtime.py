from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

MODEL_SPEC = {
    "runtime_id": "swin-t-mask-rcnn-coco-openmmlab-v3.3.0",
    "architecture": "Swin-T + Mask R-CNN",
    "task": "object-detection",
    "dataset": "COCO 2017",
    "mmdetection_version": "3.3.0",
    "mmcv_version": "2.1.0",
    "mmengine_version": "0.10.7",
    "torch_version": "2.1.2",
    "config": "swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
    "config_source_revision": "open-mmlab/mmdetection@v3.3.0",
    "checkpoint_url": "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth",
    "checkpoint_size_bytes": 191461353,
    "checkpoint_sha256": "9d6b7cfaa4aad52ef559611bea454f01d6f1f17c82a1abfac0d71631a193a291",
    "upstream_reported_box_ap": 42.7,
    "instance_masks_in_dimer_contract": False,
}


@dataclass(frozen=True)
class Detection:
    image_id: str
    class_id: int
    class_name: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox_xyxy"] = list(self.bbox_xyxy)
        return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _version(dist: str) -> str:
    return importlib.metadata.version(dist)


def verify_runtime_versions() -> dict[str, str]:
    expected = {
        "mmdet": MODEL_SPEC["mmdetection_version"],
        "mmcv": MODEL_SPEC["mmcv_version"],
        "mmengine": MODEL_SPEC["mmengine_version"],
    }
    actual = {
        "mmdet": _version("mmdet"),
        "mmcv": _version("mmcv"),
        "mmengine": _version("mmengine"),
    }
    for name, expected_version in expected.items():
        if actual[name] != expected_version:
            raise RuntimeError(
                f"Unsupported {name} {actual[name]}; this runtime is qualified for {expected_version}."
            )
    return actual


def resolve_packaged_config() -> Path:
    import mmdet

    config = (
        Path(mmdet.__file__).resolve().parent
        / ".mim"
        / "configs"
        / MODEL_SPEC["config"]
    )
    if not config.is_file():
        raise RuntimeError(
            f"The pinned MMDetection package does not contain the expected config: {config}"
        )
    return config


def acquire_verified_checkpoint(cache_dir: str | os.PathLike = ".dimer-models") -> Path:
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / Path(MODEL_SPEC["checkpoint_url"]).name

    def valid(path: Path) -> bool:
        return (
            path.is_file()
            and path.stat().st_size == MODEL_SPEC["checkpoint_size_bytes"]
            and _sha256(path) == MODEL_SPEC["checkpoint_sha256"]
        )

    if valid(target):
        return target
    if target.exists():
        target.unlink()

    fd, temporary_name = tempfile.mkstemp(prefix="dimer-swin-detection-", suffix=".pth", dir=root)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(MODEL_SPEC["checkpoint_url"], timeout=120) as response, temporary.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if temporary.stat().st_size != MODEL_SPEC["checkpoint_size_bytes"]:
            raise RuntimeError(
                f"Checkpoint size mismatch: got {temporary.stat().st_size}, expected {MODEL_SPEC['checkpoint_size_bytes']}."
            )
        digest = _sha256(temporary)
        if digest != MODEL_SPEC["checkpoint_sha256"]:
            raise RuntimeError(
                f"Checkpoint SHA-256 mismatch: got {digest}, expected {MODEL_SPEC['checkpoint_sha256']}."
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def validate_image(path: str | os.PathLike, *, max_pixels: int = 64_000_000) -> dict:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise ValueError(f"Image does not exist: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
            mode = image.mode
    except Exception as exc:
        raise ValueError(f"Input is not a readable image: {image_path}: {exc}") from exc
    if width < 1 or height < 1:
        raise ValueError(f"Image dimensions must be positive; got {width}x{height}.")
    if width * height > max_pixels:
        raise ValueError(
            f"Image has {width * height:,} pixels; ceiling is {max_pixels:,}. Resize before inference."
        )
    return {"path": str(image_path), "width": width, "height": height, "mode": mode}


class DimerSwinDetector:
    """Public task-inference API for the pinned DIMER Swin object detector.

    The upstream `.pth` checkpoint is a code-capable PyTorch serialization. This
    runtime verifies its size and SHA-256 before MMDetection deserializes it, but
    digest verification establishes byte identity, not author authenticity.
    Only use the checkpoint when the pinned OpenMMLab source is trusted.
    """

    def __init__(self, *, cache_dir: str | os.PathLike = ".dimer-models", device: str = "cpu"):
        versions = verify_runtime_versions()
        checkpoint = acquire_verified_checkpoint(cache_dir)
        config = resolve_packaged_config()
        from mmdet.apis import init_detector

        self.model = init_detector(str(config), str(checkpoint), device=device)
        self.device = device
        self.checkpoint = checkpoint
        self.config = config
        self.versions = versions
        self.classes = tuple(self.model.dataset_meta.get("classes", ()))
        if not self.classes:
            raise RuntimeError("MMDetection model did not expose its class ordering.")

    def predict(
        self,
        image: str | os.PathLike,
        *,
        score_threshold: float = 0.0,
        max_detections: int = 300,
    ) -> list[Detection]:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1.")
        if max_detections < 1:
            raise ValueError("max_detections must be positive.")
        info = validate_image(image)
        from mmdet.apis import inference_detector

        sample = inference_detector(self.model, info["path"])
        instances = sample.pred_instances.cpu()
        detections: list[Detection] = []
        for bbox, score, label in zip(instances.bboxes, instances.scores, instances.labels):
            score_value = float(score.item())
            if score_value < score_threshold:
                continue
            label_value = int(label.item())
            coords = tuple(float(v) for v in bbox.tolist())
            detections.append(
                Detection(
                    image_id=Path(info["path"]).name,
                    class_id=label_value,
                    class_name=self.classes[label_value],
                    score=score_value,
                    bbox_xyxy=coords,
                )
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:max_detections]

    def predict_many(self, images: Iterable[str | os.PathLike], **kwargs) -> list[Detection]:
        output: list[Detection] = []
        for image in images:
            output.extend(self.predict(image, **kwargs))
        return output

    def provenance(self) -> dict:
        import platform
        import torch

        return {
            "runtime": MODEL_SPEC,
            "effective": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                **self.versions,
                "device": self.device,
                "classes": list(self.classes),
                "checkpoint_path": str(self.checkpoint),
                "checkpoint_sha256": _sha256(self.checkpoint),
            },
            "score_semantics": "MMDetection class confidence scores; uncalibrated, not probabilities of correctness.",
            "decision_threshold": "caller-owned; score_threshold is an output filter, not a universal deployment threshold.",
        }

    def write_provenance(self, path: str | os.PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.provenance(), indent=2) + "\n")
        return target
