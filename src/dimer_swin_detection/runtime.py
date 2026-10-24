from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import urllib.request
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .metrics import coco_box_ap

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


# ---------------------------------------------------------------------------------------------
# Fleet snapshot scheme (DIMER standalone carrier). The identity constants below name the OpenMMLab
# distribution: MODEL_ID is the config recipe inside the pinned package, MODEL_REVISION the upstream
# git commit of that package's release tag (the config source), and the manifest pins the checkpoint
# bytes. The checkpoint host is download.openmmlab.com, not the Hugging Face Hub, so the staging
# downloader is the pinned URL in MODEL_SPEC rather than hf_hub_download.
# ---------------------------------------------------------------------------------------------
MODEL_ID = "open-mmlab/mmdetection:mask-rcnn_swin-t-p4-w7_fpn_1x_coco"
MODEL_REVISION = "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"
MODEL_LICENSE = "Apache-2.0"
MODEL_KEY = "swin-t-mask-rcnn-coco"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth"
# The served file: the checkpoint's state dict, converted once into safetensors (no code, no pickle). The
# pickled `.pth` above is only the conversion source; once this file exists, the source is never read again.
CONVERTED_WEIGHTS_FILE = "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.safetensors"
CONVERTED_SHA256 = "9ed16ad66c9ca031a67bbcfb4ae424fa069a29251de9aa6f38944823399a994b"
CONVERTED_METADATA = {"format": "pt"}
# mmengine's `load_checkpoint` default: strip a DataParallel `module.` prefix (mmengine 0.10.7
# runner/checkpoint.py, `_load_checkpoint_to_model(revise_keys=[(r'^module\.', '')])`).
CHECKPOINT_REVISE_KEYS = ((r"^module\.", ""),)
# The checkpoint's `meta['CLASSES']`, which `init_detector` copies into `dataset_meta` when it loads the `.pth`
# (equal to `mmdet.datasets.CocoDataset.METAINFO['classes']`). Building with no checkpoint would give
# `get_classes('coco')` instead, whose 15 multi-word names use underscores, so the served model carries this pin.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
MAX_PIXELS = 64_000_000  # validate_image ceiling


def verify_snapshot(path: str | os.PathLike | None = None) -> dict:
    """Check a local snapshot against its manifest; raise naming the first mismatch.

    The checkpoint entry must also carry the digest MODEL_SPEC has always pinned, so the manifest
    cannot silently re-point the runtime at different bytes.
    """
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    entries = {entry["path"]: entry for entry in manifest.get("files", [])}
    pinned = entries.get(WEIGHTS_FILE)
    if pinned is None or pinned["sha256"] != MODEL_SPEC["checkpoint_sha256"] or pinned["bytes"] != MODEL_SPEC["checkpoint_size_bytes"]:
        raise ValueError(f"manifest entry for {WEIGHTS_FILE} does not match the checkpoint digest pinned in MODEL_SPEC")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _openmmlab_download(relative_path: str, root: Path) -> None:
    """Fetch the pinned OpenMMLab checkpoint into the snapshot directory (the only manifest entry)."""
    if relative_path != WEIGHTS_FILE:
        raise ValueError(f"no pinned download source for {relative_path}")
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix="dimer-swin-", suffix=".pth", dir=root)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(MODEL_SPEC["checkpoint_url"], timeout=120) as response, temporary.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def stage_missing_files(
    path: str | os.PathLike | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the checkpoint). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them from {MODEL_SPEC['checkpoint_url']}"
        )
    fetch = downloader or _openmmlab_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


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


def build_unloaded_model(config: str | os.PathLike, device: str = "cpu"):
    """Build the detector from the pinned config with no checkpoint: ``init_detector(config, None)``.

    MMDetection's own initialiser builds the model, clears the backbone's pretrained ``init_cfg`` (nothing is
    downloaded) and adds the config's palette to ``dataset_meta``, without reading any weights file. The class
    names are then set to `COCO_CLASSES`, the checkpoint's own list (`convert_checkpoint` refuses a checkpoint
    that carries another), so ``dataset_meta`` equals what loading the checkpoint would have produced.
    """
    from mmdet.apis import init_detector

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="checkpoint is None, use COCO classes by default.")
        model = init_detector(str(config), None, device=device)
    model.dataset_meta = {**model.dataset_meta, "classes": COCO_CLASSES}
    return model


def checkpoint_state_dict(checkpoint: Mapping) -> dict:
    """The model weights of an MMEngine checkpoint, keyed as mmengine's loader keys them.

    Mirrors ``mmengine.runner.checkpoint._load_checkpoint_to_model`` (0.10.7): take ``checkpoint['state_dict']``
    when present (else the mapping itself), then apply ``revise_keys`` (strip a leading ``module.``). The
    ``meta`` entry (training config, class names, versions) is not weights and is not carried.
    """
    import torch

    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint did not unpickle to a mapping")
    state = checkpoint.get("state_dict", checkpoint)
    revised: dict = {}
    for key, value in state.items():
        for pattern, replacement in CHECKPOINT_REVISE_KEYS:
            key = re.sub(pattern, replacement, key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint entry {key!r} is not a tensor")
        revised[key] = value
    return revised


def _write_converted(state: Mapping, target: Path) -> None:
    """Write a state dict as safetensors, byte-reproducibly: every tensor is cloned into its own contiguous
    CPU storage (safetensors refuses shared storage) and the one-key metadata header has a fixed order."""
    from safetensors.torch import save_file

    tensors = {name: value.detach().to("cpu").contiguous().clone() for name, value in state.items()}
    save_file(tensors, str(target), metadata=dict(CONVERTED_METADATA))


def verify_converted(path: str | os.PathLike) -> Path:
    """Check a converted serving file against its pinned SHA-256; raise on a missing or altered file."""
    file_path = Path(path)
    if file_path.name != CONVERTED_WEIGHTS_FILE:
        raise ValueError(
            f"{file_path.name}: the detector serves only {CONVERTED_WEIGHTS_FILE}; convert the pickled "
            f"{WEIGHTS_FILE} once with convert_checkpoint()"
        )
    if not file_path.is_file():
        raise FileNotFoundError(f"converted weights missing: {file_path}")
    digest = _sha256(file_path)
    if digest != CONVERTED_SHA256:
        raise ValueError(f"{CONVERTED_WEIGHTS_FILE}: sha256 {digest} != pinned {CONVERTED_SHA256}")
    return file_path


def convert_checkpoint(source: str | os.PathLike, target_dir: str | os.PathLike | None = None) -> Path:
    """Convert the verified pickled checkpoint into the pinned safetensors file, once.

    The source must match the size and SHA-256 pinned in ``MODEL_SPEC``. It is read with
    ``torch.load(..., weights_only=True)`` (which refuses arbitrary pickled objects), reduced to its state dict
    the way mmengine's loader reduces it, checked to carry the class list pinned in `COCO_CLASSES`, and loaded
    into the model built from the pinned config with ``strict=True``. The written file must hash to
    `CONVERTED_SHA256`, or it is deleted and the conversion refused. Returns the converted file's path.
    """
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"conversion source missing: {source_path}")
    size = source_path.stat().st_size
    if size != MODEL_SPEC["checkpoint_size_bytes"] or _sha256(source_path) != MODEL_SPEC["checkpoint_sha256"]:
        raise ValueError(f"{source_path.name} does not match the checkpoint size and SHA-256 pinned in MODEL_SPEC")
    import torch

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    state = checkpoint_state_dict(checkpoint)
    meta = checkpoint.get("meta", {}) if "state_dict" in checkpoint else {}
    if tuple(meta.get("CLASSES", ())) != COCO_CLASSES:
        raise ValueError(f"{source_path.name}: its class list differs from the pinned COCO_CLASSES")
    build_unloaded_model(resolve_packaged_config(), device="cpu").load_state_dict(state, strict=True)
    target = Path(target_dir if target_dir is not None else source_path.parent) / CONVERTED_WEIGHTS_FILE
    partial = target.with_name(target.name + ".partial")
    try:
        _write_converted(state, partial)
        digest = _sha256(partial)
        if digest != CONVERTED_SHA256:
            raise ValueError(f"conversion of {source_path.name} hashed to {digest}, not the pinned {CONVERTED_SHA256}")
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    return verify_converted(target)


def acquire_converted_weights(cache_dir: str | os.PathLike = ".dimer-models") -> Path:
    """Cache path: serve the converted file from ``cache_dir``; fetch and convert the source only if it is absent."""
    target = Path(cache_dir).expanduser().resolve() / CONVERTED_WEIGHTS_FILE
    if target.is_file():
        return verify_converted(target)
    return convert_checkpoint(acquire_verified_checkpoint(cache_dir))


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


INPUT_SCHEMA: dict = {
    "input": "one or more image files readable by Pillow (any mode), given by path",
    "pixels": [1, MAX_PIXELS],
    "score_threshold": [0.0, 1.0],
    "max_detections": [1, None],
    "classes": "80 COCO 2017 categories in MMDetection order",
    "preprocessing": "MMDetection test pipeline of the pinned config (resize, normalise, pad); nothing is altered by this module",
}


def _check_request(score_threshold: float, max_detections: int) -> None:
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1.")
    if max_detections < 1:
        raise ValueError("max_detections must be positive.")


def validate_inputs(
    images: str | os.PathLike | Iterable[str | os.PathLike],
    *,
    score_threshold: float = 0.0,
    max_detections: int = 300,
    names: Iterable[str] | None = None,
) -> dict:
    """Validation stage: return the input manifest (schema, per-image observations, request, verdict).

    Rejection is reported by raising exactly as ``predict`` would (``validate_image`` for each image,
    then the request ceilings); a caller that wants the finding recorded catches the exception and
    stores ``str(exc)`` under ``findings``.
    """
    _check_request(score_threshold, max_detections)
    paths = [images] if isinstance(images, (str, os.PathLike)) else list(images)
    observed = [validate_image(p) for p in paths]
    ids = list(names) if names is not None else [Path(o["path"]).name for o in observed]
    if len(ids) != len(observed):
        raise ValueError("names must have one entry per image")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": ids[i], **o} for i, o in enumerate(observed)],
        "score_threshold": score_threshold,
        "max_detections": max_detections,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    detections: Iterable[Detection | dict],
    ground_truth: Iterable[dict] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``ground_truth`` (one ``{"image_id", "boxes": [{"class_id", "bbox_xyxy"}]}`` entry per evaluated
    image) the report carries ``coco_box_ap`` — the repository's metric helper: COCO AP@[0.50:0.95],
    AP50 and AP75 via pycocotools — with the verdict ``sample-sanity`` and the empty-detector baseline
    (AP 0 by construction). Without it the verdict is ``not-measurable`` (EVAL9) and the report says
    what labelled data would make the task measurable; detection counts are sanity evidence that the
    inference path executed.
    """
    rows = [d.to_dict() if isinstance(d, Detection) else dict(d) for d in detections]
    images = sorted({row["image_id"] for row in rows})
    base = {
        "task": "COCO-80 object detection",
        "score_semantics": "MMDetection class confidence scores; uncalibrated, not probabilities of correctness",
        "decision_threshold": "caller-owned; score_threshold is an output filter, not a deployment threshold",
        "sample_kind": sample_kind,
        "n_images": len(images),
        "n_detections": len(rows),
        "context": {
            "upstream_reported_box_ap": MODEL_SPEC["upstream_reported_box_ap"],
            "note": "upstream full-COCO box AP as reported by OpenMMLab; not measured here",
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if ground_truth is None:
        return {
            **base,
            "metrics": [],
            "baselines": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth boxes were supplied for the evaluated images",
            "needs": (
                "ground-truth boxes (class id + xyxy) for the evaluated images, scored with coco_box_ap "
                "(pycocotools COCO AP@[0.50:0.95], AP50, AP75) against the empty-detector baseline; "
                "a labelled set from the deployment domain for any generalisable claim"
            ),
        }
    ap = coco_box_ap(rows, ground_truth)
    estimation = f"single labelled sample of {ap['n_images']} image(s) / {ap['n_ground_truth_boxes']} boxes, no dispersion estimate"
    return {
        **base,
        "n_images": ap["n_images"],
        "metrics": [
            {"id": "coco_box_ap", "iou": "0.50:0.95", "value": ap["coco_ap_50_95"], "estimation": estimation},
            {"id": "coco_box_ap", "iou": "0.50", "value": ap["ap50"], "estimation": estimation},
            {"id": "coco_box_ap", "iou": "0.75", "value": ap["ap75"], "estimation": estimation},
        ],
        "baselines": [
            {"id": "empty_detector", "coco_box_ap": 0.0, "note": "a detector returning no boxes scores AP 0 by construction"}
        ],
        "verdict": "sample-sanity",
        "reason": f"{ap['n_images']} labelled image(s) with {ap['n_ground_truth_boxes']} ground-truth boxes from the tutorial sample; not a benchmark",
        "needs": "a representative labelled holdout from the deployment domain for any generalisable AP claim",
    }


class DimerSwinDetector:
    """Public task-inference API for the pinned DIMER Swin object detector.

    The upstream `.pth` checkpoint is a pickled PyTorch serialization. The runtime
    verifies its size and SHA-256, reads it once with ``torch.load(..., weights_only=True)``
    and converts it into a safetensors file whose SHA-256 is pinned (`convert_checkpoint`).
    The model is built from the pinned config with no checkpoint and served only from that
    converted file, with ``load_state_dict(strict=True)``; MMDetection's own pickle loader is
    never called. Digest verification establishes byte identity, not author authenticity.
    """

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike = ".dimer-models",
        device: str = "cpu",
        checkpoint: str | os.PathLike | None = None,
        source: str = "openmmlab-cache",
    ):
        versions = verify_runtime_versions()
        checkpoint = verify_converted(checkpoint) if checkpoint is not None else acquire_converted_weights(cache_dir)
        config = resolve_packaged_config()
        from safetensors.torch import load_file

        self.model = build_unloaded_model(config, device=device)
        self.model.load_state_dict(load_file(str(checkpoint)), strict=True)
        self.device = device
        self.checkpoint = checkpoint
        self.config = config
        self.versions = versions
        self.source = source
        self.classes = tuple(self.model.dataset_meta.get("classes", ()))
        if not self.classes:
            raise RuntimeError("MMDetection model did not expose its class ordering.")

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str = "cpu",
        weights_dir: str | os.PathLike | None = None,
        allow_download: bool = False,
    ) -> DimerSwinDetector:
        """Load from the fleet snapshot directory. When the converted safetensors file is absent: stage
        absent manifest entries (only with ``allow_download=True``, from the pinned OpenMMLab URL), re-hash
        every entry against the manifest and MODEL_SPEC, and convert the ``.pth`` once
        (`convert_checkpoint`). Then verify the converted file against its pinned SHA-256 and serve the
        model from it; the pickled source is not read when the conversion already exists.
        """
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        converted = root / CONVERTED_WEIGHTS_FILE
        if not converted.is_file():
            if not (root / MANIFEST_NAME).is_file():
                raise FileNotFoundError(
                    f"no snapshot manifest at {root}; use DimerSwinDetector(cache_dir=...) for the cache path"
                )
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            convert_checkpoint(root / WEIGHTS_FILE, root)
        return cls(device=device, checkpoint=converted, source="local-snapshot")

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
        for bbox, score, label in zip(instances.bboxes, instances.scores, instances.labels, strict=False):
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
                "source": self.source,
                "classes": list(self.classes),
                "checkpoint_path": str(self.checkpoint),
                "checkpoint_sha256": _sha256(self.checkpoint),
                "checkpoint_format": "safetensors (converted once from the pinned .pth with weights_only=True)",
                "source_checkpoint_sha256": MODEL_SPEC["checkpoint_sha256"],
            },
            "score_semantics": "MMDetection class confidence scores; uncalibrated, not probabilities of correctness.",
            "decision_threshold": "caller-owned; score_threshold is an output filter, not a universal deployment threshold.",
        }

    def write_provenance(self, path: str | os.PathLike) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.provenance(), indent=2) + "\n")
        return target
