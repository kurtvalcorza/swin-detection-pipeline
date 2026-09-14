"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install (from `tools/pins.txt`), the
embedded package modules (`metrics.py`, `runtime.py`), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "dimer_swin_detection",
    "repo_name": "swin-detection-pipeline",
    "stem": "swin_detection_task_inference",
    "notebook_name": "swin_detection_task_inference.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "DimerSwinDetector",
    "weights_key": "swin-t-mask-rcnn-coco",
    "modules": ["metrics.py", "runtime.py"],
    "entry_module": "runtime.py",
    "pins_file": "tools/pins.txt",
    "model_host": {
        "name": "the OpenMMLab checkpoint host (`download.openmmlab.com`)",
        "reference_url": "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth",
        "revision_label": "MMDetection release-tag commit",
    },
    "runtime_imports": ["torch", "numpy"],
    "title": "Swin-T + Mask R-CNN (COCO) — DIMER object-detection tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/swin-detection-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/swin-detection-pipeline/blob/main/tutorials/swin_detection_task_inference.ipynb",
        ),
        (
            "Python 3.10 required",
            "https://img.shields.io/badge/Python-3.10%20required-3776ab?style=flat&logo=python&logoColor=white",
            "https://github.com/kurtvalcorza/swin-detection-pipeline/blob/main/README.md",
        ),
        (
            "Checkpoint",
            "https://img.shields.io/badge/OpenMMLab-mask--rcnn__swin--t--p4--w7__fpn__1x__coco-ffcc4d?style=flat",
            "https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/swin",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-microsoft%2FSwin--Transformer-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/microsoft/Swin-Transformer",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2103.14030-b31b1b.svg", "https://arxiv.org/abs/2103.14030"),
    ],
    "capability": "pretrained COCO-80 object detection (class-labelled axis-aligned boxes with uncalibrated scores) using the pinned OpenMMLab `mask-rcnn_swin-t-p4-w7_fpn_1x_coco` checkpoint through the repository's `DimerSwinDetector` API",
    "intro": (
        "At inference the pinned Swin-T backbone + Mask R-CNN head maps one RGB image to class-labelled boxes and "
        "class scores over the 80 COCO 2017 categories; the DIMER v1 contract exposes boxes, classes and scores only "
        "(the checkpoint's instance masks are not part of the public API). **No adaptation occurs:** no training, "
        "fine-tuning, in-context conditioning, or preprocessing fitting happens in this notebook — the upstream "
        "OpenMMLab checkpoint supplies the weights and the pinned MMDetection 3.3.0 package supplies the config and "
        "test pipeline, and the carried package adds snapshot verification, runtime-version checks, input validation, "
        "a fixed output contract and the `coco_box_ap`, `validate_inputs` and `evaluation_report` helpers.\n\n"
        "**Trust boundary (MOD12).** The checkpoint is a code-capable PyTorch `.pth` serialization. The carried "
        "`verify_snapshot` re-hashes it against the inline manifest (and against the digest the package has always "
        "pinned in `MODEL_SPEC`) before the pinned MMDetection loader deserializes it inside `mmengine`; the "
        "deserialization call is upstream's, not the package's, and is **not** a `weights_only` load. A matching "
        "digest proves byte identity with the pinned OpenMMLab distribution, not publisher authenticity — run this "
        "notebook only where that pinned source is trusted.\n\n"
        "The default sample is a synthetic scene generated in code (no download, no ground truth), so its detections "
        "are demonstration (plumbing) evidence, not a correctness or benchmark claim. A gated option fetches the "
        "public labelled COCO8 validation subset instead and evaluates COCO AP on it."
    ),
    "learning_objectives": (
        "install the pinned Python 3.10 OpenMMLab CPU runtime, read what the carried package guarantees, resolve and "
        "digest-verify the immutable OpenMMLab checkpoint, generate a synthetic default input (or opt into the "
        "labelled COCO8 subset / a BYOD image) and validate it into an input manifest, run detection through the "
        "public API, read uncalibrated class scores and the caller-owned `score_threshold` correctly, produce an "
        "evaluation report that is `sample-sanity` with `coco_box_ap` only when ground-truth boxes exist and "
        "`not-measurable` otherwise, and export machine-readable detections plus provenance."
    ),
    "exclusions": (
        "instance or semantic segmentation (the checkpoint's mask head is outside the DIMER v1 contract), "
        "open-vocabulary detection, tracking, keypoints, image classification, or any training or fine-tuning. The "
        "label space is fixed to the 80 COCO 2017 categories; objects outside that space are either missed or "
        "assigned a COCO label."
    ),
    "prerequisites": [
        "- **Runtime:** a **CPython 3.10** Jupyter kernel on Linux (the notebook asserts `sys.version_info[:2] == (3, 10)` and stops otherwise). The qualified OpenMMLab stack — torch 2.1.2 (CPU build), MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.3.0, NumPy 1.26.4 — has prebuilt wheels for Python 3.10 only; `pip` cannot change the interpreter, so a Python 3.11+ kernel (including current default Colab runtimes) is unsupported and the pinned install fails there. CPU is the default and only qualified path; no GPU is required. The pinned torch/mmcv wheels are the largest downloads of the run.",
        "- **Knowledge:** basic Python and image handling; what a detection score and an IoU-based AP metric are.",
        "- **Data:** the default sample is a deterministic 640×480 synthetic scene (gradient background plus flat-coloured shapes) generated in code, so nothing is downloaded and there is no ground truth. Two optional gates are off by default so the sample path runs top-to-bottom without interaction: `USE_COCO8` fetches the public COCO8 validation subset (4 labelled COCO 2017 images, a 443 KB archive from the Ultralytics `assets` release `v0.0.0`, verified against its SHA-256 before path-safe extraction) and enables COCO AP evaluation; `USE_BYOD` uploads one image file decodable by Pillow (PNG/JPEG/WebP and similar, at most 64 megapixels). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Confirm the qualified runtime\n\n"
                "The carried package fails closed on version drift: `verify_runtime_versions` (called when the model "
                "was constructed above) compares the installed `mmdet`, `mmcv` and `mmengine` distributions with the "
                "versions pinned in `MODEL_SPEC`, and this cell additionally asserts the Python 3.10 interpreter the "
                "OpenMMLab wheels were built for. Look for a dictionary reporting Python 3.10.x, `torch` 2.1.2+cpu, "
                "MMDetection 3.3.0, MMCV 2.1.0, MMEngine 0.10.7, 80 classes, the verified checkpoint file name and the "
                "`local-snapshot` source."
            ),
            "code": (
                "import sys\n\n"
                "if sys.version_info[:2] != (3, 10):\n"
                "    raise RuntimeError(f'Python 3.10 is required by the qualified OpenMMLab runtime (see Prerequisites); this kernel is {{sys.version.split()[0]}}. Use a Python 3.10 kernel.')\n"
                "print({{'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__, **pipe.versions, 'classes': len(pipe.classes), 'checkpoint': pipe.checkpoint.name, 'source': pipe.source, 'device': pipe.device}})"
            ),
        },
        {
            "md": (
                "## 5. Generate the synthetic sample, or opt into COCO8 / BYOD\n\n"
                "The default sample is **synthetic**: a deterministic 640×480 scene (a red→green gradient background "
                "with three flat-coloured shapes) drawn in code and written to `sample/`, so it needs no download and "
                "its SHA-256 is printed for the record. It depicts no COCO object, so it has **no ground truth**: "
                "whatever the detector returns is a sanity check that the input contract, preprocessing and forward "
                "pass work, not a correctness measurement. Two gates are off by default. `USE_COCO8` fetches the "
                "public COCO8 archive from its pinned release URL, refuses it unless its SHA-256 equals the recorded "
                "digest, extracts only the four validation images and their YOLO-format labels member by member "
                "after path and size checks (no `extractall`), and converts the labels to ground-truth boxes with "
                "the carried `boxes_from_yolo_labels` — the notebook does not resplit or relabel anything. "
                "`USE_BYOD` uploads one image; BYOD has no ground truth unless you build it yourself. Look for a "
                "dictionary naming the sample kind, the image files, their digests and whether ground truth exists."
            ),
            "code": (
                "import hashlib\n"
                "import stat\n"
                "import zipfile\n"
                "from pathlib import Path, PurePosixPath\n\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "USE_COCO8 = False  # @param {{type:\"boolean\"}}\n"
                "SCORE_THRESHOLD = 0.0  # evaluation sees the full detector output; a display cut-off is the caller's choice\n"
                "MAX_DETECTIONS = 300\n"
                "COCO8_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip'\n"
                "COCO8_SHA256 = '54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9'\n"
                "COCO8_MAX_EXPANDED_BYTES = 25 * 1024 * 1024\n"
                "sample_dir = Path('sample')\n"
                "sample_dir.mkdir(exist_ok=True)\n"
                "ground_truth = None\n"
                "if USE_BYOD and USE_COCO8:\n"
                "    raise ValueError('Enable at most one of USE_BYOD and USE_COCO8.')\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    upload_name = next(iter(uploaded))\n"
                "    image_path = sample_dir / Path(upload_name).name\n"
                "    image_path.write_bytes(uploaded[upload_name])\n"
                "    image_paths = [image_path]\n"
                "    sample_kind = 'BYOD'\n"
                "elif USE_COCO8:\n"
                "    import urllib.request\n\n"
                "    archive = sample_dir / 'coco8.zip'\n"
                "    if not archive.exists():\n"
                "        urllib.request.urlretrieve(COCO8_URL, archive)\n"
                "    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()\n"
                "    if archive_sha256 != COCO8_SHA256:\n"
                "        raise RuntimeError(f'COCO8 archive digest {{archive_sha256}} != pinned {{COCO8_SHA256}}; refusing to extract')\n"
                "    with zipfile.ZipFile(archive) as zf:\n"
                "        members = zf.infolist()\n"
                "        expanded = 0\n"
                "        for info in members:\n"
                "            member = PurePosixPath(info.filename)\n"
                "            if member.is_absolute() or '..' in member.parts or '\\\\' in info.filename:\n"
                "                raise ValueError(f'unsafe archive member: {{info.filename}}')\n"
                "            if stat.S_ISLNK(info.external_attr >> 16):\n"
                "                raise ValueError(f'symlink refused: {{info.filename}}')\n"
                "            expanded += info.file_size\n"
                "            if expanded > COCO8_MAX_EXPANDED_BYTES:\n"
                "                raise ValueError('archive exceeds the expanded-size ceiling')\n"
                "        for info in members:\n"
                "            if info.filename.startswith(('coco8/images/val/', 'coco8/labels/val/')) and not info.is_dir():\n"
                "                zf.extract(info, sample_dir)\n"
                "    coco8 = sample_dir / 'coco8'\n"
                "    image_paths = sorted((coco8 / 'images' / 'val').glob('*.jpg'))\n"
                "    if not image_paths:\n"
                "        raise RuntimeError('COCO8 validation images were not found after extraction.')\n"
                "    ground_truth = []\n"
                "    for path in image_paths:\n"
                "        with Image.open(path) as opened:\n"
                "            width, height = opened.size\n"
                "        label_text = (coco8 / 'labels' / 'val' / f'{{path.stem}}.txt').read_text(encoding='utf-8')\n"
                "        ground_truth.append({{'image_id': path.name, 'boxes': boxes_from_yolo_labels(label_text, width, height)}})\n"
                "    sample_kind = 'COCO8-val'\n"
                "else:\n"
                "    # Deterministic synthetic scene: no randomness, so no seed is needed and the digest is stable.\n"
                "    width, height = 640, 480\n"
                "    ramp = numpy.linspace(0.0, 255.0, width)\n"
                "    red = numpy.tile(ramp, (height, 1))\n"
                "    green = numpy.tile(numpy.linspace(0.0, 255.0, height)[:, None], (1, width))\n"
                "    blue = (red + green) / 2.0\n"
                "    array = numpy.rint(numpy.stack([red, green, blue], axis=-1)).astype(numpy.uint8)\n"
                "    scene = Image.fromarray(array, mode='RGB')\n"
                "    draw = ImageDraw.Draw(scene)\n"
                "    draw.rectangle([60, 300, 260, 440], fill=(20, 20, 20))\n"
                "    draw.ellipse([380, 80, 560, 260], fill=(240, 240, 240))\n"
                "    draw.polygon([(320, 460), (400, 330), (480, 460)], fill=(30, 90, 200))\n"
                "    image_path = sample_dir / 'synthetic_scene_640x480.png'\n"
                "    scene.save(image_path)\n"
                "    image_paths = [image_path]\n"
                "    sample_kind = 'synthetic'\n"
                "image_names = [path.name for path in image_paths]\n"
                "sample_sha256 = {{path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in image_paths}}\n"
                "n_ground_truth_boxes = None if ground_truth is None else sum(len(entry['boxes']) for entry in ground_truth)\n"
                "print({{'sample_kind': sample_kind, 'images': image_names, 'sha256': sample_sha256, 'ground_truth_boxes': n_ground_truth_boxes}})"
            ),
        },
        {
            "md": (
                "## 6. Validate the input → input manifest\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks `predict` "
                "applies — the request ceilings (`score_threshold` in 0..1, `max_detections` ≥ 1) and, per image, "
                "`validate_image` (the file exists, Pillow can decode it, positive dimensions, at most `MAX_PIXELS` = "
                "64,000,000 pixels) — and returns an **input manifest** naming the schema and ceilings, each input's "
                "observed path, size and mode, the request parameters and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a "
                "path that does not exist and records the package's own error message as a finding. Inside the "
                "package every accepted image goes through the pinned config's MMDetection test pipeline (resize, "
                "normalise, pad); nothing is dropped or altered by the package itself."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MAX_PIXELS': MAX_PIXELS, 'score_threshold': INPUT_SCHEMA['score_threshold'], 'max_detections': INPUT_SCHEMA['max_detections'], 'classes': len(pipe.classes)}}}})\n"
                "input_manifest = validate_inputs(image_paths, score_threshold=SCORE_THRESHOLD, max_detections=MAX_DETECTIONS, names=image_names)\n"
                "# Demonstrate rejection on an input that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(sample_dir / 'does-not-exist.png')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'missing-file-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 7. Detect\n\n"
                "`predict_many` runs `predict` per image through the pinned MMDetection inference path and returns "
                "`Detection` records (`image_id`, `class_id`, `class_name`, `score`, `bbox_xyxy` in pixel "
                "coordinates), **ordered by descending score within each image** and cut at `max_detections`. The "
                "`score` is the MMDetection class confidence after the config's own NMS: it is **uncalibrated**, "
                "not a probability that the box is correct, and the package ships no deployment threshold — "
                "`score_threshold` is an output filter the caller owns (0.0 here so evaluation sees the full "
                "output). Inference is deterministic given the same weights, device and library versions "
                "(`model.eval()`, no sampling); CPU kernel choices can reorder near-tied scores. Look for the "
                "per-image detection counts and the five highest-scoring boxes; on the synthetic scene expect few "
                "or low-scoring detections."
            ),
            "code": (
                "detections = pipe.predict_many(image_paths, score_threshold=SCORE_THRESHOLD, max_detections=MAX_DETECTIONS)\n"
                "rows = [detection.to_dict() for detection in detections]\n"
                "counts = {{name: sum(1 for row in rows if row['image_id'] == name) for name in image_names}}\n"
                "print({{'n_detections': len(rows), 'per_image': counts, 'score_threshold': SCORE_THRESHOLD, 'max_detections': MAX_DETECTIONS}})\n"
                "for rank, row in enumerate(sorted(rows, key=lambda row: row['score'], reverse=True)[:5], start=1):\n"
                "    print(f\"{{rank:>2}}. {{row['image_id']:<24}} class {{row['class_id']:>2}} {{row['class_name']:<14}} score {{row['score']:.4f}}  bbox_xyxy {{[round(v, 1) for v in row['bbox_xyxy']]}}\")"
            ),
        },
        {
            "md": (
                "## 8. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the package's public evaluation stage and always produces a report. When "
                "ground-truth boxes exist (the `USE_COCO8` path) it carries `coco_box_ap` — the repository's metric "
                "helper, COCO AP@[0.50:0.95], AP50 and AP75 via `pycocotools` over axis-aligned boxes — with the verdict "
                "`sample-sanity` and the empty-detector baseline (AP 0 by construction): a four-image tutorial metric "
                "with high sampling variance and no dispersion estimate, not comparable to the upstream full-COCO box "
                "AP of 42.7 that `MODEL_SPEC` records as upstream-reported context. On the synthetic default sample "
                "(and on BYOD without labels) no metric exists, so the verdict is `not-measurable` and the report states "
                "what would make the task measurable: ground-truth boxes for the evaluated images, or a labelled "
                "holdout from the deployment domain. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(detections, ground_truth, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No ground-truth boxes were supplied, so coco_box_ap is not computed; the detections above are sanity evidence only.')"
            ),
        },
        {
            "md": (
                "## 9. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves every detection (score-ordered per image), the evaluation report, the "
                "input manifest, the sample identity and digests, the notebook's source (repository, revision, embedded "
                "module digest, generator), the model identifier, the immutable revision label, the checkpoint digest "
                "the package verified, and the runtime identity (Python, `torch`, `mmdet`, `mmcv`, `mmengine`, device). "
                "The detections are also written as CSV with explicit `rank`, `class_id`, `class_name`, `score` and "
                "`x1,y1,x2,y2` columns so score ordering and pixel coordinates survive downstream use. No credentials "
                "are recorded."
            ),
            "code": (
                "import csv\n\n"
                "payload = {{\n"
                "    'detections': rows,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'images': image_names, 'sha256': sample_sha256, 'ground_truth_boxes': n_ground_truth_boxes}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'checkpoint_sha256': MODEL_SPEC['checkpoint_sha256'],\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'numpy': numpy.__version__,\n"
                "        **pipe.versions,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_detections.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image_id', 'rank', 'class_id', 'class_name', 'score', 'x1', 'y1', 'x2', 'y2'])\n"
                "    for name in image_names:\n"
                "        for rank, row in enumerate([row for row in rows if row['image_id'] == name], start=1):\n"
                "            writer.writerow([name, rank, row['class_id'], row['class_name'], f\"{{row['score']:.6f}}\", *[f'{{v:.2f}}' for v in row['bbox_xyxy']]])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Each detection is a class from the fixed 80-category COCO 2017 label space with an axis-aligned box and an "
        "**uncalibrated** MMDetection score; the package ships no acceptance threshold and `score_threshold` is an output "
        "filter the caller owns. On the synthetic scene the detections are meaningless by construction and the evaluation "
        "report says `not-measurable`; a `coco_box_ap` value from the four-image COCO8 subset is tutorial evidence for "
        "those images and must not be generalized to a domain, camera, object size distribution or class mix. Objects "
        "outside the COCO categories, crowded or tiny objects, unusual viewpoints, and domain shifts (medical, aerial, "
        "line art) all degrade results in ways the package does not detect. The package provides no segmentation, "
        "tracking, keypoint, classification, or training capability, and the `.pth` checkpoint remains a code-capable "
        "serialization whose digest check fixes the bytes, not the author.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned OpenMMLab checkpoint, assert the qualified Python 3.10 / MMDetection 3.3.0 "
        "runtime, validate the demonstrated input, execute the public detection path, and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority, reproduction of the upstream COCO result, score calibration, safety for high-consequence "
        "decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_COCO8` to see the report switch to `sample-sanity` with `coco_box_ap` at "
        "IoU 0.50:0.95, 0.50 and 0.75 against the empty-detector baseline; enable `USE_BYOD` with a photograph from your "
        "own domain and inspect the score distribution before choosing a display threshold; label a small holdout from "
        "that domain in the `boxes_from_yolo_labels` format and compare its AP with the COCO8 value.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/swin-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/swin-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/swin-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Pinned checkpoint (OpenMMLab host): https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth\n"
        "- Config source (MMDetection v3.3.0, `configs/swin`): https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/swin\n"
        "- COCO8 sample (Ultralytics assets release): https://github.com/ultralytics/assets/releases/tag/v0.0.0\n"
        "- Upstream project: https://github.com/microsoft/Swin-Transformer\n"
        "- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows: https://arxiv.org/abs/2103.14030\n"
        "- Mask R-CNN: https://arxiv.org/abs/1703.06870\n"
        "- Microsoft COCO: Common Objects in Context: https://arxiv.org/abs/1405.0312"
    ),
}
