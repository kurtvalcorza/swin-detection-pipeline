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
    "mode": "GUIDED",
    # SWD-M1/m3 (NOTEBOOK_SPEC 2.2 §5): the qualified OpenMMLab stack needs CPython 3.10, which pip cannot provide. The
    # notebook therefore builds an isolated uv environment with a uv-managed CPython 3.10.18, installs the exact pins of
    # tools/pins.txt there (pins mode: the PyTorch CPU index and the OpenMMLab find-links page are needed, so no hash lock
    # is carried yet), and routes every later cell to a persistent worker in it. The kernel's own Python does not matter.
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.10.18",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": None,
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime (a Python 3.10 Jupyter kernel is the recorded runtime; Google Colab is "
        "expected to work through the isolated environment but no Colab run is recorded yet) builds an isolated environment with a "
        "uv-managed CPython 3.10.18 and the exact pins of `tools/pins.txt` (the kernel's own Python and packages are left alone, so "
        "no restart is needed), stages and digest-verifies the pinned OpenMMLab checkpoint, fetches the digest-pinned labelled COCO8 "
        "validation subset (4 images, 17 ground-truth boxes), validates it into an input manifest before the model runs, detects "
        "objects, writes an evaluation report with COCO box AP against the empty-detector baseline, and exports machine-readable "
        "outputs with provenance. The default path needs no repository clone, no DIMER worker or service, no credential, no upload "
        "dialog, no configuration edit and no runtime restart (NOTEBOOK_SPEC 2.2 §5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in the sample cell — with `BYOD_IMAGE_PATH` pointing at an image "
        "already in the runtime (works in any Jupyter kernel), or empty for the Colab upload dialog (exactly one file) — and re-run "
        "from that cell. Your image passes through the same validation, detection, evaluation-report and export cells; without "
        "ground-truth boxes the report is `not-measurable`. The upload stays inside this runtime. BYOD is optional and never part of "
        "the default path."
    ),
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
            "Open In Colab (verification pending)",
            "https://img.shields.io/badge/Colab-verification%20pending-lightgrey?style=flat&logo=googlecolab",
            "https://colab.research.google.com/github/kurtvalcorza/swin-detection-pipeline/blob/main/tutorials/swin_detection_task_inference.ipynb",
        ),
        (
            "Isolated Python 3.10",
            "https://img.shields.io/badge/Python-3.10%20(isolated%2C%20uv--managed)-3776ab?style=flat&logo=python&logoColor=white",
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
        "The default sample is the public, labelled COCO8 validation subset (4 COCO 2017 images, 17 ground-truth boxes, "
        "pinned by SHA-256), so the notebook measures what it detects: COCO box AP against an empty-detector baseline, a "
        "four-image tutorial metric rather than a benchmark. A synthetic scene drawn in code (no ground truth) remains "
        "available as `SAMPLE = 'synthetic'` to show what the report says when nothing can be measured."
    ),
    "learning_objectives": (
        "install the pinned Python 3.10 OpenMMLab CPU runtime into an isolated environment, read what the carried package "
        "guarantees, resolve and digest-verify the immutable OpenMMLab checkpoint, fetch the labelled COCO8 subset (or "
        "switch to the synthetic scene or a BYOD image) and validate it into an input manifest, run detection through the "
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
    "guided_opening": [
        (
            "## How to use this notebook\n\n"
            "**Who this notebook is for.** Learners who can run notebook cells in order and read short Python, and who want to see "
            "how a pretrained object detector is run, measured honestly on a small labelled sample and exported with provenance. No "
            "prior experience with MMDetection or Swin Transformers is assumed: each term is explained where it is first needed and "
            "again in the glossary below. No GPU is needed. The **Prerequisites** give the details.\n\n"
            "**Running it.** Choose *Run all* (in Colab: *Runtime → Run all*). The default path needs no edit, no upload, no account, "
            "no token and no runtime restart. Section 1 builds the isolated Python 3.10 environment — the torch, MMCV and MMDetection "
            "wheels make it the slowest step — and a second Run all in the same runtime reuses it. You can also run one cell at a time "
            "with *Shift + Enter*.\n\n"
            "**Where the code runs.** The first two code cells run in the notebook kernel: they build the environment and start one "
            "Python 3.10 process inside it. Every later cell is sent to that process, so the OpenMMLab stack runs on the interpreter "
            "its wheels were built for, whatever Python the kernel itself has. Printed output and errors come back to the notebook as "
            "usual, and variables persist from cell to cell.\n\n"
            "**Two kinds of cell.** *Infrastructure cells* (Sections 1–3: the isolated install and router, the carried package and "
            "the checkpoint staging) are collapsed and labelled **Infrastructure**; you may run them without studying them. *Learner "
            "cells* (Sections 4–9) are the detection workflow.\n\n"
            "**Form controls.** The sample cell (Section 5) starts with `USE_BYOD`, `BYOD_IMAGE_PATH` and `SAMPLE`. Leave them at their "
            "defaults for the first run: the notes and sample answers describe the default COCO8 path.\n\n"
            "**Section tags.** Each learner heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation "
            "practice]** — how the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and "
            "packaging.\n\n"
            "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an "
            "expectation; after it, **What to notice** describes normal output and a collapsed **Check your reasoning** block gives a "
            "worked answer. Numbers quoted there come from recorded runs named in the answer; read your own output first."
        ),
        (
            "## The task: Input → Model/System → Output\n\n"
            "| Stage | Input | Model / system | Output |\n"
            "|---|---|---|---|\n"
            "| **Validate** | image paths, `score_threshold`, `max_detections` | `validate_inputs` (decodable, positive size, ≤ 64 MP) | an input manifest |\n"
            "| **Detect** | one RGB image | MMDetection 3.3.0 test pipeline (resize, normalise, pad) → Swin-T backbone → FPN → Mask R-CNN box head → NMS | class-labelled boxes `bbox_xyxy` in pixels with uncalibrated scores, sorted per image |\n"
            "| **Evaluate** | detections + ground-truth boxes | `coco_box_ap` (pycocotools) beside an empty-detector baseline | AP@[.50:.95], AP50, AP75; or `not-measurable` without ground truth |\n"
            "| **Export** | everything above | — | JSON result, CSV detections, provenance |\n\n"
            "## Roadmap\n\n"
            "| Section | Tag | What happens | What you read |\n"
            "|---|---|---|---|\n"
            "| 1. Install the pinned runtime | [Engineering] | isolated Python 3.10 environment; later cells routed there | versions |\n"
            "| 2. Package code | [Engineering] | the repository's package, carried verbatim | nothing to run by hand |\n"
            "| 3. Pin, stage and verify the model | [Engineering] | the checkpoint downloaded and digest-checked | the verified file |\n"
            "| 4. Confirm the qualified runtime | [Engineering] | interpreter and library versions asserted | the versions |\n"
            "| 5. The sample | [Evaluation practice] | COCO8 fetched and verified (or synthetic / BYOD) | images, digests, 17 boxes |\n"
            "| 6. Validate | [Evaluation practice] | input manifest and a refusal probe | the findings |\n"
            "| 7. Detect | [Concept] | boxes, classes, scores | counts and top boxes |\n"
            "| 8. Evaluate | [Evaluation practice] | COCO box AP vs the empty detector | the principal result |\n"
            "| 9. Export | [Engineering] | JSON, CSV and provenance | the file list |\n"
            "| 10. Activity (optional) | [Concept] | switch to the synthetic scene | the `not-measurable` verdict |\n"
            "| Troubleshooting, Glossary, Conclusion | — | recovery, terms, your notes | when needed |\n\n"
            "**Fast path.** Short on time? Run all, then read Sections 7 and 8 and the conclusion."
        ),
        (
            "<details>\n<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
            "| Term | Meaning in this notebook |\n"
            "|---|---|\n"
            "| **Bounding box (`bbox_xyxy`)** | An axis-aligned rectangle `x1, y1, x2, y2` in image pixels around one object. |\n"
            "| **COCO-80** | The 80 object categories of COCO 2017; the detector can name nothing else. |\n"
            "| **Score** | The detector's class confidence for a box after NMS; it ranks boxes but is **uncalibrated**: 0.9 is not a 90 % chance of being right. |\n"
            "| **`score_threshold`** | An output filter the caller chooses; the package ships no deployment threshold. |\n"
            "| **NMS (non-maximum suppression)** | Removes boxes that overlap a higher-scoring box of the same class too much. |\n"
            "| **IoU (intersection over union)** | Overlap of two boxes: the shared area divided by the combined area; 1 is identical. |\n"
            "| **AP@[.50:.95]** | COCO average precision, averaged over IoU thresholds 0.50, 0.55, …, 0.95: how well ranked detections match ground truth. |\n"
            "| **AP50 / AP75** | Average precision at one IoU threshold, 0.50 (loose) or 0.75 (strict). |\n"
            "| **Empty-detector baseline** | A detector that returns nothing: AP 0 by construction, the floor any real detector must clear. |\n"
            "| **Ground truth** | The human-drawn boxes and classes the detections are scored against. |\n"
            "| **YOLO label format** | One line per box: `class x_center y_center width height`, all as fractions of the image size; `boxes_from_yolo_labels` converts it to pixel boxes. |\n"
            "| **`.pth` trust boundary** | The checkpoint is a PyTorch pickle that can run code when loaded; the digest check fixes its bytes, not its author. |\n"
            "| **`not-measurable`** | The report's verdict when no ground truth exists, so no metric can be computed. |\n"
            "| **`sample-sanity`** | The verdict for a metric on a tiny tutorial sample: evidence the path works, not a benchmark. |\n"
            "| **Isolated environment** | A separate Python 3.10 with the exact pins, in which every learner cell runs. |\n"
            "| **BYOD** | Bring Your Own Data: the optional switch that runs the same cells on your image. |\n\n"
            "</details>"
        ),
    ],
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime; the kernel's own Python version does not matter. The qualified OpenMMLab stack — torch 2.1.2 (CPU build), MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.3.0, NumPy 1.26.4 — has prebuilt wheels for Python 3.10 only, so Section 1 has `uv` provision a managed **CPython 3.10.18**, installs the exact pins there, and runs every later cell in that interpreter (Section 4 asserts it). The recorded runtime is a Python 3.10 Jupyter kernel on Linux (GitHub Actions); Google Colab (Python 3.12 kernel) is expected to work the same way but **no Colab run has been recorded yet**. CPU is the default and only qualified path; no GPU is required. The pinned torch/mmcv wheels are the largest downloads of the run.",
        "- **External package indexes:** the isolated install reads the PyTorch CPU index (`download.pytorch.org`) for the `+cpu` torch builds and the OpenMMLab wheel page (`download.openmmlab.com`) for MMCV, besides PyPI; the COCO8 archive comes from `github.com` (Ultralytics assets release).",
        "- **Knowledge:** basic Python and image handling; what a detection score and an IoU-based AP metric are.",
        "- **Data:** the default sample is the public COCO8 validation subset (4 labelled COCO 2017 images, 17 ground-truth boxes, a 443 KB archive from the Ultralytics `assets` release `v0.0.0`, verified against its SHA-256 before path-safe extraction), evaluated with COCO box AP. `SAMPLE = 'synthetic'` switches to a deterministic 640×480 scene generated in code (no download, no ground truth). `USE_BYOD` (off by default) takes one image file decodable by Pillow (PNG/JPEG/WebP and similar, at most 64 megapixels) from `BYOD_IMAGE_PATH`, or from the Colab upload dialog when that field is empty. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Confirm the qualified runtime · [Engineering]\n\n"
                "The carried package fails closed on version drift: `verify_runtime_versions` (called when the model "
                "was constructed above) compares the installed `mmdet`, `mmcv` and `mmengine` distributions with the "
                "versions pinned in `MODEL_SPEC`, and this cell additionally asserts the Python 3.10 interpreter the "
                "OpenMMLab wheels were built for — the isolated environment's interpreter, not the kernel's. Look for a dictionary reporting Python 3.10.x, `torch` 2.1.2+cpu, "
                "MMDetection 3.3.0, MMCV 2.1.0, MMEngine 0.10.7, 80 classes, the verified checkpoint file name and the "
                "`local-snapshot` source."
            ),
            "code": (
                "import sys\n\n"
                "if sys.version_info[:2] != (3, 10):\n"
                "    raise RuntimeError(f'Python 3.10 is required by the qualified OpenMMLab runtime (see Prerequisites); this interpreter is {{sys.version.split()[0]}}. Run the Section 1 cells first: they provide a managed Python 3.10 whatever the kernel runs.')\n"
                "print({{'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__, **pipe.versions, 'classes': len(pipe.classes), 'checkpoint': pipe.checkpoint.name, 'source': pipe.source, 'device': pipe.device}})"
            ),
        },
        {
            "md": (
                "## 5. The sample: COCO8 (default), the synthetic scene, or your image · [Evaluation practice]\n\n"
                "The default sample is **COCO8**: the cell fetches the public COCO8 archive from its pinned release URL, refuses "
                "it unless its SHA-256 equals the recorded digest, extracts only the four validation images and their YOLO-format "
                "labels member by member after path and size checks (no `extractall`), and converts the labels to ground-truth "
                "boxes with the carried `boxes_from_yolo_labels` — the notebook does not resplit or relabel anything. These are "
                "real COCO 2017 photographs with human-drawn boxes, so Section 8 can measure the detector.\n\n"
                "`SAMPLE = 'synthetic'` instead draws a deterministic 640×480 scene in code (a red→green gradient with three "
                "flat-coloured shapes). It depicts no COCO object and has **no ground truth**, so its detections are only a check "
                "that the input contract, preprocessing and forward pass work (Section 10 uses it). `USE_BYOD = True` takes one "
                "image from `BYOD_IMAGE_PATH`, or — in Colab, with the field empty — from the upload dialog, which must receive "
                "exactly one file; BYOD has no ground truth unless you build it yourself. Look for a dictionary naming the sample "
                "kind, the image files, their digests and the number of ground-truth boxes.\n\n"
                "**Predict before running:** how many images and ground-truth boxes does the default sample hold?"
            ),
            "code": (
                "import hashlib\n"
                "import stat\n"
                "import zipfile\n"
                "from pathlib import Path, PurePosixPath\n\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_IMAGE_PATH = ''  # @param {{type:\"string\"}}\n"
                "SAMPLE = 'coco8'  # @param [\"coco8\", \"synthetic\"]\n"
                "SCORE_THRESHOLD = 0.0  # evaluation sees the full detector output; a display cut-off is the caller's choice\n"
                "MAX_DETECTIONS = 300\n"
                "COCO8_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip'\n"
                "COCO8_SHA256 = '54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9'\n"
                "COCO8_MAX_EXPANDED_BYTES = 25 * 1024 * 1024\n"
                "sample_dir = Path('sample')\n"
                "sample_dir.mkdir(exist_ok=True)\n"
                "ground_truth = None\n"
                "if SAMPLE not in ('coco8', 'synthetic'):\n"
                "    raise ValueError(f\"SAMPLE must be 'coco8' or 'synthetic', not {{SAMPLE!r}}\")\n"
                "if USE_BYOD:\n"
                "    if BYOD_IMAGE_PATH:\n"
                "        image_path = Path(BYOD_IMAGE_PATH)\n"
                "        if not image_path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_IMAGE_PATH {{image_path}} is not a file in this runtime; check the path, or leave it empty for the Colab upload dialog.')\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError as exc:\n"
                "            raise RuntimeError('There is no upload dialog outside Google Colab: set BYOD_IMAGE_PATH to an image file in this runtime.') from exc\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one image file (received {{len(uploaded)}}); run the cell again, or set BYOD_IMAGE_PATH.')\n"
                "        upload_name = next(iter(uploaded))\n"
                "        image_path = sample_dir / Path(upload_name).name\n"
                "        image_path.write_bytes(uploaded[upload_name])\n"
                "    image_paths = [image_path]\n"
                "    sample_kind = 'BYOD'\n"
                "elif SAMPLE == 'coco8':\n"
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
                "**What to notice (Section 5):** `sample_kind` is `COCO8-val`, four `.jpg` images with their SHA-256 digests, and "
                "`ground_truth_boxes: 17`.\n\n"
                "<details><summary>Check your reasoning</summary>Four images and 17 boxes: the COCO8 validation split, exactly as "
                "the archive ships it. The digest check on the archive is what makes these the same 17 boxes in every run.</details>\n\n"
                "## 6. Validate the input → input manifest · [Evaluation practice]\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks `predict` "
                "applies — the request ceilings (`score_threshold` in 0..1, `max_detections` ≥ 1) and, per image, "
                "`validate_image` (the file exists, Pillow can decode it, positive dimensions, at most `MAX_PIXELS` = "
                "64,000,000 pixels) — and returns an **input manifest** naming the schema and ceilings, each input's "
                "observed path, size and mode, the request parameters and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a "
                "path that does not exist and records the package's own error message as a finding. Inside the "
                "package every accepted image goes through the pinned config's MMDetection test pipeline (resize, "
                "normalise, pad); nothing is dropped or altered by the package itself.\n\n"
                "**Predict before running:** the probe asks for a file that does not exist. Will validation reject it before the "
                "model is called, and will the message name the file?"
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
                "**What to notice (Section 6):** four accepted inputs with their sizes and modes, and one `rejected` finding for the "
                "missing-file probe.\n\n"
                "<details><summary>Check your reasoning</summary>Yes to both: `validate_inputs` applies the same checks `predict` "
                "applies, before any model work, and the refusal names `does-not-exist.png` and the rule it broke. A refusal that "
                "names the file and the rule is what lets you fix your own input.</details>\n\n"
                "## 7. Detect · [Concept]\n\n"
                "`predict_many` runs `predict` per image through the pinned MMDetection inference path and returns "
                "`Detection` records (`image_id`, `class_id`, `class_name`, `score`, `bbox_xyxy` in pixel "
                "coordinates), **ordered by descending score within each image** and cut at `max_detections`. The "
                "`score` is the MMDetection class confidence after the config's own NMS: it is **uncalibrated**, "
                "not a probability that the box is correct, and the package ships no deployment threshold — "
                "`score_threshold` is an output filter the caller owns (0.0 here so evaluation sees the full "
                "output). Inference is deterministic given the same weights, device and library versions "
                "(`model.eval()`, no sampling); CPU kernel choices can reorder near-tied scores. Look for the "
                "per-image detection counts and the five highest-scoring boxes.\n\n"
                "**Predict before running:** with `score_threshold = 0.0`, will the detector return about 17 boxes (one per "
                "labelled object), or many more? Will the top-scoring boxes be people, animals or vehicles?"
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
                "**What to notice (Section 7):** `n_detections` and the per-image counts, then the five top boxes with class names "
                "and scores.\n\n"
                "<details><summary>Check your reasoning</summary>Many more than 17: at threshold 0.0 the detector keeps every box "
                "that survives NMS, up to `max_detections`, including many low-scoring guesses. That is deliberate — COCO AP ranks "
                "all of them, and a display threshold is the caller's later choice. The top boxes should be confident, plausible "
                "COCO classes for these photographs; their scores rank boxes but are not probabilities.</details>\n\n"
                "## 8. Evaluate → evaluation report · [Evaluation practice]\n\n"
                "`evaluation_report` is the package's public evaluation stage and always produces a report. When "
                "ground-truth boxes exist (the `USE_COCO8` path) it carries `coco_box_ap` — the repository's metric "
                "helper, COCO AP@[0.50:0.95], AP50 and AP75 via `pycocotools` over axis-aligned boxes — with the verdict "
                "`sample-sanity` and the empty-detector baseline (AP 0 by construction) — this is the default path: a four-image tutorial metric "
                "with high sampling variance and no dispersion estimate, not comparable to the upstream full-COCO box "
                "AP of 42.7 that `MODEL_SPEC` records as upstream-reported context. On the synthetic scene "
                "(and on BYOD without labels) no metric exists, so the verdict is `not-measurable` and the report states "
                "what would make the task measurable: ground-truth boxes for the evaluated images, or a labelled "
                "holdout from the deployment domain. The report is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** will AP@[.50:.95] on these four images be near 0, near the upstream 42.7 % (0.427), "
                "or higher? Which will be higher, AP50 or AP75, and why?"
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
                "**What to notice (Section 8):** the verdict `sample-sanity`, the three AP values, the 17 ground-truth boxes over 4 "
                "images, and the empty-detector baseline at 0.\n\n"
                "<details><summary>Check your reasoning</summary>Probably well above 0.427. A recorded run of an earlier carrier of "
                "this notebook on the same four images (GitHub Actions, CPython 3.10, 2026-09-11) gave AP@[.50:.95] 0.707, AP50 0.957 "
                "and AP75 0.699. Four everyday photographs are easier than the full COCO validation set, and four images give a "
                "metric with high sampling variance and no dispersion estimate — it says the detector works on these images, not "
                "that it beats the published number. AP50 is higher than AP75 because a looser overlap requirement accepts more "
                "boxes.</details>\n\n"
                "## 9. Export outputs and provenance · [Engineering]\n\n"
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
        "## 10. Activity: change one thing — a scene with no ground truth · [Concept]\n\n"
        "Optional; **Predict → Change one thing → Run → Observe → Explain**. It changes nothing unless you do it.\n\n"
        "1. **Predict:** on a scene of three flat-coloured shapes on a gradient, what will the detector return, and what will the "
        "evaluation report say?\n"
        "2. **Change:** in Section 5 set `SAMPLE = 'synthetic'`. Change nothing else.\n"
        "3. **Run:** select the Section 5 cell and choose *Runtime → Run after* (or run Sections 5–9 in order). The outputs of "
        "the COCO8 run are replaced; download `outputs/` first if you want to keep them.\n"
        "4. **Observe:** the detection count and top scores in Section 7, and the verdict in Section 8.\n"
        "5. **Explain:** why does the report refuse to give a number here, even though the detector returned boxes?\n\n"
        "<details><summary>Check your reasoning</summary>Few boxes, with low scores, possibly with COCO labels that make no sense "
        "for a rectangle or a circle — the detector always answers in its 80-class vocabulary. The report says `not-measurable`, "
        "because there are no ground-truth boxes to compare with: a score, however high, is not evidence of correctness. A recorded "
        "run of this path (GitHub Actions, 2026-09-14) reported `not-measurable` as expected. Set `SAMPLE = 'coco8'` again to "
        "return to the measured path.</details>\n\n"
        "## Interpretation and limits\n\n"
        "Each detection is a class from the fixed 80-category COCO 2017 label space with an axis-aligned box and an "
        "**uncalibrated** MMDetection score; the package ships no acceptance threshold and `score_threshold` is an output "
        "filter the caller owns. The default `coco_box_ap` value from the four-image COCO8 subset is tutorial evidence for "
        "those images and must not be generalized to a domain, camera, object size distribution or class mix. Objects "
        "outside the COCO categories, crowded or tiny objects, unusual viewpoints, and domain shifts (medical, aerial, "
        "line art) all degrade results in ways the package does not detect. On the synthetic scene the detections are "
        "meaningless by construction and the report says `not-measurable`. The package provides no segmentation, "
        "tracking, keypoint, classification, or training capability, and the `.pth` checkpoint remains a code-capable "
        "serialization whose digest check fixes the bytes, not the author.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned OpenMMLab checkpoint, provision and assert the qualified Python 3.10 / "
        "MMDetection 3.3.0 runtime in an isolated environment, validate the demonstrated input, execute the public detection path, and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority, reproduction of the upstream COCO result, score calibration, safety for high-consequence "
        "decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** run the Section 10 activity; enable `USE_BYOD` with a photograph from your "
        "own domain and inspect the score distribution before choosing a display threshold; label a small holdout from "
        "that domain in the `boxes_from_yolo_labels` format and compare its AP with the COCO8 value.\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a Windows or macOS kernel, or an ARM machine | Use a Linux x86_64 Jupyter kernel or Google Colab; the pinned wheels are manylinux x86_64 builds. |\n"
        "| Section 1 fails while downloading `uv`, or `The pinned uv wheel failed its size/SHA-256 check` | a network failure, or an altered download | Run the Section 1 install cell again; never edit the digest. |\n"
        "| `CalledProcessError` from `uv pip install` in Section 1 (`No solution found`, HTTP errors) | the PyTorch CPU index, the OpenMMLab wheel page or PyPI was unreachable | Run the cell again later; all three hosts must be reachable. Never loosen a pin. |\n"
        "| `holds Python …, not 3.10.18` in Section 1 | an older `dimer_isolated_env/` folder | Delete that folder (or start a fresh runtime) and run the install cell again. |\n"
        "| `Python 3.10 is required …` in Section 4 | the cell ran outside the isolated environment (Section 1 was skipped) | Run Sections 1–3 first, or choose *Run all*. |\n"
        "| The checkpoint download fails in Section 3, or `sha256 … != manifest` | `download.openmmlab.com` unreachable, or a changed file | Run the Section 3 cell again; a digest mismatch is never loaded — do not edit the manifest. |\n"
        "| `RuntimeError` about `mmdet`, `mmcv` or `mmengine` versions | runtime-version drift (a different wheel was installed) | Delete `dimer_isolated_env/` and run Section 1 again; the pins must match `MODEL_SPEC`. |\n"
        "| `COCO8 archive digest … != pinned` in Section 5 | a partial or changed download | Delete `sample/coco8.zip` and run the cell again; a mismatch is never extracted. |\n"
        "| `There is no upload dialog outside Google Colab` / `Upload exactly one image file` | BYOD without a path outside Colab, or a cancelled or multi-file upload | Set `BYOD_IMAGE_PATH`, or run the cell again and choose one file. |\n"
        "| `ValueError` from `validate_inputs` | the image is missing, undecodable or above 64 megapixels | Fix or resize the image; the message names the file and the rule. |\n"
        "| `The isolated environment's Python process exited` | the worker ran out of memory | Use a smaller image, then choose *Run all* again. |\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. On COCO8 (4 images, 17 boxes) the detector reached AP@[.50:.95] ___, AP50 ___ and AP75 ___, against 0 for the empty detector.\n"
        "2. What that number can tell me, and what it cannot: ___.\n"
        "3. On the synthetic scene the report said ___, because ___.\n"
        "4. Before using these scores on my own images I would need ___ (for a threshold) and ___ (for a measurement).\n\n"
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
