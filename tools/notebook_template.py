"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install (from `tools/pins.txt`), the
embedded package modules (`metrics.py`, `runtime.py`), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

The default scored sample (R86-D1/D2/D3, 2026-10-18) is the pinned COCO val2017 subset in
`tools/coco_val2017_sample_50.json`: 50 images chosen by a fixed licence/overlap rule, mirrored as assets of
this repository's GitHub release `data-coco-val2017-50-v1`, with per-image SHA-256 and inline ground truth.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering
import json
from pathlib import Path

_PIN = json.loads(Path(__file__).with_name("coco_val2017_sample_50.json").read_text(encoding="utf-8"))
_LICENCE_SHORT = {
    "Attribution License": "CC BY 2.0",
    "Attribution-ShareAlike License": "CC BY-SA 2.0",
    "No known copyright restrictions": "No known copyright restrictions",
    "United States Government Work": "United States Government Work",
}
RELEASE_TAG = _PIN["release_tag"]
RELEASE_URL = f"https://github.com/kurtvalcorza/swin-detection-pipeline/releases/download/{RELEASE_TAG}/"
N_IMAGES = _PIN["n_images"]
N_BOXES = _PIN["n_boxes"]
N_CATEGORIES = _PIN["n_categories"]
SAMPLE_MB = _PIN["total_bytes"] / 1e6
assert all(image["release_url"] == RELEASE_URL + image["file_name"] for image in _PIN["images"])
_LICENCE_COUNTS = {}
for _image in _PIN["images"]:
    _short = _LICENCE_SHORT[_image["licence"]]
    _LICENCE_COUNTS[_short] = _LICENCE_COUNTS.get(_short, 0) + 1
LICENCE_SUMMARY = ", ".join(f"{count} {name}" for name, count in sorted(_LICENCE_COUNTS.items()))

# The pinned sample, rendered as plain Python literals (tuples and lists only, so `str.format` needs no escaping).
VAL2017_BLOCK = (
    f"VAL2017_RELEASE_URL = {RELEASE_URL!r}  # this repository's release mirror; the notebook downloads from here\n"
    "VAL2017_COCO_URL = 'http://images.cocodataset.org/val2017/'  # original source, recorded for provenance only\n"
    f"VAL2017_GT_SHA256 = {_PIN['gt_digest_sha256']!r}\n"
    "VAL2017_TRAINING_OVERLAP = [('checkpoint_training_split', 'train2017'), ('evaluation_split', 'val2017'), ('eval_data_in_training', False), "
    "('note', 'val2017 is the upstream benchmark split: OpenMMLab reports box AP 42.7 on it, and it may have been used for model selection. "
    "The images were screened against train2017 by image id, Flickr photo id and (width, height, capture time), not by pixels.')]\n"
    "# One row per image: (file, SHA-256, bytes, image licence, Flickr source, ground-truth boxes as [class index 0-79, x1, y1, x2, y2] in pixels)\n"
    "VAL2017_IMAGES = [\n"
    + "".join(
        "    " + repr((image["file_name"], image["sha256"], image["bytes"], _LICENCE_SHORT[image["licence"]], image["flickr_url"], image["boxes"])) + ",\n"
        for image in _PIN["images"]
    )
    + "]\n"
)


def _c(code: str) -> str:
    """A code cell written as plain Python: braces are escaped for `str.format`, `__STEM__` becomes `{stem}`."""
    return code.replace("{", "{{").replace("}", "}}").replace("__STEM__", "{stem}")


def _guarded(section: str, code: str) -> str:
    """Run `code` only when the sample cell accepted an input (DAT20); otherwise say why the section is skipped."""
    body = "".join(("    " + line if line else line) for line in code.splitlines(keepends=True))
    return (
        "if not sample_ready:\n"
        f"    print('Section {section} skipped: the sample cell (Section 5) did not accept an input; see its message, fix the input and run from Section 5 again.')\n"
        "else:\n" + body
    )


SAMPLE_CELL = (
    "import hashlib\n"
    "import json\n"
    "import os\n"
    "import shutil\n"
    "import time\n"
    "import urllib.request\n"
    "from collections import Counter\n"
    "from pathlib import Path\n\n"
    "from PIL import Image, ImageDraw\n\n"
    "USE_BYOD = False  # @param {type:\"boolean\"}\n"
    "BYOD_IMAGE_PATH = ''  # @param {type:\"string\"}\n"
    "SAMPLE = 'val2017'  # @param [\"val2017\", \"synthetic\"]\n"
    "SCORE_THRESHOLD = 0.0  # evaluation sees the full detector output; a display cut-off is the caller's choice\n"
    "MAX_DETECTIONS = 300  # kept per image in the exports; COCO AP itself scores only the 100 highest-scoring per image (pycocotools maxDets)\n"
    "DEFAULT_SAMPLE_KIND = 'COCO-val2017-50'\n"
    + VAL2017_BLOCK
    + "OUTPUT_DIR = Path('outputs')\n"
    "sample_dir = Path('sample')\n"
    "os.makedirs('outputs', exist_ok=True)\n"
    "sample_dir.mkdir(exist_ok=True)\n"
    "for leftover in [*OUTPUT_DIR.glob('*.part'), *sample_dir.glob('*.part')]:\n"
    "    leftover.unlink()  # the temporary file of a write that was interrupted before its rename\n\n\n"
    "def write_text_atomic(path, text):\n"
    "    \"\"\"Write to a temporary file, then rename it into place: a killed write never leaves a truncated file.\"\"\"\n"
    "    path = Path(path)\n"
    "    temporary = path.with_name(path.name + '.part')\n"
    "    temporary.write_text(text, encoding='utf-8', newline='')\n"
    "    os.replace(temporary, path)\n\n\n"
    "def write_json_atomic(path, payload):\n"
    "    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))\n\n\n"
    "def fetch_pinned(url, destination, sha256, size):\n"
    "    \"\"\"Download `url` unless `destination` already holds the pinned bytes; refuse any other bytes.\"\"\"\n"
    "    if destination.is_file() and destination.stat().st_size == size and hashlib.sha256(destination.read_bytes()).hexdigest() == sha256:\n"
    "        return 'cached'\n"
    "    request = urllib.request.Request(url, headers={'User-Agent': 'dimer-tutorial-notebook'})\n"
    "    error = None\n"
    "    for attempt in range(3):\n"
    "        try:\n"
    "            with urllib.request.urlopen(request, timeout=60) as response:\n"
    "                data = response.read(size + 1)\n"
    "            break\n"
    "        except OSError as exc:  # network errors are retried twice\n"
    "            error = exc\n"
    "            time.sleep(2 * (attempt + 1))\n"
    "    else:\n"
    "        raise RuntimeError(f'Could not download {url} ({error}); check the connection and run this cell again.')\n"
    "    digest = hashlib.sha256(data).hexdigest()\n"
    "    if len(data) != size or digest != sha256:\n"
    "        raise RuntimeError(f'{destination.name}: got {len(data)} bytes with SHA-256 {digest}, but the pin is {size} bytes with {sha256}; refusing to use it. Run the cell again; never edit the pin.')\n"
    "    temporary = destination.with_name(destination.name + '.part')\n"
    "    temporary.write_bytes(data)\n"
    "    os.replace(temporary, destination)\n"
    "    return 'downloaded'\n\n\n"
    "def move_outputs_aside(reason):\n"
    "    \"\"\"Move every earlier output file into outputs/earlier_<n>/, so none can be read as a result for a refused input.\"\"\"\n"
    "    earlier = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file())\n"
    "    if not earlier:\n"
    "        return None\n"
    "    n = 1\n"
    "    while (OUTPUT_DIR / f'earlier_{n}').exists():\n"
    "        n += 1\n"
    "    target = OUTPUT_DIR / f'earlier_{n}'\n"
    "    target.mkdir()\n"
    "    for path in earlier:\n"
    "        shutil.move(str(path), str(target / path.name))\n"
    "    (target / 'WHY.txt').write_text(reason + '\\n', encoding='utf-8')\n"
    "    return target\n\n\n"
    "class SampleRefused(Exception):\n"
    "    \"\"\"An input this notebook cannot use; reported as one message, not a traceback.\"\"\"\n\n\n"
    "ground_truth = None\n"
    "sample_licences = None\n"
    "sample_ready = False\n"
    "if SAMPLE not in ('val2017', 'synthetic'):\n"
    "    raise ValueError(f\"SAMPLE must be 'val2017' or 'synthetic', not {SAMPLE!r}\")\n"
    "if USE_BYOD:\n"
    "    try:\n"
    "        if BYOD_IMAGE_PATH:\n"
    "            image_path = Path(BYOD_IMAGE_PATH)\n"
    "            if not image_path.is_file():\n"
    "                raise SampleRefused(f'BYOD_IMAGE_PATH {str(image_path)!r} is not a file in this runtime. Check the path, or leave it empty to use the Colab upload dialog.')\n"
    "        else:\n"
    "            try:\n"
    "                from google.colab import files\n"
    "            except ImportError:\n"
    "                raise SampleRefused('There is no upload dialog outside Google Colab. Set BYOD_IMAGE_PATH to an image file in this runtime and run this cell again.') from None\n"
    "            uploaded = files.upload()\n"
    "            if len(uploaded) != 1:\n"
    "                raise SampleRefused(f'Upload exactly one image file (received {len(uploaded)}). Run this cell again and choose one file, or set BYOD_IMAGE_PATH.')\n"
    "            upload_name = next(iter(uploaded))\n"
    "            image_path = sample_dir / Path(upload_name).name\n"
    "            image_path.write_bytes(uploaded[upload_name])\n"
    "        try:\n"
    "            validate_inputs([image_path], score_threshold=SCORE_THRESHOLD, max_detections=MAX_DETECTIONS, names=[image_path.name])\n"
    "        except ValueError as exc:\n"
    "            raise SampleRefused(f'{image_path.name} cannot be used: {exc} Use one PNG, JPEG or WebP image of at most 64 megapixels and run this cell again.') from None\n"
    "    except SampleRefused as refusal:\n"
    "        moved = move_outputs_aside(f'Moved aside when a BYOD input was refused: {refusal}')\n"
    "        print(f'BYOD input refused: {refusal}')\n"
    "        print('Nothing was run on it; Sections 6-9 skip until this cell accepts an input.' + (f' Earlier outputs were moved to {moved}/ so they cannot be read as results for this input.' if moved else ''))\n"
    "    else:\n"
    "        image_paths = [image_path]\n"
    "        sample_kind = 'BYOD'\n"
    "        training_overlap = {'checkpoint_training_split': 'train2017', 'eval_data_in_training': 'unknown', 'note': 'your own image; not checked against COCO train2017'}\n"
    "        sample_ready = True\n"
    "elif SAMPLE == 'val2017':\n"
    "    image_paths = []\n"
    "    for name, sha256, size, licence, flickr_url, boxes in VAL2017_IMAGES:\n"
    "        fetch_pinned(VAL2017_RELEASE_URL + name, sample_dir / name, sha256, size)\n"
    "        image_paths.append(sample_dir / name)\n"
    "    gt_digest = hashlib.sha256(json.dumps([[name, boxes] for name, _, _, _, _, boxes in VAL2017_IMAGES], separators=(',', ':')).encode()).hexdigest()\n"
    "    if gt_digest != VAL2017_GT_SHA256:\n"
    "        raise RuntimeError(f'ground-truth digest {gt_digest} != pinned {VAL2017_GT_SHA256}; the inline boxes were changed')\n"
    "    ground_truth = [{'image_id': name, 'boxes': [{'class_id': int(box[0]), 'bbox_xyxy': [float(v) for v in box[1:]]} for box in boxes]} for name, _, _, _, _, boxes in VAL2017_IMAGES]\n"
    "    sample_licences = [{'image': name, 'licence': licence, 'flickr_url': flickr_url, 'coco_url': VAL2017_COCO_URL + name} for name, _, _, licence, flickr_url, _ in VAL2017_IMAGES]\n"
    "    sample_kind = DEFAULT_SAMPLE_KIND\n"
    "    training_overlap = dict(VAL2017_TRAINING_OVERLAP)\n"
    "    sample_ready = True\n"
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
    "    training_overlap = {'checkpoint_training_split': 'train2017', 'eval_data_in_training': False, 'note': 'generated in code; no ground truth'}\n"
    "    sample_ready = True\n"
    "if sample_ready:\n"
    "    # S1: only the default sample writes the default output names; any other pass gets the next free _exploratory_<n> suffix.\n"
    "    OUTPUT_SUFFIX = ''\n"
    "    if sample_kind != DEFAULT_SAMPLE_KIND:\n"
    "        n = 1\n"
    "        while any(OUTPUT_DIR.glob(f'*_exploratory_{n}.*')):\n"
    "            n += 1\n"
    "        OUTPUT_SUFFIX = f'_exploratory_{n}'\n\n"
    "    def output_path(default_name):\n"
    "        path = Path(default_name)\n"
    "        return path.with_name(path.stem + OUTPUT_SUFFIX + path.suffix)\n\n"
    "    image_names = [path.name for path in image_paths]\n"
    "    sample_sha256 = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in image_paths}\n"
    "    n_ground_truth_boxes = None if ground_truth is None else sum(len(entry['boxes']) for entry in ground_truth)\n"
    "    print({\n"
    "        'sample_kind': sample_kind,\n"
    "        'images': len(image_names),\n"
    "        'first_images': {name: sample_sha256[name][:16] + '...' for name in image_names[:3]},\n"
    "        'ground_truth_boxes': n_ground_truth_boxes,\n"
    "        'categories': None if ground_truth is None else len({box['class_id'] for entry in ground_truth for box in entry['boxes']}),\n"
    "        'image_licences': None if sample_licences is None else dict(Counter(entry['licence'] for entry in sample_licences)),\n"
    "        'eval_data_in_training': training_overlap['eval_data_in_training'],\n"
    "        'output_suffix': OUTPUT_SUFFIX or '(default sample: default file names)',\n"
    "    })"
)

SECTION6_CODE = _guarded(
    "6",
    "print({'ceilings': {'MAX_PIXELS': MAX_PIXELS, 'score_threshold': INPUT_SCHEMA['score_threshold'], 'max_detections': INPUT_SCHEMA['max_detections'], 'classes': len(pipe.classes)}})\n"
    "input_manifest = validate_inputs(image_paths, score_threshold=SCORE_THRESHOLD, max_detections=MAX_DETECTIONS, names=image_names)\n"
    "# Demonstrate rejection on an input that breaks the contract; the finding is recorded, not swallowed.\n"
    "try:\n"
    "    validate_inputs(sample_dir / 'does-not-exist.png')\n"
    "except ValueError as exc:\n"
    "    input_manifest['findings'].append({'input': 'missing-file-probe', 'verdict': 'rejected', 'message': str(exc)})\n"
    "write_json_atomic(output_path('outputs/__STEM___input_manifest.json'), input_manifest)\n"
    "findings = input_manifest['findings']\n"
    "print(json.dumps({'verdict': input_manifest.get('verdict'), 'findings_by_verdict': dict(Counter(f.get('verdict') for f in findings)), 'first_finding': findings[0], 'refusal_probe': findings[-1], 'written_to': str(output_path('outputs/__STEM___input_manifest.json'))}, indent=2))",
)

SECTION7_CODE = _guarded(
    "7",
    "detections = pipe.predict_many(image_paths, score_threshold=SCORE_THRESHOLD, max_detections=MAX_DETECTIONS)\n"
    "rows = [detection.to_dict() for detection in detections]\n"
    "counts = Counter(row['image_id'] for row in rows)\n"
    "print({'n_detections': len(rows), 'images_with_detections': len(counts), 'images': len(image_names), 'most_per_image': max(counts.values(), default=0), 'images_over_the_100_scored_by_coco_ap': sorted(name for name, count in counts.items() if count > 100), 'score_threshold': SCORE_THRESHOLD, 'max_detections': MAX_DETECTIONS})\n"
    "for rank, row in enumerate(sorted(rows, key=lambda row: row['score'], reverse=True)[:5], start=1):\n"
    "    print(f\"{rank:>2}. {row['image_id']:<24} class {row['class_id']:>2} {row['class_name']:<14} score {row['score']:.4f}  bbox_xyxy {[round(v, 1) for v in row['bbox_xyxy']]}\")",
)

SECTION8_CODE = _guarded(
    "8",
    "report = evaluation_report(detections, ground_truth, sample_kind=sample_kind)\n"
    "# DAT9: say whether the scored images are in the checkpoint's training split.\n"
    "report['training_overlap'] = training_overlap\n"
    "write_json_atomic(output_path('outputs/__STEM___evaluation_report.json'), report)\n"
    "print(json.dumps(report, indent=2))\n"
    "if report['verdict'] == 'not-measurable':\n"
    "    print('No ground-truth boxes were supplied, so coco_box_ap is not computed; the detections above are sanity evidence only.')",
)

SECTION9_CODE = _guarded(
    "9",
    "import csv\n"
    "import io\n\n"
    "payload = {\n"
    "    'detections': rows,\n"
    "    'evaluation_report': report,\n"
    "    'input_manifest': input_manifest,\n"
    "    'sample': {\n"
    "        'kind': sample_kind,\n"
    "        'images': image_names,\n"
    "        'sha256': sample_sha256,\n"
    "        'ground_truth_boxes': n_ground_truth_boxes,\n"
    "        'ground_truth_sha256': VAL2017_GT_SHA256 if sample_kind == DEFAULT_SAMPLE_KIND else None,\n"
    "        'source': VAL2017_RELEASE_URL if sample_kind == DEFAULT_SAMPLE_KIND else None,\n"
    "        'licences': sample_licences,\n"
    "        'annotation_licence': 'COCO annotations, CC BY 4.0' if sample_kind == DEFAULT_SAMPLE_KIND else None,\n"
    "        'training_overlap': training_overlap,\n"
    "    },\n"
    "    'notebook_source': NOTEBOOK_SOURCE,\n"
    "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
    "    'model_id': MODEL_ID,\n"
    "    'model_revision': MODEL_REVISION,\n"
    "    'model_license': MODEL_LICENSE,\n"
    "    'checkpoint_sha256': MODEL_SPEC['checkpoint_sha256'],\n"
    "    'runtime': {\n"
    "        'python': platform.python_version(),\n"
    "        'torch': torch.__version__,\n"
    "        'numpy': numpy.__version__,\n"
    "        **pipe.versions,\n"
    "        'device': pipe.device,\n"
    "    },\n"
    "}\n"
    "write_json_atomic(output_path('outputs/__STEM___result.json'), payload)\n"
    "buffer = io.StringIO(newline='')\n"
    "writer = csv.writer(buffer)\n"
    "writer.writerow(['image_id', 'rank', 'class_id', 'class_name', 'score', 'x1', 'y1', 'x2', 'y2'])\n"
    "for name in image_names:\n"
    "    for rank, row in enumerate([row for row in rows if row['image_id'] == name], start=1):\n"
    "        writer.writerow([name, rank, row['class_id'], row['class_name'], f\"{row['score']:.6f}\", *[f'{v:.2f}' for v in row['bbox_xyxy']]])\n"
    "write_text_atomic(output_path('outputs/__STEM___detections.csv'), buffer.getvalue())\n"
    "print(sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file()))",
)

TEMPLATE = {
    "package": "dimer_swin_detection",
    "repo_name": "swin-detection-pipeline",
    "stem": "swin_detection_task_inference",
    "notebook_name": "swin_detection_task_inference.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "kind": "TUTORIAL",
    # SWD-M1/m3 (NOTEBOOK_SPEC 2.2 §5): the qualified OpenMMLab stack needs CPython 3.10, which pip cannot provide. The
    # notebook therefore builds an isolated uv environment with a uv-managed CPython 3.10.18, installs the hash lock
    # compiled from tools/pins.txt there (--require-hashes --only-binary :all:, with the PyTorch CPU index and the OpenMMLab
    # find-links page the pins name), and routes every later cell to a persistent worker in it. The kernel's own Python does not matter.
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.10.18",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "data_host": f"this repository's GitHub release `{RELEASE_TAG}` (the {N_IMAGES} sample images, ~{SAMPLE_MB:.1f} MB, each SHA-256-verified)",
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime (Google Colab, Kaggle or a Linux Jupyter kernel; see Prerequisites for "
        "the recorded runs) builds an isolated environment with a uv-managed CPython 3.10.18 and the hash-locked pins of "
        "`tools/pins.txt` (the kernel's own Python and packages are left alone, so no restart is needed), stages and digest-verifies "
        f"the pinned OpenMMLab checkpoint, downloads the {N_IMAGES} pinned COCO val2017 images ({N_BOXES} ground-truth boxes, "
        f"{N_CATEGORIES} categories) from this repository's release mirror and checks each one's SHA-256, validates them into an input "
        "manifest before the model runs, detects objects, writes an evaluation report with COCO box AP against the empty-detector "
        "baseline, and exports machine-readable outputs with provenance. The default path needs no repository clone, no DIMER worker "
        "or service, no credential, no upload dialog, no configuration edit and no runtime restart (NOTEBOOK_SPEC 2.5 §5)."
    ),
    "byod": (
        "After the sample workflow completes, set `USE_BYOD = True` in the sample cell — with `BYOD_IMAGE_PATH` pointing at an image "
        "already in the runtime (works in any Jupyter kernel), or empty for the Colab upload dialog (exactly one file) — and re-run "
        "from that cell. Your image passes through the same validation, detection, evaluation-report and export cells; without "
        "ground-truth boxes the report is `not-measurable`. Its outputs get an `_exploratory_<n>` file-name suffix, so the default "
        "sample's report is never overwritten. An input the cell cannot use is refused with one message naming the file and the rule; "
        "earlier outputs are then moved to `outputs/earlier_<n>/` and Sections 6–9 skip. The upload stays inside this runtime. BYOD is "
        "optional and never part of the default path."
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
            "Open In Colab (hosted run of this revision pending)",
            "https://img.shields.io/badge/Colab-hosted%20run%20pending%20(this%20revision)-lightgrey?style=flat&logo=googlecolab",
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
        "At inference the pinned Swin-T backbone [1], [8] + Mask R-CNN head [2] maps one RGB image to class-labelled boxes and "
        "class scores over the 80 COCO 2017 categories [3]; the DIMER v1 contract exposes boxes, classes and scores only "
        "(the checkpoint's instance masks are not part of the public API). **No adaptation occurs:** no training, "
        "fine-tuning, in-context conditioning, or preprocessing fitting happens in this notebook — the upstream "
        "OpenMMLab checkpoint [5] supplies the weights and the pinned MMDetection 3.3.0 package [4] supplies the config and "
        "test pipeline, and the carried package [7] adds snapshot verification, runtime-version checks, input validation, "
        "a fixed output contract and the `coco_box_ap`, `validate_inputs` and `evaluation_report` helpers.\n\n"
        "**Trust boundary (MOD12).** The checkpoint is a code-capable PyTorch `.pth` serialization. The carried "
        "`verify_snapshot` re-hashes it against the inline manifest (and against the digest the package has always "
        "pinned in `MODEL_SPEC`) before the pinned MMDetection loader deserializes it inside `mmengine`; the "
        "deserialization call is upstream's, not the package's, and is **not** a `weights_only` load. A matching "
        "digest proves byte identity with the pinned OpenMMLab distribution, not publisher authenticity — run this "
        "notebook only where that pinned source is trusted.\n\n"
        f"**The scored sample and what it can show (DAT9).** The default sample is {N_IMAGES} images from the COCO 2017 "
        f"**validation** split (val2017) [3], [6] with their {N_BOXES} human-drawn ground-truth boxes, pinned per image by SHA-256. "
        "The checkpoint was trained on COCO **train2017** only (its backbone was pretrained on ImageNet-1K), and none of these "
        "images is in train2017: they were chosen by a fixed rule, before any score was seen — a permissive image licence "
        "(CC BY 2.0, CC BY-SA 2.0, no known copyright restrictions, or US Government work), at least one ordinary box and no "
        "crowd box, and no train2017 match by image id, Flickr photo id or (width, height, capture time) — then the first "
        f"{N_IMAGES} by image id. val2017 is, however, the split upstream used to report box AP 42.7 and possibly to select the "
        "model, so this is not an independent test: it is a small sanity measurement on images the model never trained on, "
        "with COCO box AP against an empty-detector baseline — tutorial evidence, not a benchmark. A synthetic scene drawn in code "
        "(no ground truth) remains available as `SAMPLE = 'synthetic'` to show what the report says when nothing can be measured."
    ),
    "learning_objectives": (
        "install the pinned Python 3.10 OpenMMLab CPU runtime into an isolated environment, read what the carried package "
        "guarantees, resolve and digest-verify the immutable OpenMMLab checkpoint, fetch the pinned labelled COCO val2017 sample "
        "(or switch to the synthetic scene or a BYOD image) and validate it into an input manifest, state whether the scored "
        "images overlap the checkpoint's training data, run detection through the public API, read uncalibrated class scores "
        "and the caller-owned `score_threshold` correctly, produce an evaluation report that is `sample-sanity` with "
        "`coco_box_ap` only when ground-truth boxes exist and `not-measurable` otherwise, and export machine-readable "
        "detections plus provenance."
    ),
    "exclusions": (
        "instance or semantic segmentation (the checkpoint's mask head is outside the DIMER v1 contract), "
        "open-vocabulary detection, tracking, keypoints, image classification, or any training or fine-tuning. The "
        "label space is fixed to the 80 COCO 2017 categories; objects outside that space are either missed or "
        "assigned a COCO label."
    ),
    "guided_opening": [
        (
            "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n"
            "> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n"
            "> model weights and datasets under their own licenses (see the model card) and uses compute\n"
            "> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n"
            "> responsible for having the right to use it. Results here are tutorial evidence, not\n"
            "> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n"
            "> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n"
            "> its use."
        ),
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
            "defaults for the first run: the notes and sample answers describe the default COCO val2017 path.\n\n"
            "**Section tags.** Each learner heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation "
            "practice]** — how the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and "
            "packaging.\n\n"
            "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an "
            "expectation; after it, **What to notice** describes normal output and a collapsed **Check your reasoning** block gives a "
            "worked answer. Numbers quoted there name the revision and run they come from; read your own output first."
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
            f"| 5. The sample | [Evaluation practice] | {N_IMAGES} COCO val2017 images fetched and verified (or synthetic / BYOD) | images, licences, {N_BOXES} boxes, training overlap |\n"
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
            "| **train2017 / val2017** | COCO 2017's training split (118k images; this checkpoint was trained on it) and validation split (5k images; upstream reports its benchmark number on it). |\n"
            "| **Train/evaluation overlap** | Scoring a model on images it was trained on (resubstitution) gives an optimistic number that says little about new images; the report's `training_overlap` field records whether that happened. |\n"
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
            "| **`sample-sanity`** | The verdict for a metric on a small tutorial sample: evidence the path works, not a benchmark. |\n"
            "| **Isolated environment** | A separate Python 3.10 built from the hash-locked pins, in which every learner cell runs. |\n"
            "| **BYOD** | Bring Your Own Data: the optional switch that runs the same cells on your image. |\n\n"
            "</details>"
        ),
    ],
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime; the kernel's own Python version does not matter. The qualified OpenMMLab stack — torch 2.1.2 (CPU build), MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.3.0, NumPy 1.26.4 — has prebuilt wheels for Python 3.10 only, so Section 1 has `uv` provision a managed **CPython 3.10.18**, installs the hash-locked pins there, and runs every later cell in that interpreter (Section 4 asserts it). Recorded runs: GitHub Actions (Python 3.10 Jupyter kernel) on every pull request, and one Colab CLI run on a fresh T4 VM (CPU execution) of the previous revision of this notebook (notebook blob `ea6b456d`, 2026-10-10, recorded in `docs/release-verification.md`); the hosted run of this revision is still to be recorded. CPU is the default and only qualified path; no GPU is required. The pinned torch/mmcv wheels are the largest downloads of the run.",
        f"- **External package indexes and data host:** the isolated install reads the PyTorch CPU index (`download.pytorch.org`) for the `+cpu` torch builds and the OpenMMLab wheel page (`download.openmmlab.com`) for MMCV, besides PyPI. The {N_IMAGES} sample images come from this repository's GitHub release `{RELEASE_TAG}` (`github.com`, which redirects to GitHub's release-asset storage).",
        "- **Knowledge:** basic Python and image handling; what a detection score and an IoU-based AP metric are.",
        f"- **Data:** the default sample is {N_IMAGES} COCO 2017 validation images ({N_BOXES} ground-truth boxes over {N_CATEGORIES} categories, ~{SAMPLE_MB:.1f} MB), each verified against its pinned SHA-256 before use, with the boxes carried inline in the sample cell and checked against their digest; it is evaluated with COCO box AP. **Licences and attribution:** COCO's annotations are CC BY 4.0 (COCO Consortium); every image keeps its own Flickr licence — {LICENCE_SUMMARY} — and the sample cell exports each image's licence, Flickr source URL and original COCO URL in the result JSON. The photographers' names are not recorded in COCO's metadata, so attribution is by Flickr source URL; the release's `ATTRIBUTION.md` lists every image. `SAMPLE = 'synthetic'` switches to a deterministic 640×480 scene generated in code (no download, no ground truth). `USE_BYOD` (off by default) takes one image file decodable by Pillow (PNG/JPEG/WebP and similar, at most 64 megapixels) from `BYOD_IMAGE_PATH`, or from the Colab upload dialog when that field is empty. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
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
                "## 5. The sample: COCO val2017 (default), the synthetic scene, or your image · [Evaluation practice]\n\n"
                f"The default sample is **{N_IMAGES} COCO 2017 validation images** [3], [6]. The cell downloads each image from this "
                f"repository's release `{RELEASE_TAG}` (a mirror of the files at `images.cocodataset.org`, which is recorded for "
                "provenance only) and refuses any file whose size or SHA-256 differs from the pin. The ground-truth boxes are carried "
                "in the cell itself — one row per image, already converted from COCO's `[x, y, w, h]` to pixel `[x1, y1, x2, y2]` with "
                "COCO's category order as the class index — and the cell checks them against their recorded digest, so every run "
                f"scores the same {N_BOXES} boxes. Nothing is resplit or relabelled.\n\n"
                "**Why these images (DAT9).** The checkpoint was trained on COCO train2017 only. Scoring it on train2017 images would "
                "measure how well it remembers its training data, not how it does on new photographs. These images are from val2017 "
                "and were screened against train2017; each row also names the image's licence and Flickr source, which the export "
                "keeps. val2017 is the split upstream reported its 42.7 box AP on, so the result is a sanity measurement, not an "
                "independent test. The `training_overlap` field travels into the evaluation report.\n\n"
                "`SAMPLE = 'synthetic'` instead draws a deterministic 640×480 scene in code (a red→green gradient with three "
                "flat-coloured shapes). It depicts no COCO object and has **no ground truth**, so its detections are only a check "
                "that the input contract, preprocessing and forward pass work (Section 10 uses it). `USE_BYOD = True` takes one "
                "image from `BYOD_IMAGE_PATH`, or — in Colab, with the field empty — from the upload dialog, which must receive "
                "exactly one file; BYOD has no ground truth unless you build it yourself. An input the cell cannot use (a missing "
                "path, no upload dialog, zero or several uploaded files, an undecodable or oversized image) is refused with one "
                "message naming the file and the rule; earlier outputs are moved to `outputs/earlier_<n>/`, and Sections 6–9 skip "
                "until the cell accepts an input. Every pass other than the default sample writes its outputs with an "
                "`_exploratory_<n>` suffix, so the default sample's files are never overwritten. Look for a dictionary naming the "
                "sample kind, the number of images, the first digests, the ground-truth boxes, the image licences and whether the "
                "images are in the training data.\n\n"
                "**Predict before running:** how many images and ground-truth boxes does the default sample hold, and are any of the "
                "images in the checkpoint's training data?"
            ),
            "code": _c(SAMPLE_CELL),
        },
        {
            "md": (
                f"**What to notice (Section 5):** `sample_kind` is `COCO-val2017-50`, {N_IMAGES} images, `ground_truth_boxes: {N_BOXES}` over "
                f"{N_CATEGORIES} categories, the licence counts ({LICENCE_SUMMARY}), `eval_data_in_training: False` and the default "
                "output names.\n\n"
                f"<details><summary>Check your reasoning</summary>{N_IMAGES} images and {N_BOXES} boxes, none of them in train2017: the "
                "images were picked by a fixed rule before any score was seen, and the per-image digests and the ground-truth digest are "
                "what make these the same images and boxes in every run. Being outside the training data is what lets Section 8 "
                "measure anything at all; being in val2017, the split upstream reports on, is why it is still a sanity measurement "
                "rather than an independent test.</details>\n\n"
                "## 6. Validate the input → input manifest · [Evaluation practice]\n\n"
                "`validate_inputs` is the package's public validation stage: it applies exactly the checks `predict` "
                "applies — the request ceilings (`score_threshold` in 0..1, `max_detections` ≥ 1) and, per image, "
                "`validate_image` (the file exists, Pillow can decode it, positive dimensions, at most `MAX_PIXELS` = "
                "64,000,000 pixels) — and returns an **input manifest** naming the schema and ceilings, each input's "
                "observed path, size and mode, the request parameters and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json` (written to a temporary file and renamed into place, so an interrupted "
                "write never leaves a truncated file). To show what rejection looks like, the cell also validates a "
                "path that does not exist and records the package's own error message as a finding. The cell prints a summary; "
                "the file holds every finding. Inside the package every accepted image goes through the pinned config's "
                "MMDetection test pipeline (resize, normalise, pad); nothing is dropped or altered by the package itself.\n\n"
                "**Predict before running:** the probe asks for a file that does not exist. Will validation reject it before the "
                "model is called, and will the message name the file?"
            ),
            "code": _c(SECTION6_CODE),
        },
        {
            "md": (
                f"**What to notice (Section 6):** the verdict, {N_IMAGES} `accepted` findings and one `rejected` finding for the "
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
                "(`model.eval()`, no sampling); CPU kernel choices can reorder near-tied scores. On CPU the "
                f"{N_IMAGES} images take a few minutes. Look for the detection count, how many images have detections, the "
                "largest per-image count and the five highest-scoring boxes.\n\n"
                f"**Predict before running:** with `score_threshold = 0.0`, will the detector return about {N_BOXES} boxes (one per "
                "labelled object), or many more? Will the top-scoring boxes be people, animals or vehicles?"
            ),
            "code": _c(SECTION7_CODE),
        },
        {
            "md": (
                "**What to notice (Section 7):** `n_detections`, the per-image maximum, any image listed as over the 100 that COCO AP "
                "scores, and the five top boxes with class names and scores.\n\n"
                f"<details><summary>Check your reasoning</summary>Many more than {N_BOXES}: at threshold 0.0 the detector keeps every box "
                "that survives NMS, up to `max_detections` (300) per image, including many low-scoring guesses. That is deliberate: a "
                "display threshold is the caller's later choice. COCO AP, however, scores only the **100 highest-scoring detections per "
                "image** (pycocotools `maxDets=100`), so on an image listed as over 100 the lowest-scoring extras are exported but not "
                "scored. The top boxes should be confident, plausible COCO classes for these photographs; their scores rank boxes but "
                "are not probabilities.</details>\n\n"
                "## 8. Evaluate → evaluation report · [Evaluation practice]\n\n"
                "`evaluation_report` is the package's public evaluation stage and always produces a report. When "
                "ground-truth boxes exist (the default val2017 sample) it carries `coco_box_ap` — the repository's metric "
                "helper, COCO AP@[0.50:0.95], AP50 and AP75 via `pycocotools` over axis-aligned boxes, top 100 detections per "
                f"image — with the verdict `sample-sanity` and the empty-detector baseline (AP 0 by construction): a {N_IMAGES}-image "
                "tutorial metric with sampling variance and no dispersion estimate. The upstream full-val2017 box AP of 42.7 that "
                "`MODEL_SPEC` records is upstream-reported context, not measured here. The cell adds the `training_overlap` field "
                "from Section 5, so the report itself says whether the scored images were in the training data. On the synthetic "
                "scene (and on BYOD without labels) no metric exists, so the verdict is `not-measurable` and the report states "
                "what would make the task measurable: ground-truth boxes for the evaluated images, or a labelled "
                "holdout from the deployment domain. The report is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                f"**Predict before running:** will AP@[.50:.95] on these {N_IMAGES} images be near 0, near the upstream 42.7 % "
                "(0.427), or much higher? Which will be higher, AP50 or AP75, and why?"
            ),
            "code": _c(SECTION8_CODE),
        },
        {
            "md": (
                f"**What to notice (Section 8):** the verdict `sample-sanity`, the three AP values, the {N_BOXES} ground-truth boxes over "
                f"{N_IMAGES} images, the empty-detector baseline at 0, and `training_overlap` with `eval_data_in_training: false`.\n\n"
                "<details><summary>Check your reasoning</summary>Expect a value in the neighbourhood of the upstream 0.427 rather than "
                f"far above it, with a wide margin either way: {N_IMAGES} images are a small sample, the report gives no dispersion "
                "estimate, and these images are a rule-selected slice of val2017, not all 5,000. The hosted run of this revision, once "
                "recorded, is listed in `docs/release-verification.md`; compare your number with it, not with a guess. For contrast: "
                "the previous revision of this notebook (notebook blob `ea6b456d`, Colab T4 run of 2026-10-10) scored AP@[.50:.95] "
                "0.707 on four COCO8 images. Those four images are COCO **train2017** images the checkpoint was trained on, so 0.707 "
                "was a resubstitution score — the model being tested on its own training data — and not evidence that it beats its "
                "published number. AP50 is higher than AP75 because a looser overlap requirement accepts more boxes.</details>\n\n"
                "## 9. Export outputs and provenance · [Engineering]\n\n"
                "Machine-readable JSON preserves every detection (score-ordered per image), the evaluation report, the "
                "input manifest, the sample identity and digests (with each image's licence and Flickr source, the annotation "
                "licence and the training-overlap statement), the notebook's source (repository, revision, embedded "
                "module digest, generator), the model identifier, the immutable revision label, the checkpoint digest "
                "the package verified, and the runtime identity (Python, `torch`, `mmdet`, `mmcv`, `mmengine`, device). "
                "The detections are also written as CSV with explicit `rank`, `class_id`, `class_name`, `score` and "
                "`x1,y1,x2,y2` columns so score ordering and pixel coordinates survive downstream use. Every file is written "
                "atomically (temporary file, then rename). No credentials are recorded."
            ),
            "code": _c(SECTION9_CODE),
        },
    ],
    "closing": (
        "## 10. Activity: change one thing — a scene with no ground truth · [Concept]\n\n"
        "Optional; **Predict → Change one thing → Run → Observe → Explain**. It changes nothing unless you do it.\n\n"
        "1. **Predict:** on a scene of three flat-coloured shapes on a gradient, what will the detector return, and what will the "
        "evaluation report say?\n"
        "2. **Change:** in Section 5 set `SAMPLE = 'synthetic'`. Change nothing else.\n"
        "3. **Run:** select the Section 5 cell and choose *Runtime → Run after* (or run Sections 5–9 in order). This pass writes its "
        "files with an `_exploratory_<n>` suffix; the default sample's outputs stay as they were.\n"
        "4. **Observe:** the detection count and top scores in Section 7, and the verdict in Section 8.\n"
        "5. **Explain:** why does the report refuse to give a number here, even though the detector returned boxes?\n\n"
        "<details><summary>Check your reasoning</summary>Few boxes, with low scores, possibly with COCO labels that make no sense "
        "for a rectangle or a circle — the detector always answers in its 80-class vocabulary. The report says `not-measurable`, "
        "because there are no ground-truth boxes to compare with: a score, however high, is not evidence of correctness. A recorded "
        "run of this path (GitHub Actions, 2026-09-14, notebook blob `966c4ad9`, commit `181c09c`) reported `not-measurable` as "
        "expected; this revision's synthetic path has not been run on a hosted runtime. Set `SAMPLE = 'val2017'` again to return to "
        "the measured path.</details>\n\n"
        "## Interpretation and limits\n\n"
        "Each detection is a class from the fixed 80-category COCO 2017 label space with an axis-aligned box and an "
        "**uncalibrated** MMDetection score; the package ships no acceptance threshold and `score_threshold` is an output "
        f"filter the caller owns. The default `coco_box_ap` value from the {N_IMAGES}-image COCO val2017 sample is tutorial "
        "evidence for those images. The images are outside the checkpoint's training split (train2017), but val2017 is the "
        "split upstream reported on and may have used for model selection, and the overlap screen compared ids and capture "
        "metadata, not pixels; the value must not be generalized to a domain, camera, object size distribution or class mix. "
        "Objects outside the COCO categories, crowded or tiny objects (crowd-annotated images were excluded from the sample), "
        "unusual viewpoints, and domain shifts (medical, aerial, line art) all degrade results in ways the package does not "
        "detect. On the synthetic scene the detections are meaningless by construction and the report says `not-measurable`. "
        "The package provides no segmentation, tracking, keypoint, classification, or training capability, and the `.pth` "
        "checkpoint remains a code-capable serialization whose digest check fixes the bytes, not the author.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned OpenMMLab checkpoint, provision and assert the qualified Python 3.10 / "
        "MMDetection 3.3.0 runtime in an isolated environment, validate the demonstrated input, execute the public detection path, and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority, reproduction of the upstream COCO result, score calibration, safety for high-consequence "
        "decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** run the Section 10 activity; enable `USE_BYOD` with a photograph from your "
        "own domain and inspect the score distribution before choosing a display threshold; label a small holdout from "
        "that domain in the `boxes_from_yolo_labels` format and compare its AP with the val2017 value.\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a Windows or macOS kernel, or an ARM machine | Use a Linux x86_64 Jupyter kernel or Google Colab; the pinned wheels are manylinux x86_64 builds. |\n"
        "| Section 1 fails while downloading `uv`, or `The pinned uv wheel failed its size/SHA-256 check` | a network failure, or an altered download | Run the Section 1 install cell again; never edit the digest. |\n"
        "| `CalledProcessError` from `uv pip install` in Section 1 (`No solution found`, HTTP errors) | the PyTorch CPU index, the OpenMMLab wheel page or PyPI was unreachable | Run the cell again later; all three hosts must be reachable. Never loosen a pin. |\n"
        "| `holds Python …, not 3.10.18` in Section 1 | an older isolated-environment folder (`dimer_isolated_env/`, or the folder `DIMER_ISOLATED_ENV` names) | Delete that folder (or start a fresh runtime) and run the install cell again. |\n"
        "| `Python 3.10 is required …` in Section 4 | the cell ran outside the isolated environment (Section 1 was skipped) | Run Sections 1–3 first, or choose *Run all*. |\n"
        "| The checkpoint download fails in Section 3, or `sha256 … != manifest` | `download.openmmlab.com` unreachable, or a changed file | Run the Section 3 cell again; a digest mismatch is never loaded — do not edit the manifest. |\n"
        "| `RuntimeError` about `mmdet`, `mmcv` or `mmengine` versions | runtime-version drift (a different wheel was installed) | Delete `dimer_isolated_env/` and run Section 1 again; the pins must match `MODEL_SPEC`. |\n"
        "| `Could not download … ` or `… refusing to use it` in Section 5 | `github.com` unreachable, or a partial or changed image download | Run the cell again: images that already match their pin are kept, and a mismatch is never used. Never edit the pin. |\n"
        "| `BYOD input refused: …` in Section 5, then `Section 6 skipped` … | a missing path, no upload dialog outside Colab, zero or several uploaded files, or an undecodable or oversized image | Follow the message (set `BYOD_IMAGE_PATH`, choose one file, or resize the image) and run Section 5 again; earlier outputs are in `outputs/earlier_<n>/`. |\n"
        "| `The isolated environment's Python process exited` | the worker ran out of memory | Use a smaller image, then choose *Run all* again. |\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        f"1. On {N_IMAGES} COCO val2017 images ({N_BOXES} boxes, none in the training split) the detector reached AP@[.50:.95] ___, AP50 ___ and AP75 ___, against 0 for the empty detector.\n"
        "2. What that number can tell me, and what it cannot (think of val2017's role upstream): ___.\n"
        "3. On the synthetic scene the report said ___, because ___.\n"
        "4. Before using these scores on my own images I would need ___ (for a threshold) and ___ (for a measurement).\n\n"
        "## References\n\n"
        "[1] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, \"Swin Transformer: Hierarchical vision transformer using shifted windows,\" in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 10012–10022, doi: 10.1109/ICCV48922.2021.00986. [Online]. Available: https://arxiv.org/abs/2103.14030\n\n"
        "[2] K. He, G. Gkioxari, P. Dollár, and R. Girshick, \"Mask R-CNN,\" in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2961–2969, doi: 10.1109/ICCV.2017.322. [Online]. Available: https://arxiv.org/abs/1703.06870\n\n"
        "[3] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, \"Microsoft COCO: Common objects in context,\" in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2014, pp. 740–755, doi: 10.1007/978-3-319-10602-1_48. [Online]. Available: https://arxiv.org/abs/1405.0312\n\n"
        "[4] K. Chen *et al.*, \"MMDetection: Open MMLab detection toolbox and benchmark,\" arXiv:1906.07155, 2019. Config used: MMDetection v3.3.0 `configs/swin`. [Online]. Available: https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/swin\n\n"
        "[5] OpenMMLab, \"mask_rcnn_swin-t-p4-w7_fpn_1x_coco checkpoint (20210902_120937-9d6b7cfa),\" 2021. [Online]. Available: https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth\n\n"
        f"[6] COCO Consortium, \"COCO 2017 validation images and annotations\" (annotations CC BY 4.0; images under their Flickr licences), mirrored as release `{RELEASE_TAG}` with per-image attribution. [Online]. Available: https://github.com/kurtvalcorza/swin-detection-pipeline/releases/tag/{RELEASE_TAG}\n\n"
        "[7] K. Valcorza, \"swin-detection-pipeline: README, model card and weight provenance,\" GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/swin-detection-pipeline/blob/main/README.md, https://github.com/kurtvalcorza/swin-detection-pipeline/blob/main/MODEL_CARD.md, https://github.com/kurtvalcorza/swin-detection-pipeline/blob/main/docs/WEIGHTS.md\n\n"
        "[8] Microsoft, \"Swin Transformer,\" GitHub repository. [Online]. Available: https://github.com/microsoft/Swin-Transformer"
    ),
}
