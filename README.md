# swin-detection-pipeline

Swin **object-detection** pipeline for DIMER — pretrained task inference is implemented; the composed-worker gradient-adaptation pipeline remains a scaffold.

## Upstream alignment

This repository corresponds to the **Object Detection** downstream task of the original Microsoft Swin Transformer project.

- **Upstream project:** [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- **Official Swin detection lineage:** [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
- **Upstream task:** Object Detection
- **Canonical benchmark:** COCO 2017
- **Reference architecture:** Swin-T backbone + Mask R-CNN
- **DIMER task boundary:** image → class-labelled axis-aligned bounding boxes and class scores
- **Primary metrics:** COCO AP@[0.50:0.95], AP50, AP75

The upstream architecture also produces instance masks, but this repository's DIMER v1 detection surface is **object detection only**. Instance-mask output is not exposed by the public task-inference API.

## Public pretrained task-inference runtime

The repository exposes a real, public, CPU-capable pretrained inference path:

- **Python API:** `dimer_swin_detection.DimerSwinDetector`
- **CLI:** `dimer-swin-detect`
- **Package:** `dimer-swin-detection` 0.1.0
- **Qualified runtime:** CPython 3.10; torch 2.1.2; MMDetection 3.3.0; MMCV 2.1.0; MMEngine 0.10.7; NumPy 1.26.4; OpenCV 4.10.0.84
- **Model distribution:** OpenMMLab `mask-rcnn_swin-t-p4-w7_fpn_1x_coco`
- **Checkpoint:** 191,461,353 bytes; SHA-256 `9d6b7cfaa4aad52ef559611bea454f01d6f1f17c82a1abfac0d71631a193a291`
- **Input validation:** readable image, positive dimensions, maximum 64,000,000 pixels
- **Outputs:** image id, 80-class COCO class id/name, uncalibrated score, `bbox_xyxy`
- **Checkpoint handling:** exact size and SHA-256 are verified before the upstream `.pth` file is deserialized

The `.pth` format is code-capable PyTorch serialization. Digest verification proves byte identity against the pinned distribution; it does not prove publisher authenticity or make an otherwise untrusted checkpoint safe.

Machine-readable capability metadata is in [`spec/task-inference-surface.json`](spec/task-inference-surface.json).

### Supported installation contract

The OpenMMLab stack includes binary wheels selected from the OpenMMLab/PyTorch indexes, so **`pip install .` by itself is not a qualified runtime installation**. `pyproject.toml` intentionally packages the DIMER wrapper and CLI without pretending that standard PyPI dependency resolution can reproduce the tested binary stack.

The supported public installation/bootstrap path is the release-grade notebook [`tutorials/swin_detection_task_inference.ipynb`](tutorials/swin_detection_task_inference.ipynb), which installs the exact qualified Python 3.10 CPU dependency graph before installing this repository package with `--no-deps`. A standalone caller may reproduce those same pinned commands, but changing Python, torch, MMCV, MMEngine or MMDetection versions is outside the qualified runtime until revalidated. The API itself also checks the OpenMMLab package versions at startup and fails closed on drift.

### Release-grade tutorial

[`tutorials/swin_detection_task_inference.ipynb`](tutorials/swin_detection_task_inference.ipynb) is the DIMER Notebook Specification 1.0 **`TASK-INFERENCE` release-grade tutorial**. It exercises the repository API rather than reimplementing model inference, evaluates a fixed labelled COCO8 validation subset with COCO AP/AP50/AP75, includes a gated BYOD image path, and exports detections, metrics, and provenance.

Clean GitHub-hosted execution of the exact committed notebook passed on 2026-09-11. The recorded four-image tutorial sample measured COCO AP@[0.50:0.95] `0.7073101933`, AP50 `0.9570957096`, and AP75 `0.6993399340`. These are **small-sample tutorial metrics**, not reproduction of the upstream full-COCO benchmark and not production-fitness evidence. See [`tutorials/README.md`](tutorials/README.md) for the release record and limitations.

## Gradient-adaptation builder status

The separate DIMER **gradient-adaptation** design remains a scaffold. The reviewed `ml-worker` contract has the object-detection task profile, but no canonical vision detection representation profile is present, and task-specific validator/finetuner releases do not yet exist. No gradient-adaptation composition or release manifest is emitted.

**Lifecycle for that composed-worker surface (DIMER Pipeline Specification 1.0):** `scaffold` — declared machine-readably in `spec/pipeline-surface.json` with intended `implementation_topology: COMPOSED-WORKERS` and `capability_modes: [GRADIENT-ADAPTATION]`. The canonical Microsoft/SwinTransformer checkpoint recorded for that future adaptation lineage carries `redistribution_status: unknown` in `provenance/open-weights.json`, so DIMER hosting remains **BLOCKED** until an authoritative weight-licence determination is recorded.

`scripts/verify_scaffold.py` continues to refuse any lifecycle above `scaffold` while adaptation blockers exist, any declared adaptation components/release, and any non-blocked DIMER hosting for an `unknown`/`prohibited` weight status. The pretrained `TASK-INFERENCE` capability does **not** satisfy or remove those adaptation blockers.

`MODEL_CARD.md` distinguishes the implemented pretrained inference surface from the still-scaffolded gradient-adaptation design. The inference runtime uses the separately pinned OpenMMLab distribution recorded in `spec/task-inference-surface.json`; the adaptation lineage remains governed by `spec/pipeline-surface.json` and `provenance/open-weights.json`.

Repository-owned artifacts include:

- `spec/task-inference-surface.json` — implemented pretrained inference capability, runtime/model/output/evaluation contract;
- `src/dimer_swin_detection/` — public inference API and CLI;
- `tutorials/swin_detection_task_inference.ipynb` — release-grade `TASK-INFERENCE` tutorial;
- `.github/workflows/verify-task-tutorial.yml` — exact-notebook clean execution gate;
- `spec/pipeline-surface.json` — future gradient-adaptation task/model/contract surface and blockers;
- `provenance/open-weights.json` — Microsoft/SwinTransformer adaptation-lineage provenance;
- `scripts/verify_scaffold.py` — verifies that the adaptation scaffold remains internally consistent and fail-closed;
- `.github/workflows/verify-scaffold.yml` — scaffold verification CI.

## Cloud verification & adaptation-lineage open-weights digest

The pre-existing scaffold and Microsoft/SwinTransformer release asset were separately verified in an isolated Kaggle cloud container:

- **Kernel:** [`kurtvalcorza/swin-detection-verify`](https://www.kaggle.com/code/kurtvalcorza/swin-detection-verify)
- **Status:** `KernelWorkerStatus.COMPLETE` (Exit Code 0)
- **Scaffold Verifier:** `PASS`
- **Microsoft/SwinTransformer checkpoint asset:** `mask_rcnn_swin_tiny_patch4_window7_1x.pth`, 191,487,694 bytes, `SwinTransformer/storage@v1.0.3`, asset ID `36780486`
- **SHA-256:** `b67f9d6cd62a4d723c78faec1b49cbf548faa22437264defb00f2f6e54d21b78`

That asset is part of the frozen future gradient-adaptation lineage and is **not claimed to be byte-identical** to the OpenMMLab task-inference checkpoint used by `DimerSwinDetector`.

## Canonical Swin family

1. **Image Classification** → `swin-classification-pipeline`
2. **Semantic Segmentation** → `swin-segmentation-pipeline`
3. **Object Detection** → `swin-detection-pipeline`

See issue #1 for the original NATIVE task, dataset, model, provenance, and conformance specification.
