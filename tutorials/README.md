# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/swin-detection-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/swin-detection-pipeline/blob/main/tutorials/swin_detection_task_inference.ipynb)
[![Python 3.10 required](https://img.shields.io/badge/Python-3.10%20required-3776ab?style=flat&logo=python&logoColor=white)](../README.md)
[![Checkpoint](https://img.shields.io/badge/Checkpoint-mask__rcnn__swin__tiny__patch4__window7__1x-ffcc4d?style=flat)](https://github.com/SwinTransformer/storage/releases/tag/v1.0.3)
[![Upstream](https://img.shields.io/badge/Upstream-microsoft%2FSwin--Transformer-181717?style=flat&logo=github&logoColor=white)](https://github.com/microsoft/Swin-Transformer)
[![arXiv](https://img.shields.io/badge/arXiv-2103.14030-b31b1b.svg)](https://arxiv.org/abs/2103.14030)
[![Model released](https://img.shields.io/badge/Model%20released-2021--05--11-6f42c1?style=flat)](https://github.com/SwinTransformer/storage/releases/tag/v1.0.3)
[![Sample eval](https://img.shields.io/badge/Sample%20eval-COCO%20AP%200.707%20%7C%20AP50%200.957-2ea44f?style=flat)](../MODEL_CARD.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `swin_detection_task_inference.ipynb` | `TASK-INFERENCE` | Verified pretrained Swin-T + Mask R-CNN object detection; COCO-style tutorial evaluation; machine-readable outputs/provenance | CPU / CPython 3.10 Jupyter | Optional image, gated off by default | **release-grade** — exact committed notebook passed clean GitHub-hosted execution on 2026-09-11 |

## Supported user-facing capability

The repository now exposes a narrow public task-inference API, `dimer_swin_detection.DimerSwinDetector`, plus the `dimer-swin-detect` CLI. It wraps the pinned OpenMMLab MMDetection 3.3.0 Swin-T + Mask R-CNN inference path, verifies the exact checkpoint size and SHA-256 before `.pth` deserialization, validates image inputs, returns only the v1 DIMER object-detection surface (boxes/classes/scores), and exports provenance. Instance-mask output remains outside the v1 contract.

This does **not** remove the existing gradient-adaptation blockers. The canonical DIMER COCO representation, validator/finetuner workers, artifact-serving composition, and accelerator qualification for training are still future work. The release-grade tutorial therefore uses `TASK-INFERENCE`, not `E2E`, and makes no fine-tuning claim.

## Model identity and source distinction

The task-inference runtime uses the official OpenMMLab `mask-rcnn_swin-t-p4-w7_fpn_1x_coco` distribution associated with MMDetection 3.3.0: checkpoint size 191,461,353 bytes and SHA-256 `9d6b7cfaa4aad52ef559611bea454f01d6f1f17c82a1abfac0d71631a193a291`. This is deliberately recorded separately from the Microsoft/SwinTransformer release asset already frozen in `provenance/open-weights.json` for the future adaptation lineage; the tutorial does not claim those two files are byte-identical.

The upstream `.pth` format is code-capable serialization. Digest verification proves that the bytes match the pinned distribution; it does not prove sender authenticity or make an otherwise untrusted checkpoint safe.

## Tutorial evidence and limits

The default labelled sample is COCO8, a very small public COCO 2017-derived tutorial subset. The notebook preserves its validation membership and reports COCO AP@[0.50:0.95], AP50, and AP75 as **sample/tutorial metrics only**. It also records the trivial empty-detector AP=0 baseline. The upstream full-COCO AP is displayed only as upstream-reported context and is not represented as notebook-measured performance.

The notebook writes `detections.json`, `detections.csv`, `metrics.json`, and `provenance.json`. BYOD inference is optional and disabled by default. Scores are explicitly described as uncalibrated; the caller owns deployment threshold calibration.

## Release verification

Clean execution evidence for the committed `TASK-INFERENCE` notebook:

- **Date:** 2026-09-11 UTC
- **PR head tested:** `354545b89fef6934d01eb932d868fc3e1260d830`
- **GitHub Actions run:** `verify-task-tutorial` run `34574579363`, conclusion **success**
- **Environment:** GitHub-hosted Ubuntu 24.04.5, CPython 3.10.19; CPU runtime
- **Effective model stack:** torch 2.1.2+cpu; MMDetection 3.3.0; MMCV 2.1.0; MMEngine 0.10.7; NumPy 1.26.4
- **Checkpoint:** 191,461,353 bytes; SHA-256 `9d6b7cfaa4aad52ef559611bea454f01d6f1f17c82a1abfac0d71631a193a291`; verified before loading
- **Default sample:** four frozen COCO8 validation images, 17 ground-truth boxes
- **Tutorial metrics observed:** COCO AP@[0.50:0.95] `0.7073101933`; AP50 `0.9570957096`; AP75 `0.6993399340`; empty-detector baseline AP `0.0`
- **Exports asserted:** `detections.json`, `detections.csv`, `metrics.json`, `provenance.json`

The registry promotion in this commit changes documentation only; the notebook bytes exercised by the recorded run are unchanged. `.github/workflows/verify-task-tutorial.yml` remains the regression gate and re-executes the committed notebook whenever the notebook, runtime, registry, or workflow changes.

Static JSON/compile checks are not treated as execution evidence. The older `verify-smoke-notebook.yml` remains useful engineering coverage but is not evidence for the task tutorial.

## Recorded SHOULD deviation

**G16 / model-card link:** the current `MODEL_CARD.md` is a scaffold-lifecycle card written for the future gradient-adaptation pipeline and still describes that capability as having no runtime. The task-inference tutorial deliberately does not present that card as documentation for the newly implemented pretrained inference surface. This is a documentation follow-up, not a `TASK-INFERENCE` execution blocker; until the model card is revised, this registry and the runtime `MODEL_SPEC` are the durable source for the supported inference capability and its exact checkpoint identity.
