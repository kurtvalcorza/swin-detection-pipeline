# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `swin_detection_task_inference.ipynb` | `TASK-INFERENCE` | Verified pretrained Swin-T + Mask R-CNN object detection; COCO-style tutorial evaluation; machine-readable outputs/provenance | CPU / CPython 3.10 Jupyter | Optional image, gated off by default | **Candidate** — real task runtime implemented; promote only after exact-revision clean notebook execution passes |
| `swin_detection_scaffold_smoke_colab.ipynb` | `SMOKE` | Scaffold lifecycle/model-card/provenance checks | CPU / Python 3.11+ | — | **Engineering-only** |

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

`.github/workflows/verify-task-tutorial.yml` is the promotion gate for `swin_detection_task_inference.ipynb`. It must execute the **committed notebook** top-to-bottom from a fresh Python 3.10 Jupyter environment, exercise the repository API, download and verify the model, evaluate the default sample, and assert the four machine-readable outputs. Static JSON/compile checks are kept separate from execution evidence.

The older `verify-smoke-notebook.yml` remains useful engineering coverage but is not evidence for the task tutorial.
