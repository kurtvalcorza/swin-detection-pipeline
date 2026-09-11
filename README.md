# swin-detection-pipeline

Swin **object-detection** pipeline for DIMER — NATIVE ml-worker vision pipeline umbrella and specification.

## Upstream alignment

This pipeline corresponds to the **Object Detection** downstream task of the original Microsoft Swin Transformer project.

- **Upstream project:** [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- **Official detection implementation:** [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
- **Upstream task:** Object Detection
- **Canonical benchmark:** COCO 2017
- **Reference architecture:** Swin-T backbone + Mask R-CNN
- **Pipeline task boundary:** image → class-labelled bounding boxes
- **Primary metrics:** COCO AP@[0.50:0.95], AP50, AP75
- **Canonical v1 checkpoint:** official `mask_rcnn_swin_tiny_patch4_window7_1x.pth` release asset

The upstream detection implementation also supports instance masks, but this pipeline's v1 contract is **object detection only**. Mask outputs are outside the normative task surface.

## Builder status

The upstream architecture, task, benchmark, config, and official checkpoint source are frozen. The repository now carries an executable, fail-closed scaffold. It deliberately does **not** emit a release manifest yet: the reviewed `ml-worker` contract has the object-detection task profile, but no canonical vision detection representation profile is present, and task-specific validator/finetuner releases do not yet exist.

**Lifecycle (DIMER Pipeline Specification 1.0):** `scaffold` — declared machine-readably in
`spec/pipeline-surface.json` (`dimerPipelineSpec`: `lifecycle_status`, intended
`implementation_topology` `COMPOSED-WORKERS`, `capability_modes` `GRADIENT-ADAPTATION`). The
canonical checkpoint `mask_rcnn_swin_tiny_patch4_window7_1x.pth` carries `redistribution_status: unknown` in
`provenance/open-weights.json` (`weightLicensing`), so DIMER hosting is **BLOCKED** until an
authoritative upstream weight-licence determination is recorded (LIC2/LIC7).
`scripts/verify_scaffold.py` refuses any lifecycle above `scaffold` while blockers exist, any
declared components/release, and any non-blocked hosting for an `unknown`/`prohibited` status.

Repository-owned build artifacts:

- `spec/pipeline-surface.json` — machine-readable task/model/contract surface and blockers;
- `provenance/open-weights.json` — exact official source revision, config blob, release asset identity, and checkpoint-digest state;
- `scripts/verify_scaffold.py` — verifies that the scaffold remains internally consistent and fail-closed;
- `.github/workflows/verify-scaffold.yml` — runs the verifier on pull requests and pushes to `main`.

Run locally:

```sh
python scripts/verify_scaffold.py
```

A passing scaffold check means the **specification is internally consistent**; it does not mean the detector is runtime-qualified or releasable.

## Cloud verification & open-weights digest (Kaggle)

The scaffold and official upstream weights were verified in an isolated Kaggle cloud container:

- **Kernel:** [`kurtvalcorza/swin-detection-verify`](https://www.kaggle.com/code/kurtvalcorza/swin-detection-verify)
- **Status:** `KernelWorkerStatus.COMPLETE` (Exit Code 0)
- **Scaffold Verifier (`scripts/verify_scaffold.py`):** `PASS` (internally consistent and fail-closed, 0.13s)
- **Official Checkpoint Asset:** `mask_rcnn_swin_tiny_patch4_window7_1x.pth` (191,487,694 bytes in 7.09s) from official release `SwinTransformer/storage@v1.0.3` (asset ID `36780486`)
- **Authoritative SHA-256:** `b67f9d6cd62a4d723c78faec1b49cbf548faa22437264defb00f2f6e54d21b78` (recorded in `provenance/open-weights.json`)
- **Deserialized State Dict Inspection:** 231 parameter tensors, 47,823,414 parameters (189 backbone, 16 neck, 6 RPN, 20 ROI)

## Canonical Swin family

1. **Image Classification** → `swin-classification-pipeline`
2. **Semantic Segmentation** → `swin-segmentation-pipeline`
3. **Object Detection** → `swin-detection-pipeline`

See issue #1 for the full NATIVE task, dataset, model, provenance, and conformance specification.
