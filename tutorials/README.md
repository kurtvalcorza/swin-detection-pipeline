# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | Release status |
|---|---|---|---|---|
| `swin_detection_scaffold_smoke_colab.ipynb` | `SMOKE` | Scaffold lifecycle, model-card, blocker, and open-weight provenance verification | CPU / Python 3.11+ | **Engineering-only** — clean execution is CI-gated, but this does not satisfy the primary release-grade tutorial requirement |

## Why this remains a smoke notebook

The repository lifecycle is explicitly `scaffold` and the current surface is `BLOCKED_PENDING_REPRESENTATION_AND_WORKERS`. The canonical COCO detection representation is absent, validator/finetuner releases do not exist, accelerator qualification is pending, no composition/release is emitted, and DIMER hosting of the upstream checkpoint is blocked while its redistribution status remains `unknown`.

Publishing an `E2E` or `TASK-INFERENCE` notebook before a real object-detection runtime exists would misrepresent the implementation. This notebook therefore limits itself to the repository-owned verifier plus lifecycle/model-card/provenance assertions. It does not download or deserialize the upstream `.pth` checkpoint.

The notebook is anchored to the stacked lifecycle + model-card revision used by PRs #5/#6. PR #4 should merge only after those prerequisite changes land (or after the anchor is repointed to their final `main` commit).

## Clean-runtime verification

`.github/workflows/verify-smoke-notebook.yml` executes the committed notebook top-to-bottom on Python 3.12 in a fresh GitHub-hosted Ubuntu/Jupyter environment whenever the notebook, registry, verifier, lifecycle/provenance files, model card, or workflow changes. A green run is execution evidence for the `SMOKE` profile only; it is not object-detection task-runtime evidence and does not make this repository tutorial-ready for end users.

The previous Kaggle execution record targeted the pre-lifecycle scaffold revision and is intentionally not carried forward as evidence for this notebook revision.

## Upgrade gate

A release-grade replacement requires the canonical detection representation and task workers, accelerator qualification, the real production-facing DIMER API, COCO-style input validation, COCO AP@[0.50:0.95]/AP50/AP75 evaluation, machine-readable detections and provenance, and clean-runtime execution evidence for the exact release revision.
