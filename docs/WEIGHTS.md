# Weight provenance and hosting

The task-inference runtime uses one checkpoint, pinned by the fleet snapshot scheme and by the package's
own `MODEL_SPEC`; the two must agree and `verify_snapshot` refuses a manifest that disagrees with `MODEL_SPEC`.

| Item | Value |
|---|---|
| `MODEL_ID` | `open-mmlab/mmdetection:mask-rcnn_swin-t-p4-w7_fpn_1x_coco` — the config recipe `configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py` inside the pinned MMDetection 3.3.0 package (`resolve_packaged_config`) |
| `MODEL_REVISION` | `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a` — the `v3.3.0` release-tag commit of `open-mmlab/mmdetection`, i.e. the config source. The checkpoint bytes carry no git revision of their own; the manifest digest below is what pins them. |
| `MODEL_KEY` | `swin-t-mask-rcnn-coco` → `weights/swin-t-mask-rcnn-coco/` |
| Checkpoint host | `download.openmmlab.com` (not the Hugging Face Hub): `https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth` |
| Manifest | `weights/swin-t-mask-rcnn-coco/dimer-base-manifest.json` — 1 file, 191,461,353 bytes in total |
| `mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth` | 191,461,353 bytes, SHA-256 `9d6b7cfaa4aad52ef559611bea454f01d6f1f17c82a1abfac0d71631a193a291` (equal to `MODEL_SPEC["checkpoint_sha256"]`) |
| Weights licence | Apache-2.0 (OpenMMLab MMDetection model zoo) |
| Format / trust boundary | code-capable PyTorch `.pth`; size and SHA-256 are verified **before** the pinned MMDetection loader (`mmdet.apis.init_detector` → `mmengine` checkpoint loading, not a `weights_only` load) deserializes it. A matching digest proves byte identity with the pinned distribution, not publisher authenticity. |

## Acquisition paths

- **Fleet snapshot path (standalone notebook, `from_pretrained(weights_dir=…, allow_download=…)`):**
  `stage_missing_files` fetches only manifest entries that are absent (the checkpoint, from the pinned URL above,
  and only with `allow_download=True`), `verify_snapshot` re-hashes every entry against the manifest and
  `MODEL_SPEC`, then `DimerSwinDetector(checkpoint=<verified .pth>, source="local-snapshot")` runs the unchanged
  OpenMMLab init on the packaged config. The checkpoint is git-ignored (`weights/**/*.pth`); only the manifest is
  committed.
- **Cache path (`DimerSwinDetector(cache_dir=…)`):** `acquire_verified_checkpoint` downloads into
  `.dimer-models/` and verifies size + SHA-256 against `MODEL_SPEC` before use (`source="openmmlab-cache"`).

## Adaptation lineage (separate, not hosted)

The Microsoft/SwinTransformer release asset `mask_rcnn_swin_tiny_patch4_window7_1x.pth` (191,487,694 bytes, SHA-256
`b67f9d6cd62a4d723c78faec1b49cbf548faa22437264defb00f2f6e54d21b78`) recorded in `provenance/open-weights.json` belongs
to the still-scaffolded gradient-adaptation lineage, is **not** byte-identical to the inference checkpoint above, and
remains `dimer_hosting: BLOCKED` while its weight licence is `unknown`.
