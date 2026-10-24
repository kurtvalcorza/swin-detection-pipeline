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
| Format / trust boundary | the upstream `.pth` is a pickled PyTorch checkpoint and is only the conversion source. After its size and SHA-256 are verified, `convert_checkpoint` reads it once with `torch.load(..., weights_only=True)`, takes `state_dict` (dropping the `meta` training record) with mmengine's own key handling (strip a leading `module.`; none of the 231 keys carries one), checks its `meta['CLASSES']` against the pinned `COCO_CLASSES`, loads it into the config-built model with `strict=True`, and writes a safetensors file. A matching digest proves byte identity with the pinned distribution, not publisher authenticity. |
| Served file: `mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.safetensors` | 191,434,712 bytes, SHA-256 `9ed16ad66c9ca031a67bbcfb4ae424fa069a29251de9aa6f38944823399a994b` (`CONVERTED_SHA256`; byte-reproducible: two conversions hash the same). Written next to the `.pth` in whichever directory the acquisition path uses (below); git-ignored (`weights/**/*.safetensors`). The detector is built with `init_detector(config, None)` (no checkpoint, so MMEngine's pickle loader is never called) and loaded from this file with `load_state_dict(strict=True)`. |

## Acquisition paths

- **Fleet snapshot path (standalone notebook, `from_pretrained(weights_dir=…, allow_download=…)`):**
  `stage_missing_files` fetches only manifest entries that are absent (the checkpoint, from the pinned URL above,
  and only with `allow_download=True`), `verify_snapshot` re-hashes every entry against the manifest and
  `MODEL_SPEC`, and `convert_checkpoint` writes the served safetensors file into the same directory; then
  `DimerSwinDetector(checkpoint=<converted .safetensors>, source="local-snapshot")` serves it. Once the converted file
  exists, the `.pth` is not staged, verified or read again. Both weight files are git-ignored (`weights/**/*.pth`,
  `weights/**/*.safetensors`); only the manifest is committed. In a repository checkout this directory is
  `weights/swin-t-mask-rcnn-coco/`; the standalone notebook uses `./weights/swin-t-mask-rcnn-coco/` under its working
  directory.
- **Cache path (`DimerSwinDetector(cache_dir=…)`, used by the `dimer-swin-detect` CLI):** `acquire_converted_weights`
  serves the converted file from `.dimer-models/` when it is there; otherwise `acquire_verified_checkpoint` downloads
  the `.pth` into `.dimer-models/`, verifies size + SHA-256 against `MODEL_SPEC`, and `convert_checkpoint` converts it
  there (`source="openmmlab-cache"`).

## Adaptation lineage (separate, not hosted)

The Microsoft/SwinTransformer release asset `mask_rcnn_swin_tiny_patch4_window7_1x.pth` (191,487,694 bytes, SHA-256
`b67f9d6cd62a4d723c78faec1b49cbf548faa22437264defb00f2f6e54d21b78`) recorded in `provenance/open-weights.json` belongs
to the still-scaffolded gradient-adaptation lineage, is **not** byte-identical to the inference checkpoint above, and
remains `dimer_hosting: BLOCKED` while its weight licence is `unknown`.
