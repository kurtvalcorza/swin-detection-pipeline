# Release verification

**Current status (2026-10-24): Candidate.** On 2026-10-24 the notebook `tutorials/swin_detection_task_inference.ipynb` was regenerated from code `2863aa7` (blob `a3725c8b7422`, commit `f300b25`) so that the detector is built from the packaged config with no checkpoint and served from the pinned safetensors conversion `mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.safetensors` (SHA-256 `9ed16ad66c9c…`, loaded with `load_state_dict(strict=True)`) instead of MMEngine's pickle load of the `.pth`. Two hosted Colab T4 runs of that exact blob on 2026-10-24 cover the default path and the optional Section 10 activity (`RUN_ACTIVITY = True` run copy); both passed in one pass with 0 errors (rows `f300b25` below). The remaining gates are REL12 (the BYOD review) and REL2 (the browser run of the interactive journeys). The runs of earlier blobs recorded below are history (REL14), not evidence for this one.

`tutorials/swin_detection_task_inference.ipynb` (`TASK-INFERENCE`, standalone) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence
under DIMER Notebook Specification 3.0. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

`.github/workflows/ci.yml` runs `ruff`, the offline unit suite (`tests/test_snapshot.py`, `tests/test_role_helpers.py`,
`tests/test_metrics.py`, `tests/test_notebook_parity.py`, `tests/test_review_fixes.py`, `tests/test_r86_review_fixes.py`,
`tests/test_spec3_migration.py`, `tests/test_val2017_release_assets.py` — the last one downloads the 50 release-asset images and parses them; OpenMMLab and pycocotools stubbed, no weights, no model),
`tools/validate_release_assets.py` and `tools/build_notebook.py --check` on Python 3.10. The validator checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE` profile, the notebook-spec
  version and the standalone carrier; `metadata.dimer` declares that profile, spec `3.0`, `notebook_kind: TUTORIAL`, `standalone: true` and
  `generated_from` (repository, revision, the two carried modules, their joined SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install, repository import, worker process or
  subprocess on the primary path (the generator-owned install cell excepted); one cell tagged `embedded_module` per
  carried module (`src/dimer_swin_detection/metrics.py`, then `runtime.py`) equal to the module after the generator's
  documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to
  `tools/pins.txt`; the notebook byte-identical to `tools/build_notebook.py` output; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cells (and repeated in the inline manifest, which
  the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same
  identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `DimerSwinDetector.from_pretrained(weights_dir=...)`, `validate_inputs`, `predict_many`, `evaluation_report`),
  the Python 3.10 assertion, the ceiling print (`MAX_PIXELS`, request ceilings, class count), the pinned val2017 sample
  (per-image SHA-256 fetch, ground-truth digest, `training_overlap`), atomic writes, the exports, the learner-facing statements (uncalibrated scores, no shipped threshold, score-ordered
  detections, the `.pth` trust boundary, `not-measurable` / `sample-sanity` verdicts), the gated-off `USE_BYOD` and
  `RUN_ACTIVITY` defaults listed in the validator, and the NOTEBOOK_SPEC 3.0 §7.2 section order (export, Interpretation and
  limits, Conclusion, the optional Section 10 activity and Section 11 BYOD, Troubleshooting, References); forbidden patterns (credential-in-URL, any `git clone` /
  `github.com/kurtvalcorza` / repository import, a mutable `revision='main'`, direct `mmdet` / `mmcv` / `mmengine` /
  `pycocotools` / `torchvision` / `transformers` / `huggingface_hub` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, any `torch.load(` without `weights_only=True` (and any `weights_only=False`), `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

These are source/provenance and unit checks. They are **not** execution evidence. The fleet CI venv used for the
lane gates has no OpenMMLab stack (`mmdet`, `mmcv`, `mmengine`, `pycocotools` are stubbed in the unit suite), so the
package's real loader and metric paths are exercised only by a notebook execution.

`.github/workflows/verify-task-tutorial.yml` executes the committed notebook with `nbconvert` on a GitHub-hosted
Python 3.10 runner from a scratch directory that contains only the notebook (no repository checkout on the notebook's
path) and asserts the four exports. A green run there is a clean-runtime execution of the default path and is the
promotion evidence to record below; a red run blocks release.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| GitHub Actions `verify-task-tutorial` (supported clean-room path) | `ubuntu-latest`, `actions/setup-python` 3.10, CPU; `nbconvert` from a scratch directory | The qualified Python 3.10 CPU runtime; a green run is promotion evidence once recorded here with the notebook blob |
| Jupyter on a CPython 3.10 kernel (user path) | Any Linux host with Python 3.10; CPU | The runtime the tutorial is written for; the notebook asserts the interpreter version |
| Google Colab (Colab CLI fresh VM, or a browser Run all) | Linux x86_64 VM; any kernel Python | Supported through the isolated uv-managed CPython 3.10.18 environment (Section 1); a Colab CLI T4 run is the hosted run of record for a revision (one is recorded below for blob `ea6b456d`) |
| Kaggle kernel | Linux x86_64; any kernel Python | Expected to work through the same isolated environment; no run recorded |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. run that exact notebook revision in a clean Linux x86_64 runtime (a fresh Colab VM through the Colab CLI, or the
   `verify-task-tutorial` workflow) with **no repository checkout** on the notebook's path and clean `weights/`,
   `sample/` and `outputs/` directories;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults for the
   sample path: `USE_BYOD = False`, `SAMPLE = 'val2017'`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `tools/pins.txt`: torch 2.1.2+cpu, mmcv 2.1.0, mmengine 0.10.7, mmdet 3.3.0, numpy 1.26.4);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` (PyPI plus the PyTorch CPU index and the OpenMMLab mmcv
     find-links) with no GitHub access;
   - the two carried module cells execute (define `DimerSwinDetector`, `coco_box_ap`, `validate_inputs`,
     `evaluation_report`) with no import of the repository package;
   - pinned checkpoint acquisition through the package: the inline `MANIFEST` is asserted against the module identity
     and written to `weights/swin-t-mask-rcnn-coco/`, `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports
     the one manifest entry on a clean runtime, `verify_snapshot` returns the manifest dict, and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` reports `source == 'local-snapshot'`;
   - Section 4 asserts Python 3.10 and prints `mmdet 3.3.0`, `mmcv 2.1.0`, `mmengine 0.10.7`, 80 classes;
   - the 50 COCO val2017 images are downloaded from release `data-coco-val2017-50-v1` and each matches its pinned size and SHA-256;
     the inline ground truth matches its digest (279 boxes over 59 categories); the sample print shows
     `COCO-val2017-50`, 33 CC BY 2.0 / 17 CC BY-SA 2.0 and `eval_data_in_training: False`;
   - `validate_inputs` writes `outputs/swin_detection_task_inference_input_manifest.json` (verdict `accepted`, 50
     accepted inputs and one recorded rejection finding from the missing-file probe);
   - detection through `predict_many(image_paths, score_threshold=0.0, max_detections=300)`;
   - `evaluation_report` writes `outputs/swin_detection_task_inference_evaluation_report.json` with verdict
     `sample-sanity`, `coco_box_ap` beside the empty-detector baseline, and `training_overlap`
     (`checkpoint_training_split: train2017`, `eval_data_in_training: false`);
   - `outputs/swin_detection_task_inference_result.json` and `outputs/swin_detection_task_inference_detections.csv`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, checkpoint digest, runtime versions and device;
6. verify the exports exist under their default names (no `_exploratory_` or `.part` file) and the interpretation
   section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, mmdet/mmcv/mmengine, device), model
   identifier and immutable revision, whether the weights directory was clean, outcome, produced outputs, and any
   warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

The synthetic scene (`SAMPLE = 'synthetic'`, verdict `not-measurable`) and BYOD are optional exploratory paths, not
part of the default-path gate. They write `_exploratory_<n>` files and never replace the default report; a run of
them may be recorded as additional evidence. A refused BYOD input moves earlier *exploratory* outputs to
`outputs/earlier_<n>/`; the default sample's files are never moved. Uploads are saved under `sample/byod/`.

## Default scored sample (review row 86, 2026-10-18)

Until 2026-10-18 the default scored sample was COCO8's `images/val` (Ultralytics), whose four files are COCO
**train2017** images 36, 42, 49 and 61. The checkpoint (`mask-rcnn_swin-t-p4-w7_fpn_1x_coco`, MMDetection v3.3.0) was
trained on `instances_train2017.json`, so every AP recorded below for COCO8 (0.707 / 0.957 / 0.699) is a
resubstitution score on training images, not a measurement. Maintainer decisions R86-D1–D3 replaced it:

- **Rule (fixed before any score was seen):** COCO val2017 images with licence id in {4 CC BY 2.0, 5 CC BY-SA 2.0,
  7 no known copyright restrictions, 8 US Government work}, at least one non-crowd box, no crowd box, and no
  train2017 match by image id, Flickr photo id or (width, height, date_captured); 1,154 images qualify; the first
  50 by image id are used (279 boxes, 59 categories, 8,346,599 bytes; 33 CC BY 2.0, 17 CC BY-SA 2.0).
- **Pin:** `tools/coco_val2017_sample_50.json` (per-image SHA-256, bytes, licence, Flickr and COCO URLs, ground
  truth as `[class index, x1, y1, x2, y2]`; ground-truth digest `2f5e416a…`, `b6057ca6…` in the reviewer's
  `[x, y, w, h]` form; source `annotations_trainval2017.zip` SHA-256 `113a836d…`). The fixer re-derived the rule and
  the 50 ids from the annotations zip, re-downloaded every image over http from `images.cocodataset.org`, and
  matched all 50 SHA-256 values and decoded sizes before uploading.
- **Hosting:** assets of the release [`data-coco-val2017-50-v1`](https://github.com/kurtvalcorza/swin-detection-pipeline/releases/tag/data-coco-val2017-50-v1) (the 50 JPEGs, `ATTRIBUTION.md`, the pin).
  The notebook downloads from there and refuses any file whose size or SHA-256 differs;
  `images.cocodataset.org` (no valid TLS certificate at review time) is recorded for provenance only.
- **Disclosure:** val2017 is the split upstream used to report box AP 42.7 and possibly for model selection, so the
  result is a sanity measurement on images outside the detector's training split (train2017), not an independent
  test. The overlap screen used ids and capture metadata, not pixels, and covered train2017 only: the images were not
  screened against the ImageNet-1K data the backbone was pretrained on.
- **Why the sample scores above 0.427 (peer review F2, 2026-10-19):** the rule excludes crowd-annotated images, so the
  sample has 5.58 boxes per image against 7.27 in all of val2017 and about 35 % small boxes against 42 % (val2017
  figures measured by the peer reviewer from `instances_val2017.json`); 33 of its 59 categories have one or two boxes,
  so the category-mean AP is volatile. A 200-resample bootstrap over images of the development-run detections gives a
  95 % interval of 0.502–0.599, which excludes 0.427: the gap is mostly a selection effect, not sampling noise. The
  notebook now prints this interval (`bootstrap_over_images`) and treats the empty detector as a floor, not a baseline.
- **CI:** `verify-task-tutorial` asserts the new pool (50 images, 279 boxes, the ground-truth digest, licences,
  `eval_data_in_training: false`, no exploratory files). Its runs of an unchanged notebook are reproductions, not
  new measurements. The hosted Colab T4 run of the new blob is the run of record.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/swin_detection_task_inference.ipynb` (verify with
`git rev-parse <commit>:tutorials/swin_detection_task_inference.ipynb`).

### Standalone carrier — clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `181c09c` / `966c4ad9a255` | GitHub Actions `verify-task-tutorial` run [`34769953971`](https://github.com/kurtvalcorza/swin-detection-pipeline/actions/runs/34769953971) (job `103757597810`), Ubuntu 24.04.5, CPython 3.10.19 kernel, CPU | Default sample path (in-kernel pinned install; synthetic scene) | 84.0 s | **PASSED** — 10/10 code cells executed cleanly in one pass, checkpoint verified & loaded, 4 outputs generated, evaluation verdict `not-measurable` on the synthetic sample; the workflow's assertions confirm `runtime.python` 3.10.x, `runtime.mmdet` 3.3.0 and the checkpoint SHA-256. The other library versions and the detection count were printed in the job log but not copied here, and the log host was not reachable when this record was corrected; the cell count was earlier mis-recorded as 17/17. |
| 2026-10-10 (11:41:13 UTC start) | `f5a6c1726bfdddd381dfcbd47a6d697fcebebf46` / `ea6b456dff309207c007097defa97a599002e2e9` (`NOTEBOOK_SOURCE.repository_revision` `b6d284425ca7`, `module_sha256` `92d2e49d5f2b…`, generator `build_notebook.py/2.1-swd`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-swin-f5a6c17-65d6`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated uv-managed CPython 3.10.18 (45 locked packages, setup 15 s), `torch 2.1.2+cpu`, MMDetection 3.3.0, MMCV 2.1.0, MMEngine 0.10.7, device `cpu` (by design; the T4 is unused), `restarted: false` | Default COCO8 path, every form field at its default (BYOD off) | 108.0 s | **PASSED** — one pass, no restart, 0 errors; 12/12 code cells in order (cells 4–5, the carried modules, print nothing); 1 checkpoint file digest-verified at `44ebd17b145c`; COCO8 val 4 images / 17 ground-truth boxes; `does-not-exist.png` refused by `validate_inputs`; 88 detections at threshold 0.0 (4 / 20 / 39 / 25 per image); COCO box AP@[.50:.95] 0.707, AP50 0.957, AP75 0.699 (AR@100 0.717) beside the empty-detector baseline 0.0; verdict `sample-sanity`; 4 outputs written. Same AP values as the 2026-09-11 GitHub Actions run of the earlier carrier. Evidence in `docs/execution-evidence/2026-10-10-f5a6c17/`: executed notebook SHA-256 `80e0000c3fba6ec96ad202a62640da7c723290301e3244258473446479e96d51`, `run_summary.json` `a98152e3efe0c4c0bb8fde455862941f88f0e6508a9be42f1aba7282a3814d0f`, `exec.log` `ab08d17d231ca2636dadc79f1a23a05bc8cea42b4fbc752f1ccfabe4eb83cd63` (two lines redacted on 2026-10-18 to replace the runner's home-directory path with `<runner-home>`; the log as executed had SHA-256 `90546e774706e2997a29dfec506573da9af1809dbbd3dcd521952fa690538d5f`). Not exercised: the synthetic-scene path, BYOD, a browser Run all |
| 2026-10-19 (16:17:00 UTC start) | `a27a4fe4f8c73091e271d3c6083c4e6992acd161` / `7775654cc12b84518eaaf95266c30781700a7210` (`NOTEBOOK_SOURCE.repository_revision` `5afcbe7d57e5`, `module_sha256` `92d2e49d5f2b…`, generator `build_notebook.py/2.2-swd`, `notebook_spec` 2.5) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-swin-a27a4fe-65d6`), committed blob fetched at the commit, input notebook unchanged by the run (`input_unchanged: true`); kernel Python 3.13.15, isolated uv-managed CPython 3.10.18 (45 locked packages, setup 12 s), `torch 2.1.2+cpu`, NumPy 1.26.4, MMDetection 3.3.0, MMCV 2.1.0, MMEngine 0.10.7, device `cpu` (by design; the T4 is unused) | Default val2017 path (release `data-coco-val2017-50-v1`), every form field at its default (BYOD off) | 489.7 s | **PASSED — run of record for blob `7775654c`.** One pass, exit code 0, 0 errors; 12/12 code cells in order (cells 4–5, the carried modules, print nothing); the executed notebook's 31 cell sources equal blob `7775654c`; 1 checkpoint file at revision `44ebd17b145c`; 50 val2017 images / 279 ground-truth boxes / 59 categories; `does-not-exist.png` refused by `validate_inputs`; 1,252 detections at threshold 0.0 on 50 of 50 images (at most 100 per image); COCO box AP@[.50:.95] **0.538** (`0.5378428165820687`), AP50 0.801, AP75 **0.584** (AR@100 0.622) beside the empty-detector baseline 0.0; `bootstrap_over_images` (200 resamples, seed 0) 95 % interval **[0.5018, 0.5992]**; verdict `sample-sanity`; `eval_data_in_training: false`; 4 outputs written. Source of the quoted values: the evaluation report printed in `exec.log` and the executed notebook. The AP values equal the CPU and CI development runs of blob `a4c94fca` to ten digits. Evidence in `docs/execution-evidence/2026-10-19-a27a4fe/`: executed notebook SHA-256 `746fd25c7e991fc4d0383538438756130185aba9f97ed22a38e282d56a5a704a` (unchanged from capture, equal to `run_summary.json` `executed_sha256`), `run_summary.json` `7d722b2e1408c8d90839fb3fda59d6f7b027ff8397e3984065fba5af33a89002` (unchanged from capture), `exec.log` `439a3fcb0d2a4452591c2574fdbbae7909cc02d91e965b0018e050f6b47361a1` (two local run-directory paths replaced with `<local-run-dir>/` before the first commit; the log as captured had SHA-256 `9c908cf44f4fe38c559d16caedfed23ed367184304dae15bc3b6e3d97defd1f4`). Not exercised: the synthetic-scene path, BYOD, a browser Run all |
| 2026-10-22 (07:38:46 UTC start) | `1fcdbd4524ab776e5c75e5ba74b312a7bf9c05c6` / `3cfa37aeeaa997f7667851e720f4ed2ba49fda75` (badge-wording revision, PR #16; evaluation code unchanged since the run of record) | Colab CLI 0.7.4 `colab exec -f`, fresh Colab Tesla T4 | Default path (50 pinned COCO val2017 images) | 496.4 s | **PASS**, exit 0, input unchanged; executed cell sources equal the blob (31 cells). Box AP@[0.50:0.95] 0.538, image-bootstrap 95 % interval [0.5018, 0.5992], identical to the 2026-10-19 run of record. Evidence: `docs/execution-evidence/2026-10-22-1fcdbd4/` — executed notebook SHA-256 `0985962d58289015d67a37fd3b0083b7fcc2845036358262918d16c67d8a70ab`, `run_summary.json` `307df9b0dfa35b690d64ade219422a995df8e8c8c850bc210cca04f00a107b03`, `exec.log` captured `4eb0956fb5513fca23f6da1b12f7012b2a8ec121b518aee6d37b279483473d0f`, committed `3bc48c9a0492afda91619141a64671b963a10212337e530e12fb03388248252b` after the local run directory was replaced by `<local-run-dir>` |
| 2026-10-22 (13:01:54 UTC start) | `faf8edbefdaa8aacfcd3d24212ebb7db7c14f2db` / `ea2ff3b34faeb66fe582a068118c98ae0d35b59d` (Notebook Specification 3.0 migration, PR #17) | Colab CLI 0.7.5 `colab exec -f` (not a browser Run all), fresh Colab Tesla T4 | Default path (50 pinned COCO val2017 images); `RUN_ACTIVITY` and `USE_BYOD` off, both printed as skipped | 469.1 s | **PASS**, exit 0, input unchanged, 14/14 code cells in order per `exec.log` (two silent cells carry no output). Box AP@[0.50:0.95] 0.538, image-bootstrap 95 % interval [0.5018, 0.5992]. Evidence: `docs/execution-evidence/2026-10-22-faf8edb/` (captured → committed SHA-256, local run directory redacted to `<local-run-dir>`): `exec.log` `b19b1261e45b…` → `1199b3157384…`; `run_summary.json` `fde2e9866133…` (unchanged); `swin_detection_task_inference_faf8edb_colab-cli-t4_output.ipynb` `5981c1975b39…` (unchanged) |
| 2026-10-22 (13:44:14 UTC start) | `9de0f7e0e0f8d495809edc86916178272949d8e9` / committed blob `ea2ff3b34faeb66fe582a068118c98ae0d35b59d` (the same blob as the `faf8edb` row), executed as a run copy with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `b62112f45760c06114a2641152dc0cefff1c1121`, SHA-256 `db17d8d056dc…`; it differs from the committed blob only in that assignment literal) | Colab CLI 0.7.5 `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 | Default path (50 pinned COCO val2017 images), then the optional Section 10 activity on the synthetic scene; `USE_BYOD` off (printed as skipped) | 600.9 s | **PASS**, exit 0, input (the run copy) unchanged, 14/14 code cells in order per `exec.log`, 0 errors. Default path: 1,252 detections, box AP@[0.50:0.95] 0.538 (`0.5378428165820687`). Section 10 activity: `synthetic_scene_640x480.png` validated (`missing-file-probe` refused), 1 detection ("class 29 frisbee score 0.5581"), verdict `not-measurable` ("no ground-truth boxes were supplied for the evaluated images"), outputs written with the `_exploratory_1` suffix. Evidence: `docs/execution-evidence/2026-10-22-9de0f7e-activity/` (SHA-256 table below the clean-runtime table) |
| 2026-10-23 (08:25:21 UTC start) | `88a939c6ec184e5634cbae1ac91bbbb29c5780f5` / `0dd698fc02cafc5d6ed5f5ebae863da2c2aef868` (refusal-display fix, PR #18) | Colab CLI 0.7.5 `colab exec -f` (not a browser Run all), fresh Colab Tesla T4 (session `suite-swin-88a939c-65d6`) | Default path (50 pinned COCO val2017 images); `RUN_ACTIVITY` and `USE_BYOD` off, both printed as skipped | 486.0 s | **PASS**, exit 0, input unchanged, 14/14 code cells in order per `exec.log` (two silent cells carry no output), 0 errors. 1,252 detections on 50 of 50 images; box AP@[0.50:0.95] 0.538 (`0.5378428165820687`), AP50 0.801, AP75 0.584, image-bootstrap 95 % interval [0.5018, 0.5992]; verdict `sample-sanity`. Evidence: `docs/execution-evidence/2026-10-23-88a939c/` (captured → committed SHA-256, local run directory redacted to `<local-run-dir>`): `exec.log` `d27a64a7f1d7…` → `8ab1e206a810…` (2 replacements); `run_summary.json` `04a489271970…` (unchanged); `swin_detection_task_inference_88a939c_colab-cli-t4_output.ipynb` `dfb86a940c06…` (unchanged, equal to `run_summary.json` `executed_sha256`) |
| 2026-10-23 (08:25:32 UTC start) | `88a939c6ec184e5634cbae1ac91bbbb29c5780f5` / committed blob `0dd698fc02cafc5d6ed5f5ebae863da2c2aef868` (the same blob as the row above), executed as a run copy with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `c97ad8279013b11eab0f2303da16cfb485fcead8`, SHA-256 `720018227138…`; it differs from the committed blob only in that assignment literal) | Colab CLI 0.7.5 `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4 (session `suite-swin-88a939c-65d6`) | Default path (50 pinned COCO val2017 images), then the optional Section 10 activity on the synthetic scene; `USE_BYOD` off (printed as skipped) | 477.7 s | **PASS**, exit 0, input (the run copy) unchanged, 14/14 code cells in order per `exec.log`, 0 errors. Default path: 1,252 detections, box AP@[0.50:0.95] 0.538 (`0.5378428165820687`). Section 10 activity: `synthetic_scene_640x480.png` validated (`missing-file-probe` refused), 1 detection ("class 29 frisbee score 0.5581"), verdict `not-measurable`, outputs written with the `_exploratory_1` suffix. Evidence: `docs/execution-evidence/2026-10-23-88a939c-activity/` (SHA-256 table below the clean-runtime table) |
| 2026-10-24 (09:16:22 UTC start) | `f300b2523caf0ee51faa78cc4d967e9736acd371` / `a3725c8b74225309514d4088a7abdd7e7aadb40c` (safetensors serving, PR #19; `NOTEBOOK_SOURCE.repository_revision` `2863aa71d4a8`, `module_sha256` `e746e28af640…`, generator `build_notebook.py/2.3-swd`, `notebook_spec` 3.0) | Colab CLI `colab exec -f` (not a browser Run all), fresh Colab Tesla T4; kernel Python 3.13.16, isolated CPython 3.10.18 (46 locked packages, setup 15 s), `torch 2.1.2+cpu`, NumPy 1.26.4, MMDetection 3.3.0, MMCV 2.1.0, MMEngine 0.10.7, device `cpu` (by design) | Default path (50 pinned COCO val2017 images); `RUN_ACTIVITY` and `USE_BYOD` off, both printed as skipped | 503.4 s | **PASS**, exit 0, input unchanged, 14/14 code cells in order per `exec.log` (two silent cells carry no output), 0 errors. Staging printed `fetched` the `.pth` (191,461,353 B) and `verified_files` 1; the runtime record printed `'checkpoint': 'mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.safetensors'`, `'source': 'local-snapshot'`, `'device': 'cpu'`, 80 classes. 1,252 detections on 50 of 50 images; box AP@[0.50:0.95] 0.538 (`0.5378428165820687`), AP50 0.801, AP75 0.584, image-bootstrap 95 % interval [0.5018, 0.5992]; empty-detector baseline 0.0; verdict `sample-sanity` — identical to the runs of earlier blobs. Evidence: `docs/execution-evidence/2026-10-24-f300b25/` (SHA-256 table below the clean-runtime table) |
| 2026-10-24 (09:17:40 UTC start) | `f300b2523caf0ee51faa78cc4d967e9736acd371` / committed blob `a3725c8b74225309514d4088a7abdd7e7aadb40c` (the same blob as the row above), executed as a run copy with one override, `RUN_ACTIVITY = False` → `True` (run copy blob `090bd4f9c87595b35d204db5366adc2ed9d7c011`, SHA-256 `c1b9d8027633…`; it differs from the committed blob only in that assignment literal) | Colab CLI `colab exec -f` (a scripted run, not a browser Run all), fresh Colab Tesla T4; isolated CPython 3.10.18 (setup 12 s), device `cpu` | Default path (50 pinned COCO val2017 images), then the optional Section 10 activity on the synthetic scene; `USE_BYOD` off (printed as skipped) | 496.3 s | **PASS**, exit 0, input (the run copy) unchanged, 14/14 code cells in order per `exec.log`, 0 errors. Default path: 1,252 detections, box AP@[0.50:0.95] 0.538 (`0.5378428165820687`). Section 10 activity: `synthetic_scene_640x480.png` validated (`missing-file-probe` refused), 1 detection ("class 29 frisbee score 0.5581"), verdict `not-measurable`, outputs written with the `_exploratory_1` suffix. Evidence: `docs/execution-evidence/2026-10-24-f300b25-activity/` (SHA-256 table below the clean-runtime table) |

The 2026-09-14 row is evidence for the in-kernel-install revision only; the 2026-10-10 row is evidence for blob `ea6b456d` (isolated uv environment, COCO8 default). Both predate the 2026-10-18 sample change: the 2026-10-10 AP values are resubstitution scores on four train2017 images (see **Default scored sample** above). The 2026-10-19 row is the run of record for blob `7775654c` (the 2026-10-18 revision after round 2 of the review).

The 2026-10-22 `9de0f7e` row is the optional-activity run of blob `ea2ff3b3` (owner ruling of 2026-10-22: the activity runs once on
the current blob before promotion). Its run copy is committed beside the evidence as `run_copy.ipynb`. Captured → committed SHA-256
(local run directory redacted to `<local-run-dir>`):

| File (under `execution-evidence/2026-10-22-9de0f7e-activity/`) | Captured | Committed |
|---|---|---|
| `swin_detection_task_inference_9de0f7e_colab-cli-t4_activity.ipynb` | `38d4d15a3686c820923cba95bdbe34d2845342174d0bf36158a7ffd60fb1f25c` | unchanged (no local path) |
| `exec.log` | `ce4743893d7ba31f55087199b88372af439982b3c33d2d94e6ab8d9e9f1ffc15` | `2cee221dda6a552d48fd92b5eac90a1e29b8ec3450e8ae1d4e5fa3ad273e28ae` (2 replacements) |
| `run_summary.json` | `697023b025c405b73c702058a27cf4ba6e3147f8ce106c443cf8d2db097589f1` | unchanged (no local path) |
| `run_copy.ipynb` | `db17d8d056dc8c2aa45d7a67f98792b6e3fe9ae159b95ca90dedc32b1cee5607` | unchanged (no local path) |

The 2026-10-23 `88a939c` activity row is the optional-activity run of blob `0dd698fc`, the blob before the 2026-10-24 safetensors regeneration. Its run copy is
committed beside the evidence as `run_copy.ipynb`. Captured → committed SHA-256 (local run directory redacted to `<local-run-dir>`):

| File (under `execution-evidence/2026-10-23-88a939c-activity/`) | Captured | Committed |
|---|---|---|
| `swin_detection_task_inference_88a939c_colab-cli-t4_activity.ipynb` | `9ba9a5d4ee5212eaa4d5364e0b77fa89041329d76c763490910bfc728a56fc5e` | unchanged (no local path; equal to `run_summary.json` `executed_sha256`) |
| `exec.log` | `5282ae54c4a3209ccbbde6121d7964ec73ba1a3567f705176c20515c7e34174f` | `54a9decef86b57229b5d950061633a4fbf826b165332e4a8138982ad638fba18` (2 replacements) |
| `run_summary.json` | `a61a53618922481c0971a0b1552f45add121808517ecb6c6404b2246ecaf1cff` | unchanged (no local path) |
| `run_copy.ipynb` | `720018227138a65fb7a93c7c84ae145d8f25974ba6752233ed773be47163068d` | unchanged (no local path) |

The 2026-10-24 `f300b25` rows are the default-path run and the optional-activity run of blob `a3725c8b`, the current blob. The activity
run copy is committed beside its evidence as `run_copy.ipynb`. Captured → committed SHA-256 (local run directory redacted to
`<local-run-dir>`, byte-wise):

| File (under `execution-evidence/2026-10-24-f300b25/`) | Captured | Committed |
|---|---|---|
| `swin_detection_task_inference_f300b25_colab-cli-t4_default.ipynb` | `972517c9b74cfa5cc0dea9f6d64eb5479ee8c66a8b50d0af536221a86e968a10` | unchanged (no local path; equal to `run_summary.json` `executed_sha256`) |
| `exec.log` | `f8c743be2bc28ccee5d7508bad7e16dfa3529a121a39df2f67d0018feee859b3` | `0aa0c11d7e091b2e0f839c8fa98476b4e07a2f44227fb4ae962e62b939848384` (2 replacements) |
| `run_summary.json` | `a2edd0d5c2d83f9b5db1f271ccfa8591d889628e70404f6a6d5195ecfccab12e` | unchanged (no local path) |

| File (under `execution-evidence/2026-10-24-f300b25-activity/`) | Captured | Committed |
|---|---|---|
| `swin_detection_task_inference_f300b25_colab-cli-t4_activity.ipynb` | `f2bd57bcd5a6f0941439699edc6c167f50d9b9754c7725adb0de1bc8313b9dd6` | unchanged (no local path; equal to `run_summary.json` `executed_sha256`) |
| `exec.log` | `60e546a980efdd30e14926b0b0dc29678fd83f939d47c939f7ebeedc85c6b1bc` | `a06505b87c3c24555241591f3d1f44bf40dc41f209d319f7790c7e087364e35f` (2 replacements) |
| `run_summary.json` | `96dd7bcea800adcf44e72d556ced328f3022a08e69aebf886b8109bebe2e081c` | unchanged (no local path) |
| `run_copy.ipynb` | `c1b9d802763335bcaa4e432aad1e4c10d76c98605a7c9eaf754d5b381f9b55fc` | unchanged (no local path) |

### CPU development runs of the 2026-10-18 revision (not the run of record)

These two runs scored the new val2017 sample on CPU while the fix was being built. The maintainer allowed it as a
development check (an inference-only measurement has no score-once rule). They are labelled here so that the look is
disclosed. The hosted Colab T4 run of the final blob remains the run of record.

| Date (UTC) | Commit / notebook blob | Executor | Outcome |
|---|---|---|---|
| 2026-10-18 | `7325021` / `a4c94fca64d6` (`repository_revision` `1628f8b`) | GitHub Actions `verify-task-tutorial` run [`39168312094`](https://github.com/kurtvalcorza/swin-detection-pipeline/actions/runs/39168312094), nbconvert from a scratch directory, isolated CPython 3.10.18, torch 2.1.2+cpu, MMDetection 3.3.0, CPU | **PASSED** (4m21s job); 50 images / 279 boxes; AP@[.50:.95] 0.5378, AP50 0.8013, AP75 0.5838; empty-detector baseline 0.0; verdict `sample-sanity`; `eval_data_in_training: false` |
| 2026-10-18 | `7325021` / `a4c94fca64d6` | **CPU development run**: nbconvert in WSL2 Ubuntu (glibc 2.39, 24 vCPU, GPU hidden with `CUDA_VISIBLE_DEVICES=-1`), the same isolated CPython 3.10.18 stack, executed-notebook SHA-256 `5be39ab3…` | **PASSED** (420 s); 1,252 detections; AP@[.50:.95] 0.5378428166, AP50 0.8013336406, AP75 0.5837752546, the same as the CI run; baseline 0.0; verdict `sample-sanity` |

The CI run on the PR head `69e81fd` (`verify-task-tutorial` run
[`39168814777`](https://github.com/kurtvalcorza/swin-detection-pipeline/actions/runs/39168814777), the same blob
`a4c94fca`) reproduced AP@[.50:.95] 0.5378428166. Every CI run of an unchanged blob is a reproduction of these values.
Round 2 of the review (2026-10-19) changes the blob again (bootstrap interval, wording, attribution), so these runs
are evidence for blob `a4c94fca` only.

Read these values as a sanity measurement on 50 val2017 images, which is the split upstream reported 42.7 on. They are
not an independent test. A bootstrap over images of these detections gives a 95 % interval of 0.502–0.599.

### Previous carrier (Notebook Specification 1.0, repository-installing) — audit trail only

| Date (UTC) | Commit | Executor | Path exercised | Outcome |
|---|---|---|---|---|
| 2026-09-11 | PR head `354545b89fef6934d01eb932d868fc3e1260d830` | GitHub Actions `verify-task-tutorial` run `34574579363`, Ubuntu 24.04.5, CPython 3.10.19, CPU | repository clone + pinned install, COCO8 `images/val` (4 COCO train2017 images, i.e. training images of this checkpoint; 17 GT boxes) | success — COCO AP@[0.50:0.95] `0.7073101933`, AP50 `0.9570957096`, AP75 `0.6993399340`; empty-detector baseline AP `0.0` |

That run exercised a notebook that cloned this repository and evaluated COCO8 by default; it is evidence for that
carrier and for the OpenMMLab inference path, not for the standalone notebook above.

## Current status

The registry status is **Candidate under Notebook Specification 3.0**. The current blob is `a3725c8b` at commit `f300b25` (the 2026-10-24 safetensors regeneration at the top of this file), and two hosted Colab T4 runs of it (2026-10-24, both PASS) cover the default path (box AP@[0.50:0.95] 0.538, image-bootstrap 95 % interval [0.5018, 0.5992]) and the optional Section 10 activity (`RUN_ACTIVITY = True` run copy); promotion is pending only REL12 (BYOD review) and REL2 (browser run of the interactive journeys). Earlier blobs' runs are history (REL14). Before it, the 2026-10-23 refusal-display fix (branch `fix/refusal-display`: a refused BYOD input is shown as one `Refused: <file name>: <rule>` line with no traceback, and the BYOD image must decode as PNG, JPEG or WebP) changed the notebook blob from `ea2ff3b3` to `0dd698fc`, so under REL14 the `faf8edb` default-path run and the `9de0f7e` activity run are history for blob `ea2ff3b3`, not evidence for `0dd698fc`. The previous blob is `0dd698fc` at commit `88a939c`, and two hosted Colab T4 runs of it (Colab CLI 0.7.5, 2026-10-23, both PASS) cover the default path (box AP@[0.50:0.95] 0.538, image-bootstrap 95 % interval [0.5018, 0.5992]) and the optional Section 10 activity (`RUN_ACTIVITY = True` run copy); both are recorded in `docs/release-verification.md`. Promotion is pending only (1) the REL12 BYOD review (one representative image accepted through the Section 6–9 stages and one incompatible input refused, in the hosted runtime) and (2) the REL2 browser run of the interactive journeys (the upload dialog and the `RUN_ACTIVITY` / `USE_BYOD` switches).

History, for blob `ea2ff3b3`: the 2026-10-22 migration (branch `spec3/migrate`)
declares `notebook_spec` 3.0 and moves the optional activity (Section 10, `RUN_ACTIVITY`, off by default) and BYOD (Section 11,
`USE_BYOD`, off by default) after the interpretation, as §7.2 orders them, each running the same stage functions as the default
path. The default path's computations and outputs are otherwise unchanged, but the notebook blob changed, so under REL14 only the 2026-10-22 `faf8edb` row is
evidence for it. The hosted Colab T4 run of the new blob `ea2ff3b3` at `faf8edb` (2026-10-22, Colab CLI) passed the default path (box AP@[0.50:0.95] 0.538). The optional Section 10 activity ran on the same blob in the `9de0f7e` run (`RUN_ACTIVITY = True` run copy, Colab CLI 0.7.5, fresh T4, 2026-10-22, PASS), which satisfies the activity gate. For that blob, promotion was pending only (1) the REL12 BYOD review (one representative image accepted through the Section 6–9 stages and one incompatible input refused, in the hosted runtime) and (2) the REL2 browser run of the interactive journeys (the upload dialog and the `RUN_ACTIVITY` / `USE_BYOD` switches). The notebook has no external-artifact section: it exports
predictions, not a reusable artifact, and this is not one of the nine companion pipelines, so XAS1 does not apply.

History, for the Notebook Specification 2.5 blobs: the 2026-10-18 revision (review row 86) replaced the scored sample with 50
pinned COCO val2017 images, migrated the notebook to Notebook Specification 2.5, and its run of record is the
2026-10-19 hosted Colab T4 run of blob `7775654c` (commit `a27a4fe`; table above): box AP@[.50:.95] 0.538
(`0.5378428165820687`) and AP75 0.584, with a 95 % image-bootstrap interval of [0.5018, 0.5992], as printed in
`docs/execution-evidence/2026-10-19-a27a4fe/exec.log` and the executed notebook beside it. These are sanity measurements on
50 val2017 images, not an independent test. Development values (CPU and CI runs) are listed above, each labelled with its blob. Earlier hosted runs (2026-09-14, 2026-10-10) do not transfer to it, and their COCO8
AP values are resubstitution scores. Static validation (`tools/validate_release_assets.py`), the generator parity
check and the offline unit suite are necessary but not sufficient. Promotion is performed by a reviewer/integrator,
not by the builder. A hosted run must record `restarted: false`, the library versions, the detection count, the
val2017 AP values beside the empty-detector baseline, and the notebook blob.
