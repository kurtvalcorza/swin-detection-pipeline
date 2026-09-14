# Release verification

`tutorials/swin_detection_task_inference.ipynb` (`TASK-INFERENCE`, standalone) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence
under DIMER Notebook Specification 1.1. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

`.github/workflows/ci.yml` runs `ruff`, the offline unit suite (`tests/test_snapshot.py`, `tests/test_role_helpers.py`,
`tests/test_metrics.py`, `tests/test_notebook_parity.py` — OpenMMLab and pycocotools stubbed, no weights, no model),
`tools/validate_release_assets.py` and `tools/build_notebook.py --check` on Python 3.10. The validator checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE` profile, the notebook-spec
  version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`, `standalone: true` and
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
  the Python 3.10 assertion, the ceiling print (`MAX_PIXELS`, request ceilings, class count), the pinned COCO8 archive
  digest, the exports, the learner-facing statements (uncalibrated scores, no shipped threshold, score-ordered
  detections, the `.pth` trust boundary, `not-measurable` / `sample-sanity` verdicts) and the gated-off `USE_BYOD` /
  `USE_COCO8` defaults listed in the validator; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com/kurtvalcorza` / repository import, a mutable `revision='main'`, direct `mmdet` / `mmcv` / `mmengine` /
  `pycocotools` / `torchvision` / `transformers` / `huggingface_hub` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
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
| Google Colab | Default Colab runtimes ship Python 3.11+ | **Unsupported**: the OpenMMLab wheels exist for Python 3.10 only and the pinned install fails; the badge is kept for the file location, not as a supported executor |
| Kaggle CLI kernel | Kaggle images ship Python 3.11+ | **Unsupported** for the same reason |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. run that exact notebook revision in a clean CPython 3.10 CPU runtime with **no repository checkout** on the
   notebook's path and a clean `weights/` directory (the `verify-task-tutorial` workflow does this);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults for the
   sample path: `USE_BYOD = False`, `USE_COCO8 = False`);
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
   - the synthetic 640×480 scene is generated in code with its SHA-256 printed;
   - `validate_inputs` writes `outputs/swin_detection_task_inference_input_manifest.json` (verdict `accepted`, one
     recorded rejection finding from the missing-file probe);
   - detection through `predict_many(image_paths, score_threshold=0.0, max_detections=300)`;
   - `evaluation_report` writes `outputs/swin_detection_task_inference_evaluation_report.json` with verdict
     `not-measurable` on the synthetic sample (no ground truth), stated as such;
   - `outputs/swin_detection_task_inference_result.json` and `outputs/swin_detection_task_inference_detections.csv`
     written with `NOTEBOOK_SOURCE`, model revision, model licence, checkpoint digest, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, mmdet/mmcv/mmengine, device), model
   identifier and immutable revision, whether the weights directory was clean, outcome, produced outputs, and any
   warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

The gated `USE_COCO8` path (labelled COCO8 validation subset, `coco_box_ap` via pycocotools) is not part of the
default-path gate; a separate run with the gate enabled may be recorded as additional evidence.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/swin_detection_task_inference.ipynb` (verify with
`git rev-parse <commit>:tutorials/swin_detection_task_inference.ipynb`).

### Standalone carrier (Notebook Specification 1.1) — clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `181c09c` / `966c4ad9a255` | GitHub Actions `verify-task-tutorial` run `34769953971`, Ubuntu 24.04.5, CPython 3.10.19, CPU | Default sample path | 84.0 s | **PASSED** — 17/17 code cells executed cleanly, checkpoint verified & loaded, 4 outputs generated, evaluation verdict `not-measurable` on synthetic sample |

### Previous carrier (Notebook Specification 1.0, repository-installing) — audit trail only

| Date (UTC) | Commit | Executor | Path exercised | Outcome |
|---|---|---|---|---|
| 2026-09-11 | PR head `354545b89fef6934d01eb932d868fc3e1260d830` | GitHub Actions `verify-task-tutorial` run `34574579363`, Ubuntu 24.04.5, CPython 3.10.19, CPU | repository clone + pinned install, COCO8 validation subset (4 images, 17 GT boxes) | success — COCO AP@[0.50:0.95] `0.7073101933`, AP50 `0.9570957096`, AP75 `0.6993399340`; empty-detector baseline AP `0.0` |

That run exercised a notebook that cloned this repository and evaluated COCO8 by default; it is evidence for that
carrier and for the OpenMMLab inference path, not for the standalone notebook above.

## Current status

No clean-runtime execution of the standalone notebook has been recorded yet; the run is **pending**. Static validation
(`tools/validate_release_assets.py`), nbformat validation, a `compile()` sweep over every code cell, and the offline
unit suite passed on the tutorial source at the candidate revision, which is necessary but not sufficient. The
registry status remains **Candidate** until a reviewer confirms a recorded run against the notebook blob under review
and an integrator promotes it; promotion is not performed by the builder. Facts a reviewer should weigh:
`stage_missing_files` was exercised only with an injected downloader in the unit suite (the real fetch from
`download.openmmlab.com` into a fresh `weights/swin-t-mask-rcnn-coco/` has not been executed on this carrier);
`verify_snapshot` was executed once over the real local checkpoint on the builder's workstation (OK); the OpenMMLab
loader, `coco_box_ap` (pycocotools) and the single-command pinned install (`--extra-index-url` PyTorch CPU +
`--find-links` OpenMMLab mmcv, in place of the previous notebook's separate `pip`/`mim` steps) have been validated
statically only — wheel availability for every pin was checked against the indexes, resolution has not been run; and
the standalone carrier itself — executing the carried module cells in a runtime that has no repository checkout — has
been validated statically (parity PASS, carrier probe) but never run.
