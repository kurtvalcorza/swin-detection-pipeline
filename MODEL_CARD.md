---
license: mit
model_card_spec: "1.0"
pipeline_spec: "1.0"
base_model: SwinTransformer/storage mask_rcnn_swin_tiny_patch4_window7_1x.pth (release v1.0.3, asset 36780486)
base_model_sha256: b67f9d6cd62a4d723c78faec1b49cbf548faa22437264defb00f2f6e54d21b78
base_model_weights_license: unknown — not yet determined from an authoritative upstream statement; DIMER hosting BLOCKED
pipeline_id: org.valcorza.swin-detection
lifecycle_status: scaffold
implementation_topology: COMPOSED-WORKERS
capability_modes:
  - GRADIENT-ADAPTATION
task_profile: core.task.vision.object-detection
---

# Swin Object Detection Pipeline (org.valcorza.swin-detection) — scaffold, no packaged version

###### Description

This repository is a **scaffold**: it freezes the task, architecture lineage, checkpoint provenance and contract blockers for a future DIMER object-detection pipeline, and ships a verifier for those frozen claims. **It contains no runtime.** Nothing here loads a model, validates a dataset, trains, or detects, and no version has been packaged. The card exists so that the intended model and its boundaries are stated before implementation, and so that a reader cannot mistake the scaffold for a runnable pipeline.

The intended model is the official **Swin-T + Mask R-CNN** checkpoint from the Microsoft Swin Transformer project — *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (Liu et al., 2021, arXiv:2103.14030) as the backbone of a Mask R-CNN detector (He et al., 2017, arXiv:1703.06870) — trained on COCO 2017 with the 1× schedule and published as `mask_rcnn_swin_tiny_patch4_window7_1x.pth` in the `SwinTransformer/storage` GitHub release `v1.0.3` (asset 36780486, 191,487,694 bytes, SHA-256 `b67f9d6c…`). A Swin transformer computes self-attention inside shifted local windows to build a multi-scale feature pyramid; Mask R-CNN adds a region-proposal network, a box head that regresses class-labelled bounding boxes, and a mask head. **The v1 contract is object detection only**: boxes and class scores are the output surface; the checkpoint's instance masks are deliberately outside it. The intended adaptation is **full-network gradient fine-tuning** on the operator's own COCO-style annotations.

What this repository adds today is the pinned provenance (`provenance/open-weights.json`), the machine-readable task surface and blockers (`spec/pipeline-surface.json`), the fail-closed verifier (`scripts/verify_scaffold.py`) and a `SMOKE` notebook. The validator and finetuner workers, the `ml-worker` COCO detection representation profile, and any evaluation are **not implemented**. Upstream detection code depends on the MMDetection toolchain, which this project has not adopted; how the detector is executed is an open design decision.

#### Intended Use and Limitations

Everything in this container describes design intent for a pipeline that does not yet exist; no use is currently supported.

###### Primary Intended Uses

**Intended task (not yet implemented):** object detection by supervised fine-tuning. Input would be images with COCO-JSON box annotations (a canonical DIMER detection representation profile does not yet exist on the reviewed contract branch); output would be, per image, a list of class-labelled bounding boxes with confidence scores, with COCO AP@[0.50:0.95], AP50 and AP75 as primary metrics, and a deployable artifact for DIMER serving.

**Application domains envisioned:** counting and localising discrete objects in photographs — vehicles, containers, equipment or defects in inspection imagery, produce or livestock in agricultural photographs, items on shelves. The COCO pretraining gives an 80-class everyday-object prior; the intended pipeline would replace that label space with the operator's.

**Role in a larger system:** the third of three Swin task pipelines for DIMER, reusing the validator/finetuner split, the digest-pinned catalog and the artifact contract established by `swin-classification-pipeline`. Two YOLO26 detectors already exist in DIMER; this pipeline's justification is Swin backbone parity across the family, not a new capability. Until the workers exist, the repository's only role is to hold the frozen specification and refuse to be treated as anything more.

###### Primary Intended Users

The intended users of the eventual pipeline are **machine-learning engineers and imaging analysts operating DIMER deployments**, in an internal enterprise or research setting where the operator controls the data, the GPU and the downstream use of detections.

They are assumed to understand bounding-box annotation and its ambiguity at object edges, the difference between detection and instance segmentation, COCO-style AP and why a single AP number hides per-class and per-size behaviour, the role of score thresholds and non-maximum suppression in turning raw detections into decisions, and that COCO pretraining transfers to consumer photographs far better than to aerial, medical or industrial imagery.

Today the repository has only one user role: **a maintainer or reviewer** checking that the scaffold's claims stay internally consistent (`python scripts/verify_scaffold.py`) and that the checkpoint identity has not drifted. It is not for hobbyist or self-service use, and it currently offers nothing an end user can run.

###### Out-of-scope use cases

Capability boundaries:

- Not for any inference, training or evaluation **today** — the repository has no runtime; the `SMOKE` notebook only runs the verifier.
- Not for instance or semantic segmentation: mask outputs are `OUT_OF_SCOPE_FOR_V1` even though the checkpoint contains a mask head; semantic segmentation is the sibling `swin-segmentation-pipeline`.
- Not for image classification (see `swin-classification-pipeline`), keypoint detection, tracking, or oriented/rotated boxes.

Input boundaries (intended, not enforced yet):

- Not for annotations outside COCO-JSON axis-aligned boxes with a frozen category map; no detection validator exists yet, so nothing validates them.
- Not for imagery whose semantics depend on bands beyond RGB; the upstream checkpoint consumes 3-channel input.
- Not for objects far outside the COCO size regime (very small objects in very large images) without the operator's own re-evaluation of tiling and resolution.

Decision boundaries:

- Not for autonomous decisions affecting people — security screening, crowd counting for enforcement, medical lesion detection — with or without a human in the loop; nothing has been validated.
- Not for any deployment that fixes a score threshold or NMS setting without calibrating it on the operator's own labelled data; none is shipped.
- Not for redistribution or hosting of the upstream checkpoint through DIMER: its `redistribution_status` is `unknown` and `dimer_hosting` is `BLOCKED` in `provenance/open-weights.json`.

#### Factors

No behaviour has been measured by this repository; the subsections record what is known about the upstream model and what the eventual pipeline will have to consider.

###### Groups

The intended pipeline is **not human-centric by design** — it localises operator-defined object classes — but COCO, the upstream training set, is dominated by everyday photographs of people (`person` is its most frequent class), and published audits of COCO document gender, skin-tone and geographic skews in both images and captions. The upstream authors publish no group-level audit of the detector, so the pretrained features are **not group-audited**. No evaluation of any kind has been performed here.

The obligation transfers to the eventual operator: where detections involve people or correlate with people's characteristics, the operator must run a group-disaggregated evaluation — per-group AP and per-group miss rate on a labelled holdout they control — before deployment. The pipeline will not perform that audit; the scaffold records the obligation so it is not forgotten when the runtime arrives.

###### Instrumentation

Training and evaluation data for the eventual pipeline will be **operator-supplied images with box annotations**; the repository will know nothing about the camera, lens, mounting or annotation tool that produced them. Upstream, COCO images are Flickr photographs of mixed provenance annotated by crowd workers.

Instrument characteristics that will reach the model: resolution and the multi-scale resize the detector applies (upstream trains at shorter-side 480–800), colour encoding (RGB), compression artefacts, and — critically for detection — **box tightness conventions** and the annotation tool's handling of occlusion and truncation. Instrument error propagates as localisation noise that depresses AP75 more than AP50, and as distribution shift when camera geometry or mounting changes. The intended validator must at minimum refuse boxes outside the image domain, degenerate (zero-area) boxes and undeclared category ids, and must record any coordinate clipping (Pipeline Spec §21.10). Nothing detects any of this today.

###### Environment

**Operating environment (intended).** Fine-tuning a Swin-T Mask R-CNN requires an NVIDIA GPU; the sibling classification finetuner's fail-closed `cuda:0` policy is intended to carry over. No qualification packet exists for this repository — `ACCELERATOR_QUALIFICATION_PENDING` is an open blocker — so no VRAM, precision or runtime envelope is claimed, and the execution toolchain itself (upstream depends on MMDetection/MMCV, which this project has avoided elsewhere) is undecided. The upstream checkpoint is a PyTorch `.pth` pickle that requires a trusted loader (`torch.load` is code-capable serialization); the repository's own tooling never deserialises it and its notebook says so. The verified environment for the scaffold itself is a Kaggle CPU kernel (Python 3.12.13, Linux 6.12.90) where `verify_scaffold.py` passed.

**Data environment (intended).** The eventual model will assume inference imagery is drawn from the same distribution as the operator's training images — same camera geometry, object scale, lighting and class semantics — and photographs in the RGB domain COCO covers. Degradation under shift is silent: a detector keeps emitting confident boxes on out-of-distribution imagery and misses objects at unfamiliar scales. None of this is measured here.

#### Metrics

No metric is computed by this repository. The subsections state what the eventual pipeline is specified to report and what the upstream authors reported, kept separate.

###### Performance Measures

**Measured by this repository: nothing.** The scaffold has no evaluation code, and its verifier proves only internal consistency of the specification and provenance files.

**Specified for the eventual pipeline** (`spec/pipeline-surface.json → metrics`): **COCO AP@[0.50:0.95]** — mean average precision averaged over IoU thresholds from 0.50 to 0.95, the standard detection summary because it rewards both finding objects and localising them tightly — with **AP50** (lenient localisation: did the detector find the object at all) and **AP75** (strict localisation) reported alongside so a reader can tell recall failures from box-quality failures. Per-class AP and small/medium/large breakdowns are intended when the evaluation contract is written; a single AP hides both.

**Reported upstream, not reproduced here:** the official checkpoint's COCO 2017 val box mAP is 43.7, with mask mAP 39.8 recorded as supplementary because masks are outside the v1 contract (`provenance/open-weights.json → canonicalV1.reportedMetrics`, from the upstream task repository's model table). These are the upstream authors' numbers on their benchmark; this repository has not executed the model and makes no performance claim.

###### Decision thresholds

Detection has **two** decision rules, and the intended pipeline will have to make both explicit (Pipeline Spec §21.10): a **score threshold** below which a box is discarded, and **non-maximum suppression** (an IoU threshold above which overlapping boxes of the same class are merged). Upstream Mask R-CNN inference uses a 0.05 score floor and 0.5 NMS IoU for COCO evaluation, which are benchmark settings, not deployment thresholds. No threshold of either kind is intended to ship as a default decision: raw scores and the NMS configuration used would be recorded in provenance, and choosing the operating point is the deploying operator's task on labelled holdout data, weighing false alarms against missed objects for their use.

No acceptance threshold was set because nothing has been trained. When the finetuner exists, its publication rule (publish whatever the validation metrics are, and record them) is intended to match the classification sibling. All of the above is design intent recorded here so it can be checked against the implementation when it lands.

###### Approaches to uncertainty and variability

**Nothing is estimated by this repository**, so there is no estimation procedure, dispersion or seed policy to report yet. The upstream AP figures quoted above are single numbers from the upstream authors' evaluation protocol; they carry no dispersion and are not this repository's evidence.

For the eventual pipeline the intended design is the classification sibling's: a single validator-frozen validation holdout, explicit seeds for initialisation and data order, `reproducibility: REEXECUTABLE` because GPU kernel selection remains non-deterministic (detection adds sampler randomness in the region-proposal stage), and detection scores declared **uncalibrated** — a box score of 0.9 is not a 90 % chance the box is correct. Anything stronger — repeated runs, confidence intervals, calibration — would be the operator's to add. Until the runtime exists, this section's honest content is that there are no numbers and therefore no uncertainty statement to make about them.

#### Ethical considerations and biases

No external board has reviewed this scaffold and no testing with any population has occurred; nothing below implies otherwise.

###### Data

**Upstream training data.** The canonical checkpoint was trained on **COCO 2017** (Lin et al.), about 118,000 training images of everyday scenes from Flickr with crowd-sourced box and mask annotations for 80 object classes. COCO is dominated by images of people, collected without individual consent from public Flickr uploads under Creative Commons licences of varying terms; published audits document demographic and geographic skews. The upstream authors do not enumerate image provenance beyond the Flickr source; whether the corpus contains personal or sensitive imagery cannot be ruled out — it is **known to contain many identifiable people** — and its licence terms as they bear on derived weights have not yet been assessed by this repository.

**Fine-tuning data** would be operator-supplied and, by the intended policy (`runtimeNetworkFetch: DENY`), would never leave the worker's filesystem.

**What this repository distributes:** JSON provenance and specification files, a verifier, a smoke notebook and this card. It distributes **no weights** and **no sample data**. The checkpoint's identity is recorded by size and SHA-256 (verified once in an isolated Kaggle container, `evidence/open-weights-kaggle.json`) but its bytes are not stored here, and DIMER hosting is **BLOCKED** while `redistribution_status` is `unknown`.

**Operator obligation.** The eventual pipeline will perform no audit of the operator's imagery or annotations for personal data, consent, licensing or confidentiality; fine-tuned artifacts derived from that data inherit its governance.

###### Human Life

The intended pipeline is **not intended** for decisions in health, safety, criminal justice, employment, credit, housing or any other matter central to human life, and **nothing has been validated** for any purpose: this repository has never executed the model.

Object detection is the capability most readily turned toward people — person detection, crowd counting, weapon or contraband detection, vehicle identification — and such uses are foreseeable. They would be admissible only with an independent domain validation study on representative data, a human decision-maker reviewing every consequential detection, the operator's own threshold calibration and group-disaggregated evaluation, and whatever regulatory clearance the domain requires. None of that is provided or implied here, and the scaffold's blockers mean none of it can even begin yet.

###### Mitigations

Only mechanisms that exist in this repository are listed; the intended worker-level controls are named as intent, not as mitigations.

*Supply-chain integrity (implemented).* The canonical checkpoint is pinned to an official GitHub release asset (`SwinTransformer/storage@v1.0.3`, asset 36780486) with recorded size and SHA-256, verified against the downloaded bytes in an isolated Kaggle container and recorded in `evidence/open-weights-kaggle.json`; the reference config is pinned by task-repository commit (`7810b893…`) and git blob id. `scripts/verify_scaffold.py` fails if the evidence digest or size disagrees with the provenance, if any revision is not an immutable 40-hex id, if the instance-mask exclusion is dropped, or if the source-of-record policy is weakened.

*Fail-closed lifecycle (implemented).* The surface declares `lifecycle_status: scaffold` and four blockers; the verifier refuses `candidate`/`release` while blockers exist, refuses any declared components or release, and requires the composition to be recorded as `NOT_EMITTED`. No pipeline manifest or release is emitted (Pipeline Spec §33).

*Hosting gate (implemented).* `redistribution_status: unknown` with `dimer_hosting: BLOCKED`; the verifier refuses `permitted` without a licence and source, and refuses an unblocked hosting flag for `unknown`/`prohibited`.

*Serialization trust boundary (implemented as a refusal).* Nothing in this repository deserialises the `.pth` checkpoint; the smoke notebook records the digest and stops, stating that PyTorch checkpoints are code-capable.

*Intended, not implemented:* COCO annotation validation (box domain, degenerate boxes, category ids), explicit score/NMS configuration recorded in provenance, digest-pinned catalog with typed off-catalog refusal, fail-closed GPU, seeded training, content-addressed artifacts with fresh reload — all specified by analogy to `swin-classification-pipeline` and all absent here.

###### Risks and harms

- **Scaffold mistaken for a pipeline.** Mechanism: a reader or tool treats the passing verifier or the notebook as runtime qualification. Bearer: the operator who plans a deployment on it. Mitigated by the lifecycle field, the verifier's refusals and this card, but the risk is the reason those exist.
- **Unlicensed redistribution.** Mechanism: the checkpoint is mirrored before its weight licence is determined. Bearer: the project and upstream rights-holders. Currently blocked by `redistribution_status: unknown`; realised if someone bypasses the gate.
- **Code-capable checkpoint.** Mechanism: `torch.load` of the `.pth` executes pickled objects. Bearer: whoever runs an eventual loader on a tampered file. The SHA-256 pin mitigates substitution, not the trust boundary itself.
- **Threshold misuse (future).** Mechanism: a benchmark score floor or NMS setting is shipped or copied as if it were a calibrated operating point. Bearer: whoever acts on the detections — false alarms waste effort, missed objects cause harm proportional to the use. Likely wherever detectors are deployed without calibration.
- **Silent distribution shift (future).** Mechanism: confident boxes on imagery unlike the training set, misses at unfamiliar object scales. Bearer: whoever acts on the detections and any people depicted. Likely under normal use over time once a runtime exists.
- **Inherited pretraining bias (future).** Mechanism: COCO's documented demographic and geographic skews encoded in the backbone and proposal stage. Bearer: under-represented data subjects, through higher miss rates. Unmeasured.
- **Repurposing toward people (future).** Mechanism: a fine-tuned detector for objects is retargeted to detect people or attributes with the same tooling. Bearer: data subjects. Use-context risk that the prohibited-uses list addresses but cannot technically prevent.
- **Automation bias (future).** Mechanism: reviewers accept machine detections without inspection. Bearer: data subjects and third parties.

###### Use cases

Uses the developers consider unacceptable for the eventual pipeline, even where it would work:

1. **Surveillance, biometric or demographic profiling, and social scoring** — detecting, counting or tracking people or their belongings to identify, follow or characterise individuals or groups, including crowd analytics for enforcement.
2. **Unlawful discrimination** — using detection-derived measures (vehicles owned, goods present, occupancy) as proxies for eligibility in credit, insurance, housing, employment, education or healthcare access.
3. **Deceptive, manipulative or predatory applications** — presenting uncalibrated detections as verified facts to people affected by them, targeting individuals from detected attributes, or fabricating or altering imagery evidence.
4. **Weapons and lethal or coercive systems** — target acquisition or any detection feeding autonomous or semi-autonomous use of force.
5. **Uses prohibited by upstream terms** — the weight licence is not yet determined and COCO's image licences as they bear on derived weights have not been assessed; until both are recorded, any redistribution of the checkpoint through DIMER is prohibited by this repository's hosting gate, and any deployment must honour the operator's data licences and DIMER platform terms.

---

## Model details

| Item | Value |
|---|---|
| Pipeline id | `org.valcorza.swin-detection` — **no packaged version**; `spec/pipeline-surface.json` status `BLOCKED_PENDING_REPRESENTATION_AND_WORKERS` |
| Lifecycle / topology / mode | `scaffold` / intended `COMPOSED-WORKERS` / intended `GRADIENT-ADAPTATION` (`dimerPipelineSpec`) |
| Task profile | `core.task.vision.object-detection` — present on the reviewed `ml-worker` branch `build/dimer-v1-freeze` |
| Dataset representation | canonical encoding COCO JSON; no representation profile on the reviewed branch |
| Canonical checkpoint | `mask_rcnn_swin_tiny_patch4_window7_1x.pth`, `SwinTransformer/storage` release `v1.0.3`, asset 36780486, 191,487,694 bytes, SHA-256 `b67f9d6cd62a4d723c78faec1b49cbf548faa22437264defb00f2f6e54d21b78` (verified: `evidence/open-weights-kaggle.json`) |
| Reference implementation | `SwinTransformer/Swin-Transformer-Object-Detection` @ `7810b893902326d88068037477c848b551e2bd4e`, config `configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py` (blob `dd42cba7…`) |
| Output surface (v1) | class-labelled bounding boxes with scores; instance masks `OUT_OF_SCOPE_FOR_V1` |
| Weight licence / hosting | `redistribution_status: unknown`, `dimer_hosting: BLOCKED` (`provenance/open-weights.json → canonicalV1.weightLicensing`) |
| Upstream reported metrics (not reproduced) | COCO 2017 val box mAP 43.7; mask mAP 39.8 (supplementary, outside v1) |
| Blockers | `CONTRACT_COCO_DETECTION_REPRESENTATION_MISSING`, `VALIDATOR_WORKER_RELEASE_MISSING`, `FINETUNER_WORKER_RELEASE_MISSING`, `ACCELERATOR_QUALIFICATION_PENDING` |
| Verifier | `python scripts/verify_scaffold.py` — internal consistency and fail-closed lifecycle only; not runtime evidence |

## References

- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. arXiv:2103.14030.
- He, K. et al. *Mask R-CNN.* ICCV 2017. arXiv:1703.06870.
- Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014. arXiv:1405.0312.
- Microsoft Swin Transformer: https://github.com/microsoft/Swin-Transformer · Object detection task repository: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
- Sibling pipelines: `swin-classification-pipeline` (candidate), `swin-segmentation-pipeline` (scaffold)
- Smoke notebook: `tutorials/swin_detection_scaffold_smoke_colab.ipynb` (`SMOKE`, engineering-only)
