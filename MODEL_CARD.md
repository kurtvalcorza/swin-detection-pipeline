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
task_inference_status: release-grade
task_inference_surface: spec/task-inference-surface.json
---

# Swin Object Detection Pipeline (org.valcorza.swin-detection) — release-grade pretrained inference; gradient adaptation scaffold

###### Description

This repository has **two deliberately separate capability surfaces**. First, it now ships an implemented pretrained `TASK-INFERENCE` runtime for Swin-T + Mask R-CNN through `dimer_swin_detection.DimerSwinDetector`, the `dimer-swin-detect` CLI, `spec/task-inference-surface.json`, and the release-grade notebook `tutorials/swin_detection_task_inference.ipynb`. That runtime uses the pinned OpenMMLab MMDetection 3.3.0 distribution, verifies the exact checkpoint size and SHA-256 before deserialization, validates image inputs, and returns the DIMER v1 detection surface: class-labelled axis-aligned boxes and uncalibrated class scores. Second, the future composed-worker `GRADIENT-ADAPTATION` pipeline remains a DIMER Pipeline Spec 1.0 `scaffold`. Its validator, finetuner, canonical detection representation, accelerator qualification, composition, and release manifest are still absent. The `lifecycle_status: scaffold` front matter applies to that adaptation composition; it does not negate the separately implemented pretrained inference capability.

#### Intended Use and Limitations

The supported capability today is pretrained inference with the frozen 80-class COCO label space. Users can run the repository-owned API or CLI in the qualified Python 3.10/OpenMMLab environment, or follow the release-grade notebook, to obtain detections from RGB images and to reproduce the small labelled tutorial evaluation. The repository does **not** currently support user-supplied fine-tuning, a DIMER-produced adapted artifact, or a production serving composition. The checkpoint format is code-capable PyTorch serialization; the runtime reduces substitution risk by verifying exact bytes before MMDetection loads the file, but a digest does not establish publisher authenticity. Detection scores are uncalibrated and no deployment threshold is prescribed. The tutorial sample is intentionally small and its AP values are educational sanity evidence, not an upstream benchmark reproduction or a claim of fitness for a deployment domain. DIMER redistribution/hosting of the separate adaptation-lineage Microsoft checkpoint remains blocked while its weight licence status is `unknown`.

###### Primary Intended Uses

The implemented use is **pretrained object-detection inference** on RGB photographs whose content is reasonably close to COCO's everyday-object domain. The model emits detections over the standard 80 COCO classes. Appropriate uses include technical exploration, integration testing, qualitative inspection, pipeline prototyping, and controlled research evaluation where a practitioner understands the model's inherited label space and limitations. The release notebook additionally demonstrates COCO-style AP/AP50/AP75 evaluation on a fixed COCO8 validation subset and optional BYOD inference without pretending that four images constitute a production validation set. A separate intended future use is supervised full-network gradient adaptation on operator-controlled COCO-style annotations, but that use remains **design intent only** until the DIMER detection representation, validator and finetuner worker releases, accelerator qualification, artifact contract and serving composition exist. The pretrained runtime must not be described as satisfying those missing adaptation requirements.

###### Primary Intended Users

The intended current users are machine-learning engineers, imaging analysts, researchers and DIMER integrators who can operate a pinned Python environment, interpret object-detection outputs, and distinguish pretrained inference from task-specific adaptation. They should understand bounding-box coordinates, class labels, score thresholds, non-maximum suppression, COCO AP, and the consequences of applying a COCO-trained model to imagery outside its training distribution. The supported installation contract is the exact bootstrap encoded in the release-grade notebook; `pip install .` alone is not a qualified OpenMMLab installation because MMCV/PyTorch binary wheels are selected from specialized package indexes. The future gradient-adaptation surface is intended for the same class of technical operator, with additional responsibility for annotation quality, train/validation separation, GPU execution, artifact governance and domain-specific evaluation. This repository is not designed as an autonomous decision system or as a zero-context end-user application.

###### Out-of-scope use cases

The v1 inference contract excludes instance-mask output even though Mask R-CNN internally produces masks; only object-detection boxes, class labels and scores are exposed. It does not provide tracking, person re-identification, keypoints, oriented boxes, panoptic segmentation or semantic segmentation. The pretrained runtime is not a fine-tuner and cannot turn a private class taxonomy into a trained DIMER artifact. The current repository does not provide a canonical DIMER COCO representation validator, finetuner worker, production inference service, calibration workflow or deployment acceptance threshold. Images requiring non-RGB bands, specialized medical interpretation, tiny-object aerial detection at resolutions far outside the COCO regime, or safety-critical real-time behavior are outside the qualified scope. Use for autonomous consequential decisions, surveillance/profiling, weapons targeting, unlawful discrimination, or deceptive evidence generation is unacceptable. Upstream checkpoint hosting through DIMER is also out of scope until licensing and redistribution status are authoritatively resolved.

#### Factors

Performance depends on factors that are only partially represented by the small tutorial evidence. The pretrained detector inherits the COCO training distribution: object categories, camera viewpoints, scene composition, object scale, lighting, occlusion and annotation conventions. Changes in image resolution, compression, color characteristics, crop strategy, aspect ratio and object density can materially change detection quality. Box annotation conventions influence measured localization quality, especially AP75. Score distributions are model outputs, not calibrated probabilities, and their practical meaning changes by class and domain. The repository's exact-head tutorial execution proves that the pinned software/model path runs and produces sensible outputs on a known labelled sample; it does not characterize all environmental or demographic factors. Future adaptation will introduce additional factors including dataset sampling, label mapping, random seeds, training duration, GPU kernels, class imbalance and annotation noise. Those remain outside the implemented runtime and must not be inferred from inference-only evidence.

###### Groups

The model is not designed as a human-attribute classifier, but COCO contains many images of people and `person` is one of its major classes. Consequently, deployment involving people can inherit demographic, geographic and contextual skews from the source corpus even though the DIMER v1 output surface contains only object labels and boxes. This repository has not conducted a group-disaggregated fairness audit of the pretrained model. The small COCO8 tutorial subset is not suitable for such an audit and should never be interpreted as one. If an operator uses detections in contexts involving people, groups, neighborhoods, occupations, mobility or proxies for sensitive characteristics, they are responsible for representative labelled evaluation, group-specific miss/false-positive analysis and governance review before any consequential use. The absence of a measured disparity in this repository is not evidence of parity. Group-sensitive use is especially inappropriate where detection outputs could influence enforcement, access to services, eligibility, monitoring or adverse decisions.

###### Instrumentation

The inference runtime accepts local image files and validates basic readability, positive dimensions and a 64,000,000-pixel operational ceiling. MMDetection performs the pinned preprocessing defined by its packaged configuration. The detector's behavior can be affected by camera optics, image compression, resizing, sensor noise, exposure and color rendering. The release notebook uses a fixed labelled COCO-derived sample and converts its annotations into COCO evaluation structures without changing sample membership. For future adaptation, instrumentation also includes the annotation tool and box-drawing convention: clipping, truncation, treatment of occlusion and category mapping all influence both training and evaluation. A proper DIMER detection validator should eventually reject degenerate/out-of-domain boxes and undeclared category ids and record any coordinate normalization. None of those future dataset-validation claims are made by the current pretrained runtime. The present validation boundary is the inference image itself plus the notebook's explicitly controlled tutorial fixtures.

###### Environment

The qualified public inference environment is CPython 3.10 with torch 2.1.2, MMDetection 3.3.0, MMCV 2.1.0, MMEngine 0.10.7, NumPy 1.26.4 and OpenCV 4.10.0.84. CPU inference is supported and is what the clean GitHub-hosted tutorial gate exercises. The API checks the OpenMMLab package versions and fails closed if they drift. Because MMCV and PyTorch depend on platform-specific binary wheels, `pip install .` is not presented as a complete environment bootstrap; the release notebook is the supported reproducible installation path. The runtime downloads the exact OpenMMLab checkpoint, verifies its 191,461,353-byte size and SHA-256 `9d6b7cfaa4aad52ef559611bea454f01d6f1f17c82a1abfac0d71631a193a291`, then permits MMDetection to deserialize it. The future gradient-adaptation environment is not qualified: GPU/VRAM requirements, worker images, precision policy and deterministic behavior remain open blockers.

#### Metrics

For the implemented pretrained inference tutorial, the task-appropriate headline metrics are COCO AP@[0.50:0.95], AP50 and AP75. The exact committed release notebook executed on a four-image COCO8 validation subset and measured AP@[0.50:0.95] `0.7073101933`, AP50 `0.9570957096`, and AP75 `0.6993399340`, with 17 ground-truth boxes. It also records an empty-detector AP baseline of `0.0`. These values are **tutorial-sample measurements** with very high sampling variance; they are not the model's production specification and are not comparable to a full COCO validation run. Separately, upstream projects report full-benchmark performance for their released models; those figures remain upstream claims unless reproduced under the same protocol. For the future gradient-adaptation pipeline, the intended metrics remain COCO AP/AP50/AP75 on an operator-controlled validation split, but no training evaluation has been implemented by this repository.

###### Performance Measures

The repository now measures performance only within the narrow, labelled tutorial path. The release-grade notebook constructs COCO ground truth for its fixed validation images, converts repository API detections to COCO result records, and invokes `pycocotools` COCO evaluation. This demonstrates that the supported API can be evaluated end to end and produces nontrivial results above the deliberately weak empty-detector baseline. It does **not** measure robustness, calibration, per-class AP, small/medium/large object AP, latency distributions, memory ceilings, demographic parity or domain-transfer performance. Those require larger and purpose-built datasets. The `MODEL_SPEC` also carries an upstream-reported full-COCO box AP value for context; the notebook labels it as upstream-reported and not measured locally. Any deployment should replace the tutorial sample with a representative labelled validation set and define acceptance criteria that correspond to the real cost of missed objects, false alarms and localization errors.

###### Decision thresholds

The public API accepts a `score_threshold` output filter and `max_detections`, but the repository does not claim a universal deployment operating point. The underlying MMDetection configuration also includes its own post-processing behavior such as NMS. A detection score is explicitly documented as **uncalibrated**: `0.9` does not mean a 90% probability that the box is correct. The release notebook uses a zero score threshold for metric evaluation so COCO evaluation sees the model output rather than a presentation-oriented cutoff; BYOD display can use a higher convenience threshold. Operators must choose thresholds on representative labelled data and should inspect class-specific precision/recall tradeoffs instead of copying benchmark defaults. No safety, business or regulatory decision threshold is supplied. The future gradient-adaptation pipeline likewise has no acceptance threshold yet because its validator/finetuner and release contract do not exist; that omission is a blocker, not a hidden default.

###### Approaches to uncertainty and variability

The tutorial records one deterministic-enough execution of a frozen inference model on fixed fixtures; it does not estimate statistical uncertainty from repeated sampling. With four evaluation images, the measured AP values have substantial sampling variability even though the software path is reproducible. Detection scores are not calibrated probabilities, and the repository does not export confidence intervals or Bayesian uncertainty. Variability under image corruption, camera shift, geography, object size, crowding and class frequency remains unmeasured. The clean CI run is evidence of **re-executability of the tested path**, not evidence that output values will be invariant across all hardware or future library builds. Version checks, immutable repository/model references and checkpoint digests reduce accidental implementation drift. For future training, uncertainty will also include random initialization, sampling and GPU kernel effects; those concerns belong to the still-blocked gradient-adaptation design and require explicit seeds, repeatability policy and evaluation evidence when that capability is implemented.

#### Ethical considerations and biases

Object detection can be repurposed into applications that affect people even when the model was trained on generic objects. COCO's source imagery includes people, homes, vehicles and public/private scenes, and its class distribution and photographic context are not representative of every population or geography. The pretrained model therefore may encode uneven error rates or contextual associations that this repository has not audited. The runtime can also make confident-looking errors under distribution shift, which creates automation-bias risk when users mistake machine boxes for verified facts. The project mitigates some technical risks through immutable model identity, pre-deserialization digest checks, explicit uncalibrated-score semantics, narrow output scope and release documentation, but those mechanisms do not solve social-context risks. Consequential human decisions, surveillance/profiling and coercive uses require independent governance and are outside intended use. Operators remain responsible for data rights, domain validation, human review, threshold selection and legal/regulatory obligations.

###### Data

The pretrained inference model was trained upstream on COCO 2017, a large collection of everyday photographs with object annotations over 80 classes. The repository does not redistribute that training dataset. Its release notebook fetches the small public COCO8 sample for tutorial evaluation and records the sample archive digest at execution time; that sample exists only to exercise the evaluation path and is not a representative validation corpus. Optional BYOD images remain in the notebook runtime and are not sent to a DIMER inference service. Users must not upload confidential, restricted or personal imagery to hosted notebook environments without authorization. The future gradient-adaptation pipeline is intended to consume operator-supplied COCO-style annotations, but no canonical DIMER detection representation or validator currently exists. The repository therefore makes no claim that arbitrary training datasets are validated, licensed, consented, de-identified or free from leakage. Those governance obligations remain with the operator.

###### Human Life

This repository is not intended for autonomous or materially consequential decisions about health, criminal justice, employment, credit, housing, education, immigration, public benefits, insurance, policing or physical safety. The pretrained runtime is a technical inference capability over general COCO categories, not a certified decision system. Where images include people, detections can still be wrong, uneven across contexts or used as proxies for sensitive information. The tutorial's successful execution and sample AP measurements do not establish safety for any human-impact domain. Any proposed use that could materially affect people would require a separate representative validation study, governance review, explicit human decision authority, calibrated operating thresholds and compliance with applicable law; many surveillance, profiling, discriminatory and coercive uses remain unacceptable regardless of technical performance. The future adaptation pipeline being scaffolded does not relax these restrictions. A more accurate model or custom fine-tuning would not by itself make a prohibited or ungoverned use acceptable.

###### Mitigations

Implemented mitigations for pretrained inference include a repository-owned API rather than notebook-only model code, immutable model metadata, checkpoint size and SHA-256 verification **before** code-capable `.pth` deserialization, pinned/checked OpenMMLab versions, basic image validation, explicit score semantics, a 64-megapixel input ceiling, a narrow boxes/classes/scores output contract, machine-readable provenance and an exact-notebook clean execution gate. The tutorial distinguishes its own sample metrics from upstream benchmark claims and provides a trivial baseline so readers do not see a number without context. The older scaffold controls remain active for gradient adaptation: `scripts/verify_scaffold.py` refuses promotion while worker/representation/qualification blockers remain, and `provenance/open-weights.json` keeps DIMER hosting blocked while adaptation-lineage weight redistribution is `unknown`. These are supply-chain and lifecycle mitigations, not substitutes for domain validation, fairness assessment, threshold calibration, user training, access control, audit logging or legal review.

###### Risks and harms

Key risks include false positives, missed detections, poor localization, class confusion and silent distribution shift; these can waste resources or mislead users who treat detections as verified observations. Uncalibrated scores can encourage unjustified confidence. COCO-derived features can inherit dataset skews, particularly in scenes involving people and geographically specific contexts. The `.pth` trust boundary remains inherently code-capable even though the exact pinned bytes are verified first. A separate licensing risk exists for DIMER redistribution of the future adaptation-lineage Microsoft checkpoint, which remains blocked because the weight licence is not authoritatively established. The existence of a convenient pretrained API can also lower the barrier to repurposing for surveillance, profiling or other harmful applications. The strongest control is therefore a combination of technical integrity, explicit capability boundaries, human review and use-policy governance. Future fine-tuning could amplify any of these harms if operator data or labels are biased, poorly governed or mismatched to deployment conditions.

###### Use cases

Unacceptable uses include surveillance or tracking intended to identify, follow or characterize people or groups; biometric/demographic profiling or social scoring; unlawful discrimination or the use of detected objects as proxies for eligibility in employment, housing, credit, insurance, education, healthcare or public services; autonomous weapons targeting or other coercive use of force; deceptive presentation of machine detections as verified evidence; and deployment in safety-critical or high-impact settings without independent domain validation and accountable human decision-making. DIMER redistribution of upstream checkpoint bytes is also prohibited while the applicable weight-licence determination remains unresolved. Acceptable technical exploration must still respect the rights and licences attached to input imagery and hosted notebook environments. The implemented pretrained runtime should not be confused with a production-approved application, and the future gradient-adaptation capability should not be represented as available until its DIMER representation, workers, qualification, artifact and serving contracts are actually implemented and reviewed.

---

## Model details

| Item | Value |
|---|---|
| Pipeline id | `org.valcorza.swin-detection` |
| Implemented capability | pretrained `TASK-INFERENCE`, `spec/task-inference-surface.json`, package `dimer-swin-detection` 0.1.0 |
| Inference model | OpenMMLab Swin-T + Mask R-CNN, MMDetection 3.3.0; checkpoint 191,461,353 bytes, SHA-256 `9d6b7cfaa4aad52ef559611bea454f01d6f1f17c82a1abfac0d71631a193a291` |
| Inference outputs | COCO class id/name, uncalibrated score, axis-aligned `bbox_xyxy`; instance masks excluded from DIMER v1 |
| Release tutorial | `tutorials/swin_detection_task_inference.ipynb` (`TASK-INFERENCE`, release-grade) |
| Tutorial evidence | 4 COCO8 validation images, 17 GT boxes; AP `0.7073101933`, AP50 `0.9570957096`, AP75 `0.6993399340`; tutorial-only evidence |
| Adaptation lifecycle | `scaffold`, intended `COMPOSED-WORKERS` / `GRADIENT-ADAPTATION` |
| Adaptation lineage checkpoint | Microsoft/SwinTransformer `mask_rcnn_swin_tiny_patch4_window7_1x.pth`, SHA-256 `b67f9d6cd62a4d723c78faec1b49cbf548faa22437264defb00f2f6e54d21b78` |
| Adaptation weight hosting | `redistribution_status: unknown`, `dimer_hosting: BLOCKED` |
| Adaptation blockers | detection representation, validator release, finetuner release, accelerator qualification |

## References

- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. arXiv:2103.14030.
- He, K. et al. *Mask R-CNN.* ICCV 2017. arXiv:1703.06870.
- Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context.* ECCV 2014. arXiv:1405.0312.
- Microsoft Swin Transformer and Swin object-detection repositories.
- OpenMMLab MMDetection 3.3.0 model distribution used by the implemented task-inference runtime.
- `spec/task-inference-surface.json`, `spec/pipeline-surface.json`, `provenance/open-weights.json`, and `tutorials/README.md` in this repository.
