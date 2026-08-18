# MUMU Challenge 2026

**Mobile Unified Multimodal Understanding Challenge**
**Theme:** Efficient Unified Multimodal Architectures for Mobile-Oriented Visual Understanding

Official website: <https://lsvos.github.io/mumu>

## Challenge Overview

The Mobile Unified Multimodal Understanding (MUMU) Challenge 2026 is a benchmark competition for efficient unified multimodal architectures targeting mobile-oriented visual understanding.

Unlike task-specific benchmarks, participants must design one unified model that simultaneously performs semantic tagging, open-vocabulary object detection, and image caption generation under constrained computational budgets. The challenge emphasizes algorithmic efficiency, architectural unification, and deployability in resource-constrained environments. Evaluation uses PC-side compute statistics; on-device execution is not required.

## Objectives

- Promote unified vision-language modeling.
- Encourage efficient multi-task representation learning.
- Benchmark trade-offs between performance and complexity.
- Advance mobile-friendly multimodal perception systems.

## Competition Tasks

### Task A: Multi-Concept Image Tagging

Predict semantic concept labels covering image quality attributes, scene categories, and event categories.

- **Image quality attributes:** blur, noise, exposure, compression artifacts.
- **Scene categories:** indoor/outdoor, office, kitchen, street, stadium, and similar categories.
- **Event categories:** concert, wedding, sports event, protest, festival, and similar categories.
- **Metric:** Macro-F1.

### Task B: Open-Vocabulary Object Detection

Detect objects from a provided open vocabulary and generalize to unseen categories. Each detection must provide:

- Bounding-box coordinates.
- A confidence score.
- A text label.

**Metric:** Novel-aware mAP@[0.5:0.95].

### Task C: Efficient Image Captioning

Generate a concise natural-language description for each input image.

- English only.
- One sentence per image.
- **Metric:** `0.4 CIDEr + 0.3 SPICE + 0.3 CLIPScore`.

## Official Datasets

| Task area | Dataset(s) | Purpose | Usage |
| --- | --- | --- | --- |
| Quality | KonIQ-10k / SPAQ / LIVE Challenge | Image quality attributes | Train |
| Scene | Places365 | Scene recognition | Train |
| Event | NUS-WIDE | Event semantics | Train |
| Detection | Objects365 / LVIS / Visual Genome | Detection and open-vocabulary learning | Train |
| Captioning | COCO Captions / Flickr30k / Conceptual Captions | Caption generation | Train |

### Data Splits

- Training set: approximately 1.2 million images.
- Validation set: 1,000 images.
- Public test set: 5,000 images.
- Private final test set: 5,000 images.
- Novel detection categories are hidden during private evaluation.

### Proposed Validation and Public Test Composition

The following balanced allocation is used to construct the validation and public test sets. Counts are image counts; the public test annotations remain hidden for evaluation.

| Task | Dataset | Validation | Public test |
| --- | --- | ---: | ---: |
| A: Multi-Concept Image Tagging | KonIQ-10k | 67 | 334 |
| A: Multi-Concept Image Tagging | SPAQ | 67 | 334 |
| A: Multi-Concept Image Tagging | LIVE Challenge | 67 | 333 |
| A: Multi-Concept Image Tagging | Places365 | 67 | 333 |
| A: Multi-Concept Image Tagging | NUS-WIDE | 66 | 333 |
| **Task A subtotal** |  | **334** | **1,667** |
| B: Open-Vocabulary Object Detection | Objects365 | 111 | 556 |
| B: Open-Vocabulary Object Detection | LVIS | 111 | 556 |
| B: Open-Vocabulary Object Detection | Visual Genome | 111 | 555 |
| **Task B subtotal** |  | **333** | **1,667** |
| C: Efficient Image Captioning | COCO Captions | 111 | 556 |
| C: Efficient Image Captioning | Flickr30k | 111 | 555 |
| C: Efficient Image Captioning | Conceptual Captions | 111 | 555 |
| **Task C subtotal** |  | **333** | **1,666** |
| **Overall total** |  | **1,000** | **5,000** |

This is a proposed sampling plan rather than a claim about the native splits of the source datasets. Images must be deduplicated against training data, labels must be mapped to the challenge taxonomy, and public test ground truth must not be released.

### Prepared Evaluation Data

The proposed split has been materialized under `/home/volume_shared/share_datasets/data_nvme/mumu_eval_v1` with seed `20260817`.

- Validation: 1,000 images.
- Public test: 5,000 images.
- Integrity: 6,000 unique SHA-256 hashes, no missing images, and no duplicate IDs.
- Public manifests exclude ground truth; internal annotations retain it for local evaluation.

Public distribution and reproducibility resources:

- Evaluation images and public manifests: <https://huggingface.co/datasets/JinyuLiu/MUMU-Eval-6000>
- China mirror: <https://hf-mirror.com/datasets/JinyuLiu/MUMU-Eval-6000>
- Evaluation code and complete results: <https://github.com/JinyuLiuu/MUMU>
- Public-test ground truth and all model weights are intentionally excluded from the GitHub repository.
- Exact model revisions and download links are recorded in `docs/MODELS.md`; complete metrics are recorded in `docs/RESULTS.md`.

### Florence-2 Official-Style Source-Data Baselines

All four official Microsoft checkpoints were evaluated on the same 1,000-image validation set and 5,000-image public test set. Inference follows the official Florence-2 sample notebook: one image per call, FP32 model loading, `max_new_tokens=1024`, `early_stopping=False`, `do_sample=False`, `num_beams=3`, `skip_special_tokens=False`, and direct `post_process_generation` parsing. No NMS, box deduplication, box clipping, or custom prediction filtering is applied to the saved inference output.

#### Validation Results

| Model | Task A Macro-F1 | Task B label-aware mAP | Task B class-agnostic mAP | CIDEr | SPICE | CLIPScore | Task C composite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Florence-2-base | 0.1947 | 0.0314 | 0.0492 | 1.0491 | 0.1951 | 0.7854 | 0.7138 |
| Florence-2-base-ft | 0.1589 | 0.0198 | 0.0389 | 1.0431 | 0.1931 | 0.7599 | 0.7031 |
| Florence-2-large | 0.1998 | 0.0357 | 0.0553 | 1.1149 | 0.2023 | 0.7900 | 0.7436 |
| Florence-2-large-ft | 0.1592 | 0.0189 | 0.0398 | 1.1522 | 0.2047 | 0.7772 | 0.7555 |

#### Public Test Results

| Model | Task A Macro-F1 | Task B label-aware mAP | Task B class-agnostic mAP | CIDEr | SPICE | CLIPScore | Task C composite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Florence-2-base | 0.1555 | 0.0111 | 0.0488 | 0.9406 | 0.1856 | 0.7793 | 0.6657 |
| Florence-2-base-ft | 0.1195 | 0.0061 | 0.0396 | 0.9178 | 0.1824 | 0.7568 | 0.6489 |
| Florence-2-large | 0.1672 | 0.0138 | 0.0541 | 0.9563 | 0.1912 | 0.7839 | 0.6751 |
| Florence-2-large-ft | 0.1494 | 0.0063 | 0.0409 | 0.9865 | 0.1944 | 0.7703 | 0.6840 |

#### Runtime And Integrity

| Model | Parameters | Inference time | Peak allocated GPU memory | Rows | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Florence-2-base | 231,414,016 | 5,053.1 s | 1.18 GiB | 6,000 | 0 |
| Florence-2-base-ft | 231,414,016 | 3,392.3 s | 1.19 GiB | 6,000 | 0 |
| Florence-2-large | 776,465,408 | 5,733.2 s | 3.56 GiB | 6,000 | 0 |
| Florence-2-large-ft | 770,173,952 | 4,080.9 s | 3.53 GiB | 6,000 | 0 |

Every result contains exactly 6,000 unique IDs with task counts A/B/C = 2,001/2,000/1,999. There are no missing predictions, parser failures, padding artifacts, or incomplete generations. Four zero-width boxes produced by the official parser are retained in the raw predictions: one from base-ft, two from base, one from large-ft, and none from large. The maximum raw detections on one image are 183 for base-ft, 169 for base, 128 for large-ft, and 95 for large. Visual review shows that these are generally dense scenes such as shelves of dishes, books, or piles of oranges, but some outputs still substantially over-segment or hallucinate repeated instances. They are retained because the official sample inference does not apply NMS.

The output is structurally sound but is not equally well matched to all challenge tasks. Task A gets zero lexical F1 on the three image-quality datasets because detailed captions rarely state the required quality bins; its score is only a tagging proxy. Task B has plausible labels and in-bounds parsed coordinates overall, but repeated-instance outliers and the lack of confidence scores limit mAP. Task C is the strongest direct match: all predictions are non-empty, and only one of 1,999 captions per model is detected as containing more than one sentence.

Predictions, run summaries, and metric reports are stored under `/home/volume_shared/share_datasets/data_nvme/mumu_eval_v1/predictions` using the stems `florence2_base_official`, `florence2_base_ft_official`, `florence2_large_official`, and `florence2_large_ft_official`.

These are source-data proxy metrics, not official challenge scores. Task A uses strict lexical extraction from detailed captions because Florence-2 has no native tagging head. Task B uses generation order as a deterministic confidence surrogate because `<OD>` does not emit confidence scores; the evaluator uses a standard 100-detection maximum and ignores degenerate boxes only while computing IoU, without modifying the saved predictions. Novel categories are not defined. The final MUMU score cannot be computed until the unified taxonomy, novel split, `alpha`, and `beta` are specified. The two large checkpoints also exceed the challenge's 0.5B parameter limit and are comparison baselines only.

### 2026 Small-VLM Baseline: LFM2.5-VL-450M

`LiquidAI/LFM2.5-VL-450M` was released on Hugging Face on 8 April 2026. It is a 448,718,848-parameter unified vision-language model and remains below the challenge's 0.5B limit. Unlike the earlier LFM2-VL-450M and SmolVLM2 candidates, its official model card explicitly supports normalized JSON bounding-box prediction and reports a RefCOCO-M score of 81.28. It was therefore selected as the recent comparison model.

Inference uses the model card's Transformers API with Transformers 5.1.0, bfloat16, one image per call, greedy decoding, and `max_new_tokens=64`. Task A receives the source-data label vocabulary, Task B requests the model card's normalized `[x1, y1, x2, y2]` JSON format, and Task C requests an 8-to-12-word COCO-style caption. No NMS, box deduplication, confidence calibration, or custom prediction filtering is applied. Generation order remains the deterministic confidence surrogate for Task B.

#### LFM2.5-VL-450M Results

| Split | Task A Macro-F1 | Task B label-aware mAP | Task B class-agnostic mAP | CIDEr | SPICE | CLIPScore | Task C composite | Raw weighted score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.4459 | 0.0037 | 0.0043 | 0.6099 | 0.1537 | 0.7949 | 0.5285 | 0.2938 |
| Public test | 0.3795 | 0.0016 | 0.0073 | 0.5912 | 0.1596 | 0.7959 | 0.5232 | 0.2715 |

The raw weighted score is `0.3A + 0.4B + 0.3C`, before the unspecified efficiency factor. Against the parameter-compliant Florence-2-base, LFM2.5-VL-450M improves the raw score from 0.2851 to 0.2938 on validation and from 0.2508 to 0.2715 on public test. It also exceeds every Florence-2 checkpoint on public test, including the over-limit Florence-2-large result of 0.2582. On validation it remains slightly below Florence-2-large's 0.2973.

The gain comes from Task A, where LFM2.5-VL-450M more than doubles Florence-2-base's lexical tagging proxy. It does not beat Florence-2 on Task B or Task C individually. Task B is the main weakness: 1,796 of 2,000 images contain parsed detections, with 1,805 total boxes and at most two boxes per image, so recall is low in dense LVIS and Visual Genome scenes. All parsed boxes are finite, non-degenerate, and inside their image bounds. Task C has no empty or multi-sentence outputs; captions average 11 words. The COCO-style prompt raises Task C from 0.4779/0.4805 to 0.5285/0.5232 on validation/public test compared with the first, more verbose prompt.

The selected predictions and metric report are stored as `lfm2_5_vl_450m_coco_prompt.jsonl` and `lfm2_5_vl_450m_coco_prompt.metrics.json` under `/home/volume_shared/share_datasets/data_nvme/mumu_eval_v1/predictions`. The original verbose-caption run is retained as `lfm2_5_vl_450m.jsonl` for auditability. Both selected split counts are complete: 1,000 validation rows, 5,000 public-test rows, 6,000 unique IDs, and zero inference failures.

This establishes a better unpenalized three-task source-data score than Florence-2-base, not a guaranteed official challenge win. LFM2.5-VL-450M has about 1.94 times as many parameters as Florence-2-base, so the unknown `alpha` and `beta` efficiency penalty can reverse the ranking. The comparison also uses an explicit Task A vocabulary, whereas the Florence-2 proxy extracts labels from captions; an official unified taxonomy and inference protocol are required for a definitive ranking.


## Rules & Evaluation

### Model Constraints

- Single unified model only; Adapter / LoRA is allowed within the unified architecture.
- No task-specific independent models. All task heads must be integrated within the unified model.
- Total parameter count: `<= 0.5B`.
- Peak inference memory: `<= 8 GB` on the PC side.

### Evaluation Metrics

- Task A: Macro-F1 (`A`).
- Task B: Novel-aware mAP@[0.5:0.95] (`B`).
- Task C: `0.4 CIDEr + 0.3 SPICE + 0.3 CLIPScore` (`C`).
- Final score: `(0.3A + 0.4B + 0.3C) * E`.
- Efficiency factor: `E = 1 / (1 + alpha * PeakMemory + beta * Params)`.

The website defines `alpha` and `beta` symbolically but does not specify their numeric values on the challenge page.

### Ranking Rules

1. Primary ranking: final score.
2. Tie-break 1: lower peak inference memory.
3. Tie-break 2: lower parameter count.
4. Tie-break 3: higher detection performance.

## Submission Format

Submit a ZIP archive containing model weights, configuration, an inference entrypoint, dependencies, and documentation:

```text
submission.zip
model/
  weights.pt
  config.yaml
inference.py
requirements.txt
README.md
```

A unified predictor API is required. The predictor must initialize from a weight path and return tags, detections, and one caption for each input image.

## Important Dates

| Event | Date |
| --- | --- |
| Validation data opens | 8 August 2026 |
| Final submission deadline | 15 August 2026 |
| Winner announcement | 20 August 2026 |

## Awards & Prizes

### Prize Distribution

- 1st Place Overall: 6,000 CNY.
- 2nd Place Overall: 3,000 CNY.
- 3rd Place Overall: 1,000 CNY.

### Additional Awards

- Best Efficiency Award.
- Best Unified Architecture Award.
- Best Student Team Award.

## Sponsor

[Transsion](https://www.transsion.com/)

## Contact

- [henghui.ding@gmail.com](mailto:henghui.ding@gmail.com)
- [changliu73@outlook.com](mailto:changliu73@outlook.com)

---

Copyright MUMU. All Rights Reserved.
