# MUMU Challenge 2026

This repository provides the evaluation code and complete baseline results for
four Florence-2 checkpoints and `LiquidAI/LFM2.5-VL-450M` on a balanced
6,000-image, three-task source-data evaluation split.

> These are source-data proxy results, not official MUMU Challenge scores. The
> unified challenge taxonomy, novel-category split, and efficiency coefficients
> were not available when the evaluation was run.

## Challenge Overview

**Mobile Unified Multimodal Understanding Challenge**<br>
**Theme:** Efficient Unified Multimodal Architectures for Mobile-Oriented
Visual Understanding

Official website: [lsvos.github.io/mumu](https://lsvos.github.io/mumu)

MUMU benchmarks efficient unified vision-language architectures for
mobile-oriented visual understanding. Participants must use one model to solve
semantic tagging, open-vocabulary object detection, and image captioning under
strict parameter and memory budgets. Evaluation uses PC-side compute statistics;
on-device execution is not required.

The challenge aims to:

- promote unified vision-language modeling;
- encourage efficient multi-task representation learning;
- benchmark performance-versus-complexity trade-offs;
- advance deployable multimodal perception for resource-constrained devices.

### Competition Tasks

| Task | Description | Official metric |
| --- | --- | --- |
| **A: Multi-Concept Image Tagging** | Predict image-quality attributes, scene categories, and event concepts. | Macro-F1 (`A`) |
| **B: Open-Vocabulary Object Detection** | Return boxes, confidence scores, and text labels, including generalization to unseen categories. | Novel-aware mAP@[0.5:0.95] (`B`) |
| **C: Efficient Image Captioning** | Generate one concise English sentence describing the image. | `0.4 CIDEr + 0.3 SPICE + 0.3 CLIPScore` (`C`) |

### Official Training Data

| Task area | Dataset(s) | Purpose |
| --- | --- | --- |
| Quality | KonIQ-10k / SPAQ / LIVE Challenge | Image-quality attributes |
| Scene | Places365 | Scene recognition |
| Event | NUS-WIDE | Event semantics |
| Detection | Objects365 / LVIS / Visual Genome | Detection and open-vocabulary learning |
| Captioning | COCO Captions / Flickr30k / Conceptual Captions | Caption generation |

### Official Data Splits

| Split | Approximate size |
| --- | ---: |
| Training | 1.2 million images |
| Validation | 1,000 images |
| Public test | 5,000 images |
| Private final test | 5,000 images |

Novel detection categories remain hidden for the private evaluation. The
6,000-image source-data split in this repository follows the validation and
public-test sizes and balances the three tasks, but it is an independently
prepared evaluation set rather than an official challenge release. Its exact
11-dataset composition is documented in [docs/DATASET.md](docs/DATASET.md).

### Rules and Scoring

- One unified model is required; Adapter and LoRA modules are allowed inside
  the unified architecture.
- Independent task-specific models are not allowed.
- Total parameters must be no more than `0.5B`.
- Peak PC-side inference memory must be no more than `8 GB`.
- Final score is `(0.3A + 0.4B + 0.3C) * E`.
- Efficiency is `E = 1 / (1 + alpha * PeakMemory + beta * Params)`.

The challenge page defines `alpha` and `beta` symbolically but does not provide
their numeric values. Rankings use final score first, followed by lower peak
memory, lower parameter count, and higher detection performance as tie-breakers.

## Resources

| Resource | Link |
| --- | --- |
| Evaluation data (1,000 validation + 5,000 public test) | [JinyuLiu/MUMU-Eval-6000](https://huggingface.co/datasets/JinyuLiu/MUMU-Eval-6000) |
| China mirror | [hf-mirror.com/datasets/JinyuLiu/MUMU-Eval-6000](https://hf-mirror.com/datasets/JinyuLiu/MUMU-Eval-6000) |
| Model weights | [docs/MODELS.md](docs/MODELS.md) |
| Full metrics and interpretation | [docs/RESULTS.md](docs/RESULTS.md) |
| Reproduction commands | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| Dataset construction and sources | [docs/DATASET.md](docs/DATASET.md) |

The GitHub repository intentionally contains no images, dataset annotations, or
model weights. Public-test ground truth is not released. Complete prediction
files are stored as compressed JSONL under `results/predictions/`; full metric
reports and run summaries are under `results/metrics/` and `results/summaries/`.

## Main Results

Raw score is `0.3 * A + 0.4 * B + 0.3 * C`, before the unspecified MUMU
efficiency factor. Task B uses label-aware mAP@[0.5:0.95].

| Split | Model | Params | Eligibility | A&nbsp;Macro&#8209;F1 | B&nbsp;mAP | C&nbsp;composite | Raw&nbsp;score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Validation | Florence&#8209;2&#8209;base | 231.4M | `≤0.5B` | 0.1947 | 0.0314 | 0.7138 | 0.2851 |
| Validation | Florence&#8209;2&#8209;base&#8209;ft | 231.4M | `≤0.5B` | 0.1589 | 0.0198 | 0.7031 | 0.2665 |
| Validation | LFM2.5&#8209;VL&#8209;450M | 448.7M | `≤0.5B` | **0.4459** | 0.0037 | 0.5285 | 0.2938 |
| Validation | Florence&#8209;2&#8209;large | 776.5M | ![>0.5B](https://img.shields.io/badge/%3E0.5B-over_limit-lightgrey) | 0.1998 | 0.0357 | 0.7436 | **0.2973** |
| Validation | Florence&#8209;2&#8209;large&#8209;ft | 770.2M | ![>0.5B](https://img.shields.io/badge/%3E0.5B-over_limit-lightgrey) | 0.1592 | 0.0189 | **0.7555** | 0.2820 |
| Public&nbsp;test | Florence&#8209;2&#8209;base | 231.4M | `≤0.5B` | 0.1555 | 0.0111 | 0.6657 | 0.2508 |
| Public&nbsp;test | Florence&#8209;2&#8209;base&#8209;ft | 231.4M | `≤0.5B` | 0.1195 | 0.0061 | 0.6489 | 0.2329 |
| Public&nbsp;test | LFM2.5&#8209;VL&#8209;450M | 448.7M | `≤0.5B` | **0.3795** | 0.0016 | 0.5232 | **0.2715** |
| Public&nbsp;test | Florence&#8209;2&#8209;large | 776.5M | ![>0.5B](https://img.shields.io/badge/%3E0.5B-over_limit-lightgrey) | 0.1672 | **0.0138** | 0.6751 | 0.2582 |
| Public&nbsp;test | Florence&#8209;2&#8209;large&#8209;ft | 770.2M | ![>0.5B](https://img.shields.io/badge/%3E0.5B-over_limit-lightgrey) | 0.1494 | 0.0063 | **0.6840** | 0.2526 |

Florence-2-large and large-ft exceed the challenge's 0.5B parameter limit and
are comparison baselines only. LFM2.5 has the best public-test raw score and is
within the limit, but it has 1.94 times the parameters of Florence-2-base; the
unknown efficiency penalty may change their official ranking.

## Quick Start

```bash
git clone https://github.com/JinyuLiuu/MUMU.git
cd MUMU

HF_ENDPOINT=https://hf-mirror.com \
  hf download JinyuLiu/MUMU-Eval-6000 \
  --repo-type dataset --local-dir ./mumu_eval_v1

HF_ENDPOINT=https://hf-mirror.com \
  hf download microsoft/Florence-2-base \
  --revision 5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac \
  --local-dir ./models/Florence-2-base

python -m venv .venv-florence
source .venv-florence/bin/activate
pip install -r requirements/florence2.txt

python scripts/run_florence2_eval.py \
  --eval-root ./mumu_eval_v1 \
  --model ./models/Florence-2-base \
  --output ./predictions/florence2_base.jsonl
```

The downloaded public dataset supports inference on all 6,000 images and local
evaluation on validation. Recomputing the recorded public-test metrics requires
the withheld organizer-only ground truth.

## Official Submission Format

The challenge requests a ZIP archive containing model weights, configuration,
an inference entry point, dependencies, and documentation:

```text
submission.zip
model/
  weights.pt
  config.yaml
inference.py
requirements.txt
README.md
```

The unified predictor must initialize from a weight path and return tags,
detections, and one caption for every input image.

## Schedule and Awards

| Event | Date |
| --- | --- |
| Validation data opens | 8 August 2026 |
| Final submission deadline | 15 August 2026 |
| Winner announcement | 20 August 2026 |

Prize distribution:

- 1st Place Overall: 6,000 CNY;
- 2nd Place Overall: 3,000 CNY;
- 3rd Place Overall: 1,000 CNY;
- additional Best Efficiency, Best Unified Architecture, and Best Student Team
  awards.

Sponsor: [Transsion](https://www.transsion.com/)

Challenge contacts:
[henghui.ding@gmail.com](mailto:henghui.ding@gmail.com) and
[changliu73@outlook.com](mailto:changliu73@outlook.com).

## Repository Layout

```text
configs/             exact label vocabulary used by LFM2.5
docs/                data, model, result, and reproduction documentation
requirements/        separate Florence-2, LFM2.5, and evaluator environments
results/             aggregate tables, full metrics, summaries, predictions
scripts/             dataset builder, inference, parsing, and evaluation code
information.md       original MUMU challenge and experiment notes
```

The Florence-2 inference path follows the official sample notebook: one image
per call, FP32, three beams, and native `post_process_generation`. No NMS,
deduplication, box clipping, or output filtering is applied. LFM2.5 uses its
official Transformers API and normalized JSON grounding format.
