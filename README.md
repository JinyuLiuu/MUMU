# MUMU Florence-2 / LFM2.5 Evaluation

This repository provides the evaluation code and complete baseline results for
four Florence-2 checkpoints and `LiquidAI/LFM2.5-VL-450M` on a balanced
6,000-image, three-task source-data evaluation split.

> These are source-data proxy results, not official MUMU Challenge scores. The
> unified challenge taxonomy, novel-category split, and efficiency coefficients
> were not available when the evaluation was run.

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

| Split | Model | Params | A Macro-F1 | B mAP | C composite | Raw score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation | Florence-2-base | 231.4M | 0.1947 | 0.0314 | 0.7138 | 0.2851 |
| Validation | Florence-2-base-ft | 231.4M | 0.1589 | 0.0198 | 0.7031 | 0.2665 |
| Validation | Florence-2-large | 776.5M | 0.1998 | 0.0357 | 0.7436 | **0.2973** |
| Validation | Florence-2-large-ft | 770.2M | 0.1592 | 0.0189 | **0.7555** | 0.2820 |
| Validation | LFM2.5-VL-450M | 448.7M | **0.4459** | 0.0037 | 0.5285 | 0.2938 |
| Public test | Florence-2-base | 231.4M | 0.1555 | 0.0111 | 0.6657 | 0.2508 |
| Public test | Florence-2-base-ft | 231.4M | 0.1195 | 0.0061 | 0.6489 | 0.2329 |
| Public test | Florence-2-large | 776.5M | 0.1672 | **0.0138** | 0.6751 | 0.2582 |
| Public test | Florence-2-large-ft | 770.2M | 0.1494 | 0.0063 | **0.6840** | 0.2526 |
| Public test | LFM2.5-VL-450M | 448.7M | **0.3795** | 0.0016 | 0.5232 | **0.2715** |

Florence-2-large and large-ft exceed the challenge's 0.5B parameter limit and
are comparison baselines only. LFM2.5 has the best public-test raw score and is
within the limit, but it has 1.94 times the parameters of Florence-2-base; the
unknown efficiency penalty may change their official ranking.

## Quick Start

```bash
git clone <this-repository-url>
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
