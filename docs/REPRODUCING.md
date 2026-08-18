# Reproducing the Evaluation

## 1. Download Data and Weights

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf download JinyuLiu/MUMU-Eval-6000 \
  --repo-type dataset --local-dir ./mumu_eval_v1

hf download microsoft/Florence-2-base \
  --revision 5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac \
  --local-dir ./models/Florence-2-base

hf download LiquidAI/LFM2.5-VL-450M \
  --revision fc6221ca597f3315e4f82fc2df606783267b34ba \
  --local-dir ./models/LFM2.5-VL-450M

hf download openai/clip-vit-base-patch32 \
  --local-dir ./models/clip-vit-base-patch32
```

The other pinned Florence-2 revisions are listed in `docs/MODELS.md`.

## 2. Florence-2 Inference

The recorded environment was Python 3.11.7, PyTorch 2.5.1+cu124,
Transformers 4.41.2, and an NVIDIA A800. One image is processed per call in
FP32 with `num_beams=3`, `max_new_tokens=1024`, `early_stopping=False`, and
`do_sample=False`.

```bash
python -m venv .venv-florence
source .venv-florence/bin/activate
pip install -r requirements/florence2.txt

python scripts/run_florence2_eval.py \
  --eval-root ./mumu_eval_v1 \
  --model ./models/Florence-2-base \
  --output ./predictions/florence2_base_official.jsonl \
  --dtype float32 --num-beams 3
```

Repeat with the other three checkpoints. Existing complete IDs are skipped, so
an interrupted run can be resumed with the same command.

## 3. LFM2.5 Inference

The recorded environment was Python 3.11.7, PyTorch 2.5.1+cu124,
Transformers 5.1.0, BF16, greedy decoding, and `max_new_tokens=64`.

```bash
deactivate
python -m venv .venv-lfm25
source .venv-lfm25/bin/activate
pip install -r requirements/lfm25.txt

python scripts/run_small_vlm_eval.py \
  --eval-root ./mumu_eval_v1 \
  --model ./models/LFM2.5-VL-450M \
  --output ./predictions/lfm2_5_vl_450m.jsonl \
  --label-vocabulary ./configs/task_a_label_vocabulary.json \
  --dtype bfloat16 --max-new-tokens 64
```

The script's current Task C prompt is the selected 8–12 word COCO-style prompt.
Task B requests the checkpoint's normalized JSON box format. Parsed predictions
are saved with raw model text; no NMS, deduplication, confidence calibration, or
custom prediction filtering is applied.

## 4. Metrics

Use a separate evaluation environment because the inference checkpoints require
different Transformers versions. SPICE requires Java; Java 11 was used for the
recorded results.

```bash
deactivate
python -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements/evaluation.txt

python scripts/evaluate_florence2.py \
  --eval-root ./mumu_eval_v1 \
  --predictions ./predictions/florence2_base_official.jsonl \
  --output ./predictions/florence2_base_official.metrics.json \
  --clip-model ./models/clip-vit-base-patch32 \
  --prediction-kind florence2 \
  --splits valid
```

For LFM2.5, use `--prediction-kind small-vlm`. The public Hugging Face package
does not contain public-test ground truth, so the command can only reproduce
validation metrics without organizer-only annotations. The repository retains
the original full metric reports for auditability.

## Metric Policies

- Task A uses strict normalized lexical label matching and Macro-F1.
- Task B uses mAP at IoU thresholds 0.50 through 0.95, generation order as a
  deterministic confidence surrogate, and at most 100 detections per image.
- Invalid boxes are ignored only inside metric computation; saved predictions
  are not rewritten.
- CIDEr and SPICE use `pycocoevalcap`.
- CLIPScore uses OpenAI CLIP ViT-B/32 and `max(2.5 * cosine, 0)`.
