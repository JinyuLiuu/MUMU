#!/usr/bin/env python3
"""Evaluate small 2025+ vision-language models on the prepared MUMU split.

The models do not expose Florence-2's task tokens.  We therefore use explicit
natural-language instructions and keep their raw responses in the JSONL.  No
NMS, confidence calibration, duplicate removal, or other prediction filtering
is applied.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["valid", "public_test"])
    parser.add_argument("--tasks", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--label-vocabulary", type=Path)
    return parser.parse_args()


def load_records(root: Path, splits: list[str], tasks: list[str]) -> list[dict[str, Any]]:
    records = []
    for split in splits:
        internal_path = root / "annotations" / f"internal_{split}.jsonl"
        public_path = root / "manifests" / f"{split}.jsonl"
        path = internal_path if internal_path.exists() else public_path
        if not path.exists():
            raise FileNotFoundError(f"no annotation or manifest found for split {split!r}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["task"] in tasks:
                    records.append(row)
    return records


def load_completed(path: Path) -> tuple[set[str], Counter[str]]:
    if not path.exists():
        return set(), Counter()
    completed = set()
    task_counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                completed.add(row["id"])
                task_counts[row["task"]] += 1
            except (ValueError, KeyError):
                continue
    return completed, task_counts


def model_kind(model_dir: Path) -> str:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    kind = config.get("model_type")
    if kind not in {"lfm2_vl", "smolvlm"}:
        raise ValueError(f"unsupported model_type={kind!r}")
    return kind


def labels_by_dataset(root: Path, vocabulary_path: Path | None) -> dict[str, list[str]]:
    if vocabulary_path is not None:
        vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        if not isinstance(vocabulary, dict) or not all(
            isinstance(dataset, str)
            and isinstance(labels, list)
            and all(isinstance(label, str) for label in labels)
            for dataset, labels in vocabulary.items()
        ):
            raise ValueError("label vocabulary must map dataset names to string lists")
        return {dataset: sorted(set(labels)) for dataset, labels in vocabulary.items()}

    values: defaultdict[str, set[str]] = defaultdict(set)
    for split in ("valid", "public_test"):
        internal_path = root / "annotations" / f"internal_{split}.jsonl"
        public_path = root / "manifests" / f"{split}.jsonl"
        path = internal_path if internal_path.exists() else public_path
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["task"] == "A" and "ground_truth" in row:
                    values[row["dataset"]].update(row["ground_truth"]["labels"])
    if not values:
        raise RuntimeError("Task A requires --label-vocabulary when manifests have no ground truth")
    return {key: sorted(value) for key, value in values.items()}


def prompt_for(record: dict[str, Any], label_vocab: dict[str, list[str]]) -> str:
    task = record["task"]
    if task == "A":
        labels = ", ".join(label_vocab.get(record["dataset"], []))
        return (
            "Classify this image for the MUMU image-tagging task. Return only a comma-separated "
            "list of labels chosen from the candidate labels below, with no explanation. "
            f"Candidate labels: {labels}"
        )
    if task == "B":
        return (
            "Detect the visible objects in this image. Return only a JSON array, with one item "
            'per detected instance in the form {"label": "object name", "bbox": [x1, y1, x2, y2]}. '
            "Coordinates must be normalized to the range 0 to 1, where (0,0) is the top-left. "
            "Do not include markdown or explanations."
        )
    return (
        "Write one short COCO-style English caption of 8 to 12 words describing only the "
        "main visible content. Output only the caption."
    )


def decode_json_array(text: str) -> list[dict[str, Any]]:
    candidates = [text]
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    # Dense scenes can hit the generation limit after several complete items.
    # Retain each syntactically complete flat JSON object before the cutoff.
    items = []
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            items.append(value)
    return items


def parse_detection(text: str, width: int, height: int) -> dict[str, Any]:
    labels: list[str] = []
    boxes: list[list[float]] = []
    for item in decode_json_array(text):
        label = item.get("label") or item.get("name")
        raw_box = item.get("bbox")
        if raw_box is None and all(key in item for key in ("x1", "y1", "x2", "y2")):
            raw_box = [item["x1"], item["y1"], item["x2"], item["y2"]]
        if not isinstance(label, str) or not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        try:
            values = [float(value) for value in raw_box]
        except (TypeError, ValueError):
            continue
        # Models occasionally use 0..1000 despite the instruction.  Convert
        # that convention as well, while retaining every parsed detection.
        if max(abs(value) for value in values) <= 1.5:
            values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
        elif max(abs(value) for value in values) <= 1000.0:
            values = [values[0] * width / 1000.0, values[1] * height / 1000.0,
                      values[2] * width / 1000.0, values[3] * height / 1000.0]
        labels.append(label.strip())
        boxes.append(values)
    return {"labels": labels, "bboxes": boxes}


def make_inputs(processor: Any, kind: str, image: Image.Image, prompt: str, device: str, dtype: torch.dtype) -> dict[str, Any]:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[key] = value.to(device=device, dtype=dtype if torch.is_floating_point(value) else value.dtype)
        else:
            result[key] = value
    return result


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    kind = model_kind(args.model)
    print(f"loading {args.model} ({kind})", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, local_files_only=True,
    ).eval().to(args.device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameter_count:,}", flush=True)

    records = load_records(args.eval_root, args.splits, args.tasks)
    completed, existing_counts = load_completed(args.output)
    records = [record for record in records if record["id"] not in completed]
    vocab = labels_by_dataset(args.eval_root, args.label_vocabulary) if "A" in args.tasks else {}
    print(f"pending={len(records)} completed={len(completed)}", flush=True)
    counts: Counter[str] = Counter()
    timings: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    started_total = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    with args.output.open("a", encoding="utf-8") as handle:
        for task in args.tasks:
            for record in [item for item in records if item["task"] == task]:
                try:
                    with Image.open(args.eval_root / record["image"]) as source:
                        image = source.convert("RGB")
                    prompt = prompt_for(record, vocab)
                    inputs = make_inputs(processor, kind, image, prompt, args.device, dtype)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize(args.device)
                    timings[task] += time.perf_counter() - started
                    input_length = inputs["input_ids"].shape[1]
                    raw = processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)[0].strip()
                    prediction: Any = parse_detection(raw, image.width, image.height) if task == "B" else raw
                    row = {
                        "id": record["id"], "split": record["split"], "task": task,
                        "dataset": record["dataset"], "prompt": prompt,
                        "raw_output": raw, "prediction": prediction,
                    }
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    handle.flush()
                    counts[task] += 1
                    if counts[task] % 100 == 0:
                        print(f"task={task} done={counts[task]} rate={counts[task] / timings[task]:.2f} images/s", flush=True)
                except Exception as exc:
                    failures.append({"id": record["id"], "error": repr(exc)})
                    print(f"failed {record['id']}: {exc!r}", flush=True)
    peak = torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0
    summary = {
        "model": str(args.model), "model_type": kind, "parameter_count": parameter_count,
        "transformers_version": transformers.__version__,
        "dtype": args.dtype, "device": args.device, "max_new_tokens": args.max_new_tokens,
        "counts": dict(existing_counts + counts), "resumed_rows": sum(existing_counts.values()),
        "task_seconds": dict(timings),
        "total_seconds": time.perf_counter() - started_total,
        "peak_memory_bytes": peak, "failures": failures,
        "notes": ["Natural-language prompts replace Florence-2 task tokens.",
                  "No NMS, duplicate removal, confidence calibration, or custom filtering is applied.",
                  "Task B boxes are parsed only when the model returns JSON coordinates."],
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
