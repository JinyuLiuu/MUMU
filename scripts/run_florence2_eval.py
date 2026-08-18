#!/usr/bin/env python3
"""Run Florence-2 inference on the prepared MUMU source-data evaluation split."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


TASK_SETTINGS = {
    # Keep the official notebook's generation budget for every task.
    "A": {"max_new_tokens": 1024},
    "B": {"max_new_tokens": 1024},
    "C": {"max_new_tokens": 1024},
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["valid", "public_test"])
    parser.add_argument("--tasks", nargs="+", default=["A", "B", "C"])
    parser.add_argument(
        "--batch-size-a", type=int, default=1,
        help="Kept for CLI compatibility; official mode processes one image at a time.",
    )
    parser.add_argument(
        "--batch-size-b", type=int, default=1,
        help="Kept for CLI compatibility; official mode processes one image at a time.",
    )
    parser.add_argument(
        "--batch-size-c", type=int, default=1,
        help="Kept for CLI compatibility; official mode processes one image at a time.",
    )
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
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


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                completed.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def move_inputs(inputs: dict[str, torch.Tensor], device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in inputs.items():
        if torch.is_floating_point(value):
            result[key] = value.to(device=device, dtype=dtype)
        else:
            result[key] = value.to(device=device)
    return result


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=True,
    ).eval().to(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameter_count:,}", flush=True)

    records = load_records(args.eval_root, args.splits, args.tasks)
    completed = load_completed(args.output)
    records = [record for record in records if record["id"] not in completed]
    print(f"pending={len(records)} completed={len(completed)}", flush=True)

    timings: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    start_total = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    with args.output.open("a", encoding="utf-8") as output_handle:
        for task in args.tasks:
            task_records = [record for record in records if record["task"] == task]
            setting = TASK_SETTINGS[task]
            for record in task_records:
                try:
                    with Image.open(args.eval_root / record["image"]) as source_image:
                        image = source_image.convert("RGB")
                except OSError as exc:
                    failures.append({"id": record["id"], "error": str(exc)})
                    continue

                # This mirrors the official notebook: a single image, no padding,
                # and the processor's native BatchFeature transfer path.
                inputs = processor(
                    text=record["prompt"],
                    images=image,
                    return_tensors="pt",
                )
                inputs = move_inputs(inputs, args.device, dtype)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                started = time.perf_counter()
                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=setting["max_new_tokens"],
                        early_stopping=False,
                        do_sample=False,
                        num_beams=args.num_beams,
                    )
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - started
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                try:
                    parsed = processor.post_process_generation(
                        generated_text,
                        task=record["prompt"],
                        image_size=(image.width, image.height),
                    )
                    prediction = parsed.get(record["prompt"], parsed)
                except Exception as exc:
                    prediction = None
                    failures.append({"id": record["id"], "error": str(exc)})
                output_handle.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "split": record["split"],
                            "task": task,
                            "dataset": record["dataset"],
                            "prompt": record["prompt"],
                            "raw_output": generated_text,
                            "prediction": prediction,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                output_handle.flush()
                counts[task] += 1
                timings[task] += elapsed
                done = counts[task]
                if done % 100 == 0 or done == len(task_records):
                    rate = done / timings[task]
                    print(
                        f"task={task} done={done}/{len(task_records)} rate={rate:.2f} images/s",
                        flush=True,
                    )

    elapsed_total = time.perf_counter() - start_total
    peak_memory = (
        torch.cuda.max_memory_allocated(args.device)
        if args.device.startswith("cuda")
        else 0
    )
    summary = {
        "model": str(args.model),
        "parameter_count": parameter_count,
        "dtype": args.dtype,
        "device": args.device,
        "num_beams": args.num_beams,
        "counts": dict(counts),
        "task_seconds": dict(timings),
        "total_seconds": elapsed_total,
        "peak_memory_bytes": peak_memory,
        "failures": failures,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
