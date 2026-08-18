#!/usr/bin/env python3
"""Atomically reparse Task B JSON from saved small-VLM raw outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from run_small_vlm_eval import parse_detection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    image_sizes = {}
    for split in ("valid", "public_test"):
        with (args.eval_root / "annotations" / f"internal_{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                image_sizes[row["id"]] = (row["width"], row["height"])
    output = args.predictions.with_suffix(args.predictions.suffix + ".reparsed.tmp")
    rows = parsed_rows = boxes = 0
    with args.predictions.open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            if row["task"] == "B":
                width, height = image_sizes[row["id"]]
                row["prediction"] = parse_detection(row["raw_output"], width, height)
                parsed_rows += bool(row["prediction"]["bboxes"])
                boxes += len(row["prediction"]["bboxes"])
            target.write(json.dumps(row, ensure_ascii=True) + "\n")
            rows += 1
    output.replace(args.predictions)
    print(json.dumps({"rows": rows, "task_b_parsed_rows": parsed_rows, "task_b_boxes": boxes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
