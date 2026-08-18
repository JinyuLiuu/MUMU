#!/usr/bin/env python3
"""Replace one task in a complete prediction JSONL with a variant run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replacement = {}
    with args.replacement.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["task"] != args.task:
                raise ValueError(f"replacement contains task {row['task']!r}, expected {args.task!r}")
            replacement[row["id"]] = row
    rows = replaced = 0
    with args.base.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            if row["task"] == args.task:
                row = replacement.pop(row["id"])
                replaced += 1
            target.write(json.dumps(row, ensure_ascii=True) + "\n")
            rows += 1
    if replacement:
        raise ValueError(f"replacement has {len(replacement)} IDs absent from base")
    print(json.dumps({"rows": rows, "replaced": replaced, "task": args.task}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
