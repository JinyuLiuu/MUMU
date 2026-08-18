#!/usr/bin/env python3
"""Evaluate Florence-2 predictions on the prepared MUMU source-data split."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
SPECIAL_TOKEN_RE = re.compile(r"<(?:pad|s|/s)>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path)
    parser.add_argument("--clip-device", default="cuda")
    parser.add_argument("--clip-batch-size", type=int, default=64)
    parser.add_argument("--skip-spice", action="store_true")
    parser.add_argument(
        "--splits", nargs="+", choices=["valid", "public_test"],
        default=["valid", "public_test"],
    )
    parser.add_argument(
        "--prediction-kind", choices=["florence2", "small-vlm"], default="florence2"
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(SPECIAL_TOKEN_RE.sub(" ", value).split())


def normalized_words(value: str) -> str:
    value = value.casefold().replace("_", " ").replace("/", " ").replace("-", " ")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def label_is_present(label: str, text: str) -> bool:
    label_words = normalized_words(label)
    text_words = f" {normalized_words(text)} "
    return bool(label_words) and f" {label_words} " in text_words


def macro_f1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({label for row in rows for label in row["ground_truth"]["labels"]})
    counts = {label: [0, 0, 0] for label in labels}  # TP, FP, FN
    exact_matches = 0
    predicted_total = 0
    ground_truth_total = 0
    for row in rows:
        truth = set(row["ground_truth"]["labels"])
        predicted = {label for label in labels if label_is_present(label, row["prediction_text"])}
        exact_matches += int(predicted == truth)
        predicted_total += len(predicted)
        ground_truth_total += len(truth)
        for label in labels:
            if label in truth and label in predicted:
                counts[label][0] += 1
            elif label in predicted:
                counts[label][1] += 1
            elif label in truth:
                counts[label][2] += 1
    per_class = {}
    for label, (tp, fp, fn) in counts.items():
        denominator = 2 * tp + fp + fn
        per_class[label] = 2 * tp / denominator if denominator else 0.0
    return {
        "macro_f1": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "classes": len(labels),
        "images": len(rows),
        "ground_truth_labels": ground_truth_total,
        "predicted_labels": predicted_total,
        "exact_set_accuracy": exact_matches / len(rows) if rows else 0.0,
    }


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def valid_box(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def box_iou(box: list[float], boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty(0, dtype=np.float64)
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    values = []
    for recall_threshold in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= recall_threshold]
        values.append(float(eligible.max()) if len(eligible) else 0.0)
    return float(np.mean(values))


def detection_ap(rows: list[dict[str, Any]], class_agnostic: bool) -> dict[str, Any]:
    ground_truth: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    predictions: dict[str, list[tuple[float, str, list[float]]]] = defaultdict(list)
    gt_count = 0
    prediction_count = 0
    for row in rows:
        gt = row["ground_truth"]
        for box, label in zip(gt["boxes"], gt["labels"]):
            converted = xywh_to_xyxy(box)
            if valid_box(converted):
                category = "__object__" if class_agnostic else normalized_words(label)
                ground_truth[category][row["id"]].append(converted)
                gt_count += 1

        prediction = row.get("prediction")
        if not isinstance(prediction, dict):
            continue
        boxes = prediction.get("bboxes", [])[:100]
        labels = prediction.get("labels", [])[:100]
        for rank, (box, label) in enumerate(zip(boxes, labels)):
            if valid_box(box):
                category = "__object__" if class_agnostic else normalized_words(str(label))
                score = 1.0 - rank * 1e-6
                predictions[category].append((score, row["id"], list(map(float, box))))
                prediction_count += 1

    per_threshold = []
    recall_per_threshold = []
    categories = sorted(ground_truth)
    for threshold in IOU_THRESHOLDS:
        class_aps = []
        total_tp = 0
        for category in categories:
            gt_by_image = ground_truth[category]
            category_gt_count = sum(len(boxes) for boxes in gt_by_image.values())
            matched = {
                image_id: np.zeros(len(boxes), dtype=bool)
                for image_id, boxes in gt_by_image.items()
            }
            true_positives = []
            false_positives = []
            for _, image_id, predicted_box in sorted(
                predictions.get(category, []), key=lambda item: item[0], reverse=True
            ):
                image_boxes = np.asarray(gt_by_image.get(image_id, []), dtype=np.float64).reshape(-1, 4)
                ious = box_iou(predicted_box, image_boxes)
                available = np.where(~matched.get(image_id, np.empty(0, dtype=bool)))[0]
                if len(available):
                    best = available[np.argmax(ious[available])]
                    is_match = ious[best] >= threshold
                else:
                    best = -1
                    is_match = False
                true_positives.append(float(is_match))
                false_positives.append(float(not is_match))
                if is_match:
                    matched[image_id][best] = True
            tp = np.cumsum(true_positives)
            fp = np.cumsum(false_positives)
            recall = tp / category_gt_count
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            class_aps.append(interpolated_ap(recall, precision))
            total_tp += int(tp[-1]) if len(tp) else 0
        per_threshold.append(float(np.mean(class_aps)) if class_aps else 0.0)
        recall_per_threshold.append(total_tp / gt_count if gt_count else 0.0)
    return {
        "map_50_95": float(np.mean(per_threshold)),
        "ap_50": per_threshold[0],
        "ap_75": per_threshold[5],
        "recall_50": recall_per_threshold[0],
        "categories": len(categories),
        "ground_truth_boxes": gt_count,
        "predicted_boxes": prediction_count,
        "confidence_policy": "generation order, descending by 1e-6",
    }


def tokenize_captions(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    raw_references = {
        row["id"]: [{"caption": caption} for caption in row["ground_truth"]["captions"]]
        for row in rows
    }
    raw_predictions = {
        row["id"]: [{"caption": row["prediction_text"]}] for row in rows
    }
    tokenizer = PTBTokenizer()
    return tokenizer.tokenize(raw_references), tokenizer.tokenize(raw_predictions)


def compute_spice(
    references: dict[str, list[str]], predictions: dict[str, list[str]]
) -> tuple[float, dict[str, float]]:
    from pycocoevalcap.spice import spice as spice_module

    image_ids = sorted(references)
    input_data = [
        {"image_id": image_id, "test": predictions[image_id][0], "refs": references[image_id]}
        for image_id in image_ids
    ]
    module_dir = Path(spice_module.__file__).resolve().parent
    temp_dir = module_dir / spice_module.TEMP_DIR
    cache_dir = module_dir / spice_module.CACHE_DIR
    temp_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=temp_dir, encoding="utf-8") as input_file:
        json.dump(input_data, input_file)
        input_path = input_file.name
    with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir) as output_file:
        output_path = output_file.name
    command = [
        "java",
        "-Xmx8G",
        "-jar",
        spice_module.SPICE_JAR,
        input_path,
        "-cache",
        str(cache_dir),
        "-out",
        output_path,
        "-subset",
        "-silent",
    ]
    try:
        subprocess.check_call(command, cwd=module_dir)
        results = json.loads(Path(output_path).read_text(encoding="utf-8"))
    finally:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
    scorer = spice_module.Spice()
    per_image = {
        item["image_id"]: scorer.float_convert(item["scores"]["All"]["f"])
        for item in results
    }
    return float(np.nanmean(list(per_image.values()))), per_image


def caption_language_scores(
    rows: list[dict[str, Any]], skip_spice: bool
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    from pycocoevalcap.cider.cider import Cider

    references, predictions = tokenize_captions(rows)
    cider, cider_values = Cider().compute_score(references, predictions)
    image_ids = sorted(references)
    per_image = {
        image_id: {"cider": float(value)}
        for image_id, value in zip(image_ids, cider_values)
    }
    result: dict[str, Any] = {"cider": float(cider)}
    if not skip_spice:
        spice, spice_values = compute_spice(references, predictions)
        result["spice"] = spice
        for image_id, value in spice_values.items():
            per_image[image_id]["spice"] = float(value)
    return result, per_image


def clip_scores(
    rows: list[dict[str, Any]], model_path: Path, device: str, batch_size: int
) -> dict[str, float]:
    from transformers import CLIPModel, CLIPProcessor

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = CLIPModel.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype
    ).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
    scores: dict[str, float] = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        images = []
        for row in batch:
            with Image.open(row["absolute_image"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(
            text=[row["prediction_text"] for row in batch],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            values = 2.5 * torch.clamp((image_features * text_features).sum(dim=-1), min=0)
        scores.update(
            {row["id"]: float(value) for row, value in zip(batch, values.cpu())}
        )
    return scores


def grouped_means(
    rows: list[dict[str, Any]], per_image: dict[str, dict[str, float]]
) -> dict[str, Any]:
    result = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        ids = [row["id"] for row in rows if row["dataset"] == dataset]
        metrics = sorted({metric for image_id in ids for metric in per_image[image_id]})
        result[dataset] = {
            metric: float(np.nanmean([per_image[image_id][metric] for image_id in ids]))
            for metric in metrics
        }
        result[dataset]["images"] = len(ids)
    return result


def main() -> int:
    args = parse_args()
    annotations = []
    for split in args.splits:
        internal_path = args.eval_root / "annotations" / f"internal_{split}.jsonl"
        public_path = args.eval_root / "manifests" / f"{split}.jsonl"
        path = internal_path if internal_path.exists() else public_path
        if not path.exists():
            raise FileNotFoundError(f"no annotation or manifest found for split {split!r}")
        split_annotations = read_jsonl(path)
        if any("ground_truth" not in row for row in split_annotations):
            raise RuntimeError(
                f"split {split!r} has no public ground truth; request a labeled split"
            )
        annotations.extend(split_annotations)
    annotation_by_id = {row["id"]: row for row in annotations}
    prediction_rows = read_jsonl(args.predictions)
    prediction_by_id = {
        row["id"]: row for row in prediction_rows if row["id"] in annotation_by_id
    }
    missing = sorted(set(annotation_by_id) - set(prediction_by_id))
    if missing:
        raise RuntimeError(f"predictions are missing {len(missing)} requested IDs")

    rows = []
    for image_id, annotation in annotation_by_id.items():
        prediction = prediction_by_id[image_id]
        row = {**annotation, "prediction": prediction.get("prediction")}
        row["prediction_text"] = clean_text(prediction.get("prediction"))
        row["absolute_image"] = str(args.eval_root / annotation["image"])
        rows.append(row)

    if args.prediction_kind == "florence2":
        model_limitations = [
            "Task A uses strict lexical extraction from detailed captions because Florence-2 has no tagging head.",
            "Task B novel categories are unspecified and Florence-2 OD emits no confidence scores.",
        ]
    else:
        model_limitations = [
            "Task A uses an explicit source-data label vocabulary because the challenge taxonomy is unspecified.",
            "Task B parses the model's normalized JSON boxes; generation order is used because it emits no confidence scores.",
        ]
    report: dict[str, Any] = {
        "scope": "source-data baseline; not an official MUMU challenge score",
        "predictions": str(args.predictions),
        "prediction_kind": args.prediction_kind,
        "counts": {
            "total": len(rows),
            **{
                split: sum(row["split"] == split for row in rows)
                for split in args.splits
            },
        },
        "splits": {},
        "limitations": model_limitations + [
            "Task C CLIPScore uses OpenAI CLIP ViT-B/32 with max(2.5*cosine, 0).",
            "The challenge taxonomy, novel split, alpha, and beta are not specified.",
        ],
    }

    for split in args.splits:
        split_rows = [row for row in rows if row["split"] == split]
        task_a = [row for row in split_rows if row["task"] == "A"]
        task_b = [row for row in split_rows if row["task"] == "B"]
        report["splits"][split] = {
            "task_a": {
                "overall": macro_f1(task_a),
                "by_dataset": {
                    dataset: macro_f1([row for row in task_a if row["dataset"] == dataset])
                    for dataset in sorted({row["dataset"] for row in task_a})
                },
            },
            "task_b": {
                "label_aware": detection_ap(task_b, class_agnostic=False),
                "class_agnostic": detection_ap(task_b, class_agnostic=True),
                "by_dataset": {
                    dataset: {
                        "label_aware": detection_ap(
                            [row for row in task_b if row["dataset"] == dataset], False
                        ),
                        "class_agnostic": detection_ap(
                            [row for row in task_b if row["dataset"] == dataset], True
                        ),
                    }
                    for dataset in sorted({row["dataset"] for row in task_b})
                },
            },
        }

    task_c_rows = [row for row in rows if row["task"] == "C"]
    for split in args.splits:
        split_rows = [row for row in task_c_rows if row["split"] == split]
        language_scores, per_image = caption_language_scores(split_rows, args.skip_spice)
        if args.clip_model:
            clip_values = clip_scores(
                split_rows, args.clip_model, args.clip_device, args.clip_batch_size
            )
            language_scores["clipscore"] = float(np.mean(list(clip_values.values())))
            for image_id, value in clip_values.items():
                per_image[image_id]["clipscore"] = value
        if "spice" in language_scores and "clipscore" in language_scores:
            language_scores["composite"] = (
                0.4 * language_scores["cider"]
                + 0.3 * language_scores["spice"]
                + 0.3 * language_scores["clipscore"]
            )
        language_scores["images"] = len(split_rows)
        report["splits"][split]["task_c"] = {
            "overall": language_scores,
            "by_dataset": grouped_means(split_rows, per_image),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
