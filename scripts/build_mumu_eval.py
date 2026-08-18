#!/usr/bin/env python3
"""Build the deterministic 1,000/5,000 MUMU source-data evaluation split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


SEED = 20260817
ALLOCATIONS = {
    "KonIQ-10k": ("A", 67, 334),
    "SPAQ": ("A", 67, 334),
    "LIVE Challenge": ("A", 67, 333),
    "Places365": ("A", 67, 333),
    "NUS-WIDE": ("A", 66, 333),
    "Objects365": ("B", 111, 556),
    "LVIS": ("B", 111, 556),
    "Visual Genome": ("B", 111, 555),
    "COCO Captions": ("C", 111, 556),
    "Flickr30k": ("C", 111, 555),
    "Conceptual Captions": ("C", 111, 555),
}
TASK_PROMPTS = {
    "A": "<MORE_DETAILED_CAPTION>",
    "B": "<OD>",
    "C": "<CAPTION>",
}
IMAGE_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def shuffled(items: Iterable[Any], salt: str) -> list[Any]:
    result = list(items)
    random.Random(f"{SEED}:{salt}").shuffle(result)
    return result


def image_info(data: bytes) -> tuple[int, int, str, str]:
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        image_format = image.format or ""
        image.verify()
    extension = IMAGE_EXTENSIONS.get(image_format)
    if extension is None:
        raise ValueError(f"unsupported image format: {image_format!r}")
    if width < 16 or height < 16:
        raise ValueError(f"image is too small: {width}x{height}")
    return width, height, image_format, extension


class EvalWriter:
    def __init__(self, root: Path):
        self.root = root
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"output directory is not empty: {root}")
        (root / "images" / "valid").mkdir(parents=True, exist_ok=True)
        (root / "images" / "public_test").mkdir(parents=True, exist_ok=True)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.hashes: set[str] = set()
        self.split_counts: Counter[str] = Counter()
        self.source_counts: Counter[tuple[str, str]] = Counter()

    def quota(self, dataset: str, split: str) -> int:
        _, valid, public_test = ALLOCATIONS[dataset]
        return valid if split == "valid" else public_test

    def full(self, dataset: str, split: str) -> bool:
        return self.source_counts[(dataset, split)] >= self.quota(dataset, split)

    def add(
        self,
        *,
        dataset: str,
        split: str,
        source_id: str,
        source_split: str,
        data: bytes,
        ground_truth: dict[str, Any],
    ) -> bool:
        if self.full(dataset, split):
            return False
        digest = hashlib.sha256(data).hexdigest()
        if digest in self.hashes:
            return False
        try:
            width, height, image_format, extension = image_info(data)
        except (OSError, ValueError):
            return False

        task = ALLOCATIONS[dataset][0]
        ordinal = self.split_counts[split] + 1
        sample_id = f"{split}_{ordinal:06d}"
        relative_path = Path("images") / split / f"{sample_id}{extension}"
        destination = self.root / relative_path
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(data)
        os.replace(temporary, destination)

        self.hashes.add(digest)
        self.split_counts[split] += 1
        self.source_counts[(dataset, split)] += 1
        self.records.append(
            {
                "id": sample_id,
                "split": split,
                "task": task,
                "dataset": dataset,
                "image": relative_path.as_posix(),
                "width": width,
                "height": height,
                "format": image_format,
                "sha256": digest,
                "source_id": source_id,
                "source_split": source_split,
                "prompt": TASK_PROMPTS[task],
                "ground_truth": ground_truth,
            }
        )
        return True

    def assert_source_complete(self, dataset: str) -> None:
        for split in ("valid", "public_test"):
            actual = self.source_counts[(dataset, split)]
            expected = self.quota(dataset, split)
            if actual != expected:
                raise RuntimeError(
                    f"{dataset}/{split}: expected {expected}, built {actual}"
                )
        print(
            f"{dataset}: valid={self.source_counts[(dataset, 'valid')]} "
            f"public_test={self.source_counts[(dataset, 'public_test')]}",
            flush=True,
        )

    def finalize(self) -> None:
        if self.split_counts != Counter({"public_test": 5000, "valid": 1000}):
            raise RuntimeError(f"unexpected split counts: {self.split_counts}")
        if len(self.hashes) != 6000:
            raise RuntimeError(f"expected 6000 unique hashes, got {len(self.hashes)}")

        records = sorted(self.records, key=lambda row: row["id"])
        internal_paths = {
            "valid": self.root / "annotations" / "internal_valid.jsonl",
            "public_test": self.root / "annotations" / "internal_public_test.jsonl",
        }
        public_paths = {
            "valid": self.root / "manifests" / "valid.jsonl",
            "public_test": self.root / "manifests" / "public_test.jsonl",
        }
        handles = {
            key: path.open("w", encoding="utf-8")
            for key, path in {**internal_paths, **{f"public_{k}": v for k, v in public_paths.items()}}.items()
        }
        try:
            for row in records:
                split = row["split"]
                handles[split].write(json.dumps(row, ensure_ascii=True) + "\n")
                public_row = {k: v for k, v in row.items() if k != "ground_truth"}
                if split == "valid":
                    public_row["ground_truth"] = row["ground_truth"]
                handles[f"public_{split}"].write(
                    json.dumps(public_row, ensure_ascii=True) + "\n"
                )
        finally:
            for handle in handles.values():
                handle.close()

        summary = {
            "seed": SEED,
            "total": len(records),
            "split_counts": dict(self.split_counts),
            "task_counts": dict(Counter(row["task"] for row in records)),
            "dataset_counts": {
                dataset: {
                    split: self.source_counts[(dataset, split)]
                    for split in ("valid", "public_test")
                }
                for dataset in ALLOCATIONS
            },
            "unique_sha256": len(self.hashes),
        }
        (self.root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def candidate_pools(
    rows: list[Any], valid: int, public_test: int, salt: str, extra: int = 80
) -> dict[str, list[Any]]:
    rows = shuffled(rows, salt)
    valid_end = valid + extra
    test_end = valid_end + public_test + extra
    if len(rows) < test_end:
        raise RuntimeError(f"{salt}: only {len(rows)} candidates, need {test_end}")
    return {
        "valid": rows[:valid_end],
        "public_test": rows[valid_end:test_end],
    }


def quality_ground_truth(
    row: dict[str, str], mos_key: str, low: float, high: float
) -> dict[str, Any]:
    mos = float(row[mos_key])
    quality_bin = "low quality" if mos <= low else "high quality" if mos >= high else "medium quality"
    result: dict[str, Any] = {
        "type": "quality",
        "mos": mos,
        "quality_bin": quality_bin,
        "labels": [quality_bin],
        "mos_tertiles": [low, high],
    }
    for source_key, output_key in (
        ("Brightness", "brightness"),
        ("Colorfulness", "colorfulness"),
        ("Contrast", "contrast"),
        ("Noisiness", "noisiness"),
        ("Sharpness", "sharpness"),
    ):
        if source_key in row and row[source_key]:
            result[output_key] = float(row[source_key])
    return result


def build_quality_archives(root: Path, writer: EvalWriter) -> None:
    meta_root = root / "quality" / "chaofengc_IQA-PyTorch-Datasets-metainfo"
    archive_root = root / "quality" / "chaofengc_IQA-PyTorch-Datasets"
    configs = [
        {
            "dataset": "KonIQ-10k",
            "csv": meta_root / "meta_info_KonIQ10kDataset.csv",
            "archive": archive_root / "koniq10k.tgz",
            "name_key": "img_name",
            "mos_key": "mos",
            "member": lambda name: f"koniq10k/512x384/{name}",
            "native": ("official_split", "val", "test"),
        },
        {
            "dataset": "SPAQ",
            "csv": meta_root / "meta_info_SPAQDataset.csv",
            "archive": archive_root / "spaq.tgz",
            "name_key": "Image name",
            "mos_key": "MOS",
            "member": lambda name: f"SPAQ/TestImage/{name}",
            "native": None,
        },
        {
            "dataset": "LIVE Challenge",
            "csv": meta_root / "meta_info_LIVEChallengeDataset.csv",
            "archive": archive_root / "live_challenge.tgz",
            "name_key": "img_name",
            "mos_key": "mos",
            "member": lambda name: (
                f"LIVEC/Images/trainingImages/{name}"
                if name.startswith("t")
                else f"LIVEC/Images/{name}"
            ),
            "native": ("ratio802_seed123_split_01", "val", "train"),
        },
    ]

    for config in configs:
        with config["csv"].open(newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle) if row[config["mos_key"]]]
        mos_values = np.array([float(row[config["mos_key"]]) for row in rows])
        low, high = (float(x) for x in np.quantile(mos_values, [1 / 3, 2 / 3]))
        _, valid_count, test_count = ALLOCATIONS[config["dataset"]]

        if config["native"]:
            key, valid_value, test_value = config["native"]
            valid_rows = shuffled(
                [row for row in rows if row.get(key) == valid_value],
                f"{config['dataset']}:valid",
            )[: valid_count + 80]
            test_rows = shuffled(
                [row for row in rows if row.get(key) == test_value],
                f"{config['dataset']}:public_test",
            )[: test_count + 80]
            pools = {"valid": valid_rows, "public_test": test_rows}
        else:
            pools = candidate_pools(
                rows, valid_count, test_count, config["dataset"]
            )

        selected: dict[str, tuple[str, dict[str, str]]] = {}
        for split, candidates in pools.items():
            for row in candidates:
                selected[config["member"](row[config["name_key"]])] = (split, row)

        with tarfile.open(config["archive"], "r:gz") as archive:
            for member in archive:
                item = selected.get(member.name)
                if item is None or not member.isfile():
                    continue
                split, row = item
                if writer.full(config["dataset"], split):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                writer.add(
                    dataset=config["dataset"],
                    split=split,
                    source_id=member.name,
                    source_split=(
                        row.get("official_split")
                        or row.get("ratio802_seed123_split_01")
                        or "unspecified"
                    ),
                    data=extracted.read(),
                    ground_truth=quality_ground_truth(
                        row, config["mos_key"], low, high
                    ),
                )
        writer.assert_source_complete(config["dataset"])


def parquet_feature_names(file: Path, feature: str) -> dict[int, str]:
    metadata = pq.ParquetFile(file).schema_arrow.metadata or {}
    huggingface = json.loads(metadata[b"huggingface"])
    names = huggingface["info"]["features"][feature]["names"]
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(key): value for key, value in names.items()}


def process_parquet_indices(
    file: Path,
    selections: dict[int, str],
    columns: list[str],
    callback: Callable[[str, int, dict[str, Any]], None],
) -> None:
    parquet = pq.ParquetFile(file)
    offset = 0
    for group_index in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        local = [index - offset for index in selections if offset <= index < offset + count]
        if local:
            table = parquet.read_row_group(group_index, columns=columns)
            rows = table.take(local).to_pylist()
            for local_index, row in zip(local, rows):
                global_index = offset + local_index
                callback(selections[global_index], global_index, row)
        offset += count


def build_places(root: Path, writer: EvalWriter) -> None:
    file = sorted((root / "places365" / "ljnlonoljpiljm_places365-256px" / "data").glob("*.parquet"))[0]
    _, valid_count, test_count = ALLOCATIONS["Places365"]
    pools = candidate_pools(
        list(range(pq.ParquetFile(file).metadata.num_rows)),
        valid_count,
        test_count,
        "Places365",
    )
    selections = {index: split for split, indices in pools.items() for index in indices}
    names = parquet_feature_names(file, "label")

    def add(split: str, index: int, row: dict[str, Any]) -> None:
        label = names[int(row["label"])]
        writer.add(
            dataset="Places365",
            split=split,
            source_id=f"{file.name}:{index}",
            source_split="train",
            data=row["image"]["bytes"],
            ground_truth={"type": "scene", "labels": [label], "class_id": row["label"]},
        )

    process_parquet_indices(file, selections, ["image", "label"], add)
    writer.assert_source_complete("Places365")


def build_nus_wide(root: Path, writer: EvalWriter) -> None:
    label_zip_path = root / "nus_wide" / "lxyhaha_NUS-WIDE" / "NUS-WIDE.zip"
    image_zip_path = root / "nus_wide" / "moneyzz432_nus_wide" / "Flickr.zip"
    with zipfile.ZipFile(label_zip_path) as labels_zip:
        concepts = labels_zip.read("ConceptsList/Concepts81.txt").decode().splitlines()
        columns = []
        for concept in concepts:
            values = np.fromstring(
                labels_zip.read(f"Groundtruth/AllLabels/Labels_{concept}.txt").decode(),
                dtype=np.int8,
                sep="\n",
            )
            columns.append(values)
        labels = np.column_stack(columns)

    event_names = {"dancing", "earthquake", "fire", "protest", "running", "soccer", "sports", "wedding"}
    event_columns = [concepts.index(name) for name in sorted(event_names)]
    candidates = np.flatnonzero(labels[:, event_columns].any(axis=1)).tolist()
    _, valid_count, test_count = ALLOCATIONS["NUS-WIDE"]
    pools = candidate_pools(candidates, valid_count, test_count, "NUS-WIDE")
    selections = {index: split for split, indices in pools.items() for index in indices}

    with zipfile.ZipFile(image_zip_path) as images_zip:
        image_members = [
            name
            for name in images_zip.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(image_members) != labels.shape[0]:
            raise RuntimeError(
                f"NUS-WIDE image/label mismatch: {len(image_members)} vs {labels.shape[0]}"
            )
        for index, split in selections.items():
            positive = [concepts[i] for i in np.flatnonzero(labels[index])]
            writer.add(
                dataset="NUS-WIDE",
                split=split,
                source_id=image_members[index],
                source_split="all",
                data=images_zip.read(image_members[index]),
                ground_truth={"type": "event", "labels": positive},
            )
    writer.assert_source_complete("NUS-WIDE")


def boxes_from_annotations(annotations: list[dict[str, Any]], category_key: str) -> dict[str, Any]:
    boxes = []
    labels = []
    category_ids = []
    for annotation in annotations:
        bbox = [float(value) for value in annotation["bbox"]]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        boxes.append(bbox)
        labels.append(str(annotation[category_key]))
        if "category_id" in annotation:
            category_ids.append(int(annotation["category_id"]))
    return {
        "type": "detection",
        "bbox_format": "xywh",
        "boxes": boxes,
        "labels": labels,
        "category_ids": category_ids,
    }


def build_objects365(root: Path, writer: EvalWriter) -> None:
    annotation_root = root / "objects365" / "jxu124_objects365" / "data"
    annotations: dict[str, dict[str, Any]] = {}
    for file in sorted(annotation_root.glob("*.parquet")):
        parquet = pq.ParquetFile(file)
        for group_index in range(parquet.num_row_groups):
            paths = parquet.read_row_group(group_index, columns=["image_path"])
            indices = [
                index
                for index, path in enumerate(paths.column(0).to_pylist())
                if "/patch16/" in path
            ]
            if not indices:
                continue
            table = parquet.read_row_group(
                group_index, columns=["image_path", "image_info", "anns_info"]
            ).take(indices)
            for row in table.to_pylist():
                annotations[Path(row["image_path"]).name] = row

    _, valid_count, test_count = ALLOCATIONS["Objects365"]
    pools = candidate_pools(
        list(annotations), valid_count, test_count, "Objects365"
    )
    selections = {name: split for split, names in pools.items() for name in names}
    archive_path = root / "objects365" / "guozonghao96_objects365" / "patch16.tar.gz"
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = Path(member.name).name
            split = selections.get(name)
            if split is None or not member.isfile() or writer.full("Objects365", split):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            row = annotations[name]
            writer.add(
                dataset="Objects365",
                split=split,
                source_id=member.name,
                source_split="patch16",
                data=extracted.read(),
                ground_truth=boxes_from_annotations(row["anns_info"], "category"),
            )
    writer.assert_source_complete("Objects365")


def read_json_zip(path: Path) -> Any:
    with zipfile.ZipFile(path) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            return json.load(handle)


def build_lvis(root: Path, writer: EvalWriter) -> None:
    data = read_json_zip(root / "lvis" / "official_annotations" / "lvis_v1_val.json.zip")
    categories = {int(row["id"]): row["name"] for row in data["categories"]}
    annotations: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotation = dict(annotation)
        annotation["category"] = categories[int(annotation["category_id"])]
        annotations[int(annotation["image_id"])].append(annotation)

    coco_root = root / "coco_captions"
    coco_files = sorted(
        (coco_root / "Multimodal-Fatima_COCO_captions_train" / "data").glob("*.parquet")
    ) + sorted(
        (coco_root / "Multimodal-Fatima_COCO_captions_validation" / "data").glob("*.parquet")
    )
    available: list[tuple[Path, int, int]] = []
    for coco_file in coco_files:
        coco_ids = pq.read_table(coco_file, columns=["cocoid"]).column(0).to_pylist()
        available.extend(
            (coco_file, index, int(image_id))
            for index, image_id in enumerate(coco_ids)
            if int(image_id) in annotations
        )
        if len(available) >= 2000:
            break
    _, valid_count, test_count = ALLOCATIONS["LVIS"]
    pools = candidate_pools(available, valid_count, test_count, "LVIS")
    by_file: defaultdict[Path, dict[int, str]] = defaultdict(dict)
    for split, locations in pools.items():
        for coco_file, index, _ in locations:
            by_file[coco_file][index] = split

    for coco_file, selections in by_file.items():
        def add(split: str, index: int, row: dict[str, Any]) -> None:
            image_id = int(row["cocoid"])
            writer.add(
                dataset="LVIS",
                split=split,
                source_id=f"coco:{image_id}",
                source_split="lvis_v1_val",
                data=row["image"]["bytes"],
                ground_truth=boxes_from_annotations(annotations[image_id], "category"),
            )

        process_parquet_indices(coco_file, selections, ["image", "cocoid"], add)
    writer.assert_source_complete("LVIS")


def build_visual_genome(root: Path, writer: EvalWriter) -> None:
    archive_root = root / "visual_genome" / "official_archives"
    objects = read_json_zip(archive_root / "objects.json.zip")
    by_id = {
        int(row["image_id"]): row["objects"]
        for row in objects
        if row.get("objects")
    }
    del objects

    _, valid_count, test_count = ALLOCATIONS["Visual Genome"]
    pools = candidate_pools(list(by_id), valid_count, test_count, "Visual Genome")
    selections = {image_id: split for split, ids in pools.items() for image_id in ids}

    zip_paths = [archive_root / "images.zip", archive_root / "images2.zip"]
    archives = [zipfile.ZipFile(path) for path in zip_paths]
    try:
        member_maps = []
        for archive in archives:
            member_maps.append(
                {
                    int(Path(name).stem): name
                    for name in archive.namelist()
                    if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
                    and Path(name).stem.isdigit()
                }
            )
        for image_id, split in selections.items():
            archive_index = 0 if image_id in member_maps[0] else 1
            member = member_maps[archive_index].get(image_id)
            if member is None:
                continue
            annotations = []
            for obj in by_id[image_id]:
                names = obj.get("names") or []
                if not names:
                    continue
                annotations.append(
                    {
                        "bbox": [obj["x"], obj["y"], obj["w"], obj["h"]],
                        "category": names[0],
                    }
                )
            writer.add(
                dataset="Visual Genome",
                split=split,
                source_id=member,
                source_split="all",
                data=archives[archive_index].read(member),
                ground_truth=boxes_from_annotations(annotations, "category"),
            )
    finally:
        for archive in archives:
            archive.close()
    writer.assert_source_complete("Visual Genome")


def build_caption_parquet(
    *,
    dataset: str,
    valid_file: Path,
    test_file: Path,
    image_id_key: str,
    captions_key: str,
    writer: EvalWriter,
) -> None:
    _, valid_count, test_count = ALLOCATIONS[dataset]
    for split, file, needed in (
        ("valid", valid_file, valid_count),
        ("public_test", test_file, test_count),
    ):
        candidates = shuffled(
            list(range(pq.ParquetFile(file).metadata.num_rows)),
            f"{dataset}:{split}",
        )[: needed + 80]
        selections = {index: split for index in candidates}

        def add(row_split: str, index: int, row: dict[str, Any]) -> None:
            captions = row[captions_key]
            if isinstance(captions, str):
                captions = [captions]
            writer.add(
                dataset=dataset,
                split=row_split,
                source_id=f"{file.name}:{row[image_id_key]}",
                source_split=split,
                data=row["image"]["bytes"],
                ground_truth={"type": "caption", "captions": captions},
            )

        process_parquet_indices(
            file, selections, ["image", image_id_key, captions_key], add
        )
    writer.assert_source_complete(dataset)


def build_coco_captions(root: Path, writer: EvalWriter) -> None:
    valid_file = sorted(
        (root / "coco_captions" / "Multimodal-Fatima_COCO_captions_validation" / "data").glob("*.parquet")
    )[1]
    test_file = sorted(
        (root / "coco_captions" / "Multimodal-Fatima_COCO_captions_test" / "data").glob("*.parquet")
    )[0]
    build_caption_parquet(
        dataset="COCO Captions",
        valid_file=valid_file,
        test_file=test_file,
        image_id_key="cocoid",
        captions_key="sentences_raw",
        writer=writer,
    )


def build_flickr30k(root: Path, writer: EvalWriter) -> None:
    data_root = root / "flickr30k" / "noonamkha_flickr30k-karpathy" / "data"
    build_caption_parquet(
        dataset="Flickr30k",
        valid_file=data_root / "validation-00000-of-00001.parquet",
        test_file=data_root / "test-00000-of-00001.parquet",
        image_id_key="image_id",
        captions_key="captions",
        writer=writer,
    )


def build_conceptual_captions(root: Path, writer: EvalWriter) -> None:
    image_root = root / "conceptual_captions" / "images_validation_official_urls"
    with (image_root / "manifest.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    _, valid_count, test_count = ALLOCATIONS["Conceptual Captions"]
    pools = candidate_pools(
        rows, valid_count, test_count, "Conceptual Captions", extra=40
    )
    for split, candidates in pools.items():
        for row in candidates:
            if writer.full("Conceptual Captions", split):
                break
            writer.add(
                dataset="Conceptual Captions",
                split=split,
                source_id=row["url"],
                source_split="official_validation_urls",
                data=(image_root / row["filename"]).read_bytes(),
                ground_truth={"type": "caption", "captions": [row["caption"]]},
            )
    writer.assert_source_complete("Conceptual Captions")


def main() -> int:
    args = parse_args()
    writer = EvalWriter(args.output)
    build_quality_archives(args.datasets_root, writer)
    build_places(args.datasets_root, writer)
    build_nus_wide(args.datasets_root, writer)
    build_objects365(args.datasets_root, writer)
    build_lvis(args.datasets_root, writer)
    build_visual_genome(args.datasets_root, writer)
    build_coco_captions(args.datasets_root, writer)
    build_flickr30k(args.datasets_root, writer)
    build_conceptual_captions(args.datasets_root, writer)
    writer.finalize()
    print(json.dumps(json.loads((args.output / "summary.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
