#!/usr/bin/env python3
"""Download a deterministic, resumable Conceptual Captions image pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import random
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


FORMAT_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=800)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def load_existing(manifest_path: Path) -> tuple[set[str], int]:
    urls: set[str] = set()
    count = 0
    if not manifest_path.exists():
        return urls, count
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if (manifest_path.parent / row["filename"]).is_file():
                urls.add(row["url"])
                count += 1
    return urls, count


def read_response(response, max_bytes: int) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared and int(declared) > max_bytes:
        raise ValueError(f"content length exceeds {max_bytes} bytes")
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"download exceeds {max_bytes} bytes")
        chunks.append(chunk)


def fetch_image(
    item: tuple[int, str, str],
    output: Path,
    timeout: float,
    retries: int,
    max_bytes: int,
) -> tuple[dict[str, object] | None, str | None]:
    index, url, caption = item
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; MUMU-dataset-fetch/1.0)",
            "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*,*/*;q=0.8",
        },
    )
    last_error = "unknown error"
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = read_response(response, max_bytes)
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image_format = image.format or ""
                width, height = image.size
            extension = FORMAT_EXTENSIONS.get(image_format)
            if extension is None:
                raise ValueError(f"unsupported image format {image_format!r}")
            if width < 32 or height < 32 or width * height > 100_000_000:
                raise ValueError(f"invalid dimensions {width}x{height}")
            digest = hashlib.sha256(data).hexdigest()
            filename = f"cc_valid_{index:05d}_{digest[:12]}{extension}"
            destination = output / filename
            temporary = output / f".{filename}.part"
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            return (
                {
                    "source_index": index,
                    "filename": filename,
                    "url": url,
                    "caption": caption,
                    "sha256": digest,
                    "bytes": len(data),
                    "width": width,
                    "height": height,
                    "format": image_format,
                },
                None,
            )
        except (
            OSError,
            ValueError,
            socket.timeout,
            urllib.error.URLError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return None, last_error


def main() -> int:
    args = parse_args()
    if args.target < 1 or args.workers < 1:
        raise SystemExit("target and workers must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.tsv"
    failures_path = args.output / "failures.tsv"
    existing_urls, success_count = load_existing(manifest_path)
    if success_count >= args.target:
        print(f"target already satisfied: {success_count} images")
        return 0

    table = pq.read_table(args.parquet, columns=["image_url", "caption"])
    candidates = [
        (index, row["image_url"], row["caption"])
        for index, row in enumerate(table.to_pylist())
        if row["image_url"] and row["image_url"] not in existing_urls
    ]
    random.Random(args.seed).shuffle(candidates)

    manifest_exists = manifest_path.exists() and manifest_path.stat().st_size > 0
    failures_exists = failures_path.exists() and failures_path.stat().st_size > 0
    fields = [
        "source_index",
        "filename",
        "url",
        "caption",
        "sha256",
        "bytes",
        "width",
        "height",
        "format",
    ]
    batch_size = args.workers * 4

    with (
        manifest_path.open("a", newline="", encoding="utf-8") as manifest_handle,
        failures_path.open("a", newline="", encoding="utf-8") as failures_handle,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        manifest_writer = csv.DictWriter(
            manifest_handle, fieldnames=fields, delimiter="\t"
        )
        failure_writer = csv.DictWriter(
            failures_handle,
            fieldnames=["source_index", "url", "error"],
            delimiter="\t",
        )
        if not manifest_exists:
            manifest_writer.writeheader()
        if not failures_exists:
            failure_writer.writeheader()

        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            futures = {
                executor.submit(
                    fetch_image,
                    item,
                    args.output,
                    args.timeout,
                    args.retries,
                    args.max_bytes,
                ): item
                for item in batch
            }
            for future in as_completed(futures):
                item = futures[future]
                result, error = future.result()
                if result is not None:
                    manifest_writer.writerow(result)
                    success_count += 1
                else:
                    failure_writer.writerow(
                        {"source_index": item[0], "url": item[1], "error": error}
                    )
            manifest_handle.flush()
            failures_handle.flush()
            print(f"valid images: {success_count}/{args.target}", flush=True)
            if success_count >= args.target:
                return 0

    print(f"only downloaded {success_count}/{args.target} valid images")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
