#!/usr/bin/env python3
"""Download dataset images from Ultralytics NDJSON export files.

Each ``.ndjson`` file describes one dataset: the first record is a ``dataset``
metadata line, the rest are ``image`` records carrying a temporary signed
``url`` and the original ``file`` name. This script reads those files and
downloads every image into an output directory named after the NDJSON file
(e.g. ``imgs/corn-leaf.ndjson`` -> ``<out>/corn-leaf/...``).

The signed URLs expire, so run this soon after exporting.

Examples:
    # Download every dataset under imgs/ into ./dataset_images/
    python -m yolorag.scripts.download_dataset_images

    # Specific files, split into train/val/test subfolders, 16 workers
    python -m yolorag.scripts.download_dataset_images imgs/crotales.ndjson \\
        --out /data/sets --by-split --workers 16
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Repo root is three levels up from src/yolorag/scripts/.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "imgs"
DEFAULT_OUTPUT = REPO_ROOT / "dataset_images"
USER_AGENT = "yolorag-dataset-downloader/1.0"


@dataclass
class ImageTask:
    """A single image to fetch and where to write it."""

    url: str
    dest: Path


def iter_ndjson_files(inputs: list[Path]) -> list[Path]:
    """Expand the input paths into a sorted list of .ndjson files."""
    files: list[Path] = []
    for item in inputs:
        item = item.expanduser()
        if item.is_dir():
            files.extend(sorted(item.glob("*.ndjson")))
        elif item.is_file():
            files.append(item)
        else:
            print(f"warning: input not found, skipping: {item}", file=sys.stderr)
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def safe_filename(name: str, fallback: str) -> str:
    """Reduce an arbitrary ``file`` field to a safe basename.

    Strips any directory components (guards against path traversal) and
    normalizes separators. Original spaces and unicode are preserved.
    """
    # Handle both POSIX and Windows separators embedded in the field.
    base = name.replace("\\", "/").split("/")[-1].strip()
    if not base or base in {".", ".."}:
        return fallback
    return base


def build_tasks(ndjson_path: Path, dataset_dir: Path, by_split: bool) -> list[ImageTask]:
    """Parse one NDJSON file into a list of image download tasks."""
    tasks: list[ImageTask] = []
    used_names: set[Path] = set()

    with ndjson_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"warning: {ndjson_path.name}:{line_no} invalid JSON, skipping ({exc})",
                    file=sys.stderr,
                )
                continue

            url = record.get("url")
            # Only image records carry a downloadable url + file; the leading
            # dataset record also has a `url` (a platform page), so require type.
            if record.get("type") != "image" or not url:
                continue

            filename = safe_filename(record.get("file", ""), f"image_{line_no}.jpg")
            target_dir = dataset_dir
            if by_split:
                split = str(record.get("split") or "unsplit")
                target_dir = dataset_dir / safe_filename(split, "unsplit")

            dest = target_dir / filename
            # Avoid clobbering when two records share a filename.
            if dest in used_names:
                stem, suffix = dest.stem, dest.suffix
                counter = 1
                while dest in used_names:
                    dest = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            used_names.add(dest)

            tasks.append(ImageTask(url=url, dest=dest))
    return tasks


def download_one(task: ImageTask, timeout: float, retries: int) -> tuple[bool, str]:
    """Download a single image. Returns (success, message)."""
    task.dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = task.dest.with_name(task.dest.name + ".part")
    request = urllib.request.Request(task.url, headers={"User-Agent": USER_AGENT})

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with tmp.open("wb") as out:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            tmp.replace(task.dest)
            return True, str(task.dest)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            tmp.unlink(missing_ok=True)
    return False, f"{task.dest.name} ({last_error})"


def download_dataset(
    ndjson_path: Path,
    out_root: Path,
    *,
    by_split: bool,
    workers: int,
    timeout: float,
    retries: int,
    overwrite: bool,
) -> tuple[int, int, int]:
    """Download all images for one NDJSON file.

    Returns (downloaded, skipped, failed) counts.
    """
    dataset_dir = out_root / ndjson_path.stem
    tasks = build_tasks(ndjson_path, dataset_dir, by_split)

    if not tasks:
        print(f"  {ndjson_path.name}: no image records found")
        return 0, 0, 0

    pending = tasks
    skipped = 0
    if not overwrite:
        pending = [t for t in tasks if not t.dest.exists()]
        skipped = len(tasks) - len(pending)

    print(
        f"  {ndjson_path.name}: {len(tasks)} images -> {dataset_dir}"
        + (f"  ({skipped} already present)" if skipped else "")
    )

    downloaded = 0
    failed = 0
    done = 0
    total = len(pending)
    lock = threading.Lock()

    def report(ok: bool, message: str) -> None:
        nonlocal downloaded, failed, done
        with lock:
            done += 1
            if ok:
                downloaded += 1
            else:
                failed += 1
                print(f"    [{done}/{total}] FAILED {message}", file=sys.stderr)
            # Lightweight progress heartbeat.
            if done == total or done % 10 == 0:
                print(f"    [{done}/{total}] downloaded={downloaded} failed={failed}")

    if total:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_one, task, timeout, retries): task for task in pending
            }
            for future in as_completed(futures):
                ok, message = future.result()
                report(ok, message)

    return downloaded, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download dataset images from Ultralytics NDJSON export files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(DEFAULT_INPUT)],
        help="NDJSON files and/or directories to scan for *.ndjson "
        f"(default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=str(DEFAULT_OUTPUT),
        help=f"Output base directory (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--by-split",
        action="store_true",
        help="Nest images under train/val/test subfolders per dataset.",
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=8, help="Concurrent downloads (default: 8)."
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Per-request timeout in seconds."
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Attempts per image (default: 3)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download images even if the file already exists.",
    )
    args = parser.parse_args()

    ndjson_files = iter_ndjson_files([Path(p) for p in args.inputs])
    if not ndjson_files:
        print("No .ndjson files found.", file=sys.stderr)
        return 1

    out_root = Path(args.out).expanduser()
    print(f"Found {len(ndjson_files)} NDJSON file(s). Output root: {out_root.resolve()}\n")

    total_downloaded = total_skipped = total_failed = 0
    for ndjson_path in ndjson_files:
        downloaded, skipped, failed = download_dataset(
            ndjson_path,
            out_root,
            by_split=args.by_split,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            overwrite=args.overwrite,
        )
        total_downloaded += downloaded
        total_skipped += skipped
        total_failed += failed

    print(
        f"\nDone. downloaded={total_downloaded} "
        f"skipped={total_skipped} failed={total_failed}"
    )
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
