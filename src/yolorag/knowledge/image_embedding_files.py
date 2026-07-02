"""Vendor-neutral on-disk format for image embeddings.

Embedding generation (the expensive Nomic ONNX pass) runs once and writes these
files; DB ingestion replays them into whatever store you are evaluating. The
format is deliberately generic -- newline-delimited JSON plus a small JSON
sidecar -- so it carries no dependency on numpy, SQLAlchemy, or any DB driver.

Layout, one pair per dataset::

    <out_dir>/<dataset_id>.jsonl        # one {dataset_id, img_id, embedding} per line
    <out_dir>/<dataset_id>.meta.json    # {dataset_id, model, dimensions, count, source, created_at}

Each JSONL line is a self-contained record::

    {"dataset_id": "corn-leaf", "img_id": "2bcf9074...", "embedding": [0.01, -0.02, ...]}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

EMBEDDINGS_SUFFIX = ".jsonl"
META_SUFFIX = ".meta.json"


@dataclass(frozen=True)
class EmbeddingFileMeta:
    dataset_id: str
    model: str
    dimensions: int
    count: int
    source: str
    created_at: str


def dataset_paths(out_dir: str | Path, dataset_id: str) -> tuple[Path, Path]:
    """Return the (jsonl, meta) paths for a dataset under ``out_dir``."""
    base = Path(out_dir)
    return (
        base / f"{dataset_id}{EMBEDDINGS_SUFFIX}",
        base / f"{dataset_id}{META_SUFFIX}",
    )


class EmbeddingsWriter:
    """Stream embedding records to ``<dataset_id>.jsonl`` and finalize the meta sidecar.

    Used as a context manager so the meta file (with the final count) is only
    written on clean exit -- a crashed run leaves a partial ``.jsonl`` with no
    ``.meta.json``, which is easy to detect and re-run.
    """

    def __init__(
        self,
        out_dir: str | Path,
        dataset_id: str,
        *,
        model: str,
        dimensions: int,
        source: str,
    ) -> None:
        self.dataset_id = dataset_id
        self.model = model
        self.dimensions = dimensions
        self.source = source
        self.jsonl_path, self.meta_path = dataset_paths(out_dir, dataset_id)
        self.count = 0
        self._handle = None

    def __enter__(self) -> "EmbeddingsWriter":
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.jsonl_path.open("w", encoding="utf-8")
        return self

    def add(self, img_id: str, embedding: Sequence[float]) -> None:
        if self._handle is None:
            raise RuntimeError("EmbeddingsWriter used outside its context manager.")
        row = {
            "dataset_id": self.dataset_id,
            "img_id": img_id,
            "embedding": [float(value) for value in embedding],
        }
        self._handle.write(json.dumps(row, separators=(",", ":")))
        self._handle.write("\n")
        self.count += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if exc_type is None:
            write_meta(
                self.meta_path,
                EmbeddingFileMeta(
                    dataset_id=self.dataset_id,
                    model=self.model,
                    dimensions=self.dimensions,
                    count=self.count,
                    source=self.source,
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
            )


def write_meta(meta_path: str | Path, meta: EmbeddingFileMeta) -> None:
    path = Path(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(meta), handle, indent=2)
        handle.write("\n")


def read_embeddings(jsonl_path: str | Path) -> Iterator[dict]:
    """Yield ``{dataset_id, img_id, embedding}`` records from a ``.jsonl`` file."""
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            missing = {"dataset_id", "img_id", "embedding"} - record.keys()
            if missing:
                raise ValueError(
                    f"{jsonl_path}:{line_no} missing required keys: {sorted(missing)}"
                )
            yield record


def read_meta(meta_path: str | Path) -> dict:
    with Path(meta_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_embedding_files(inputs: Sequence[str | Path]) -> list[Path]:
    """Expand files/directories into a sorted, de-duplicated list of ``.jsonl`` files."""
    files: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            files.extend(sorted(path.glob(f"*{EMBEDDINGS_SUFFIX}")))
        elif path.is_file():
            files.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique
