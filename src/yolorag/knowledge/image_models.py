from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ImageEmbeddingRecord:
    """A single image embedding to persist: composite key plus the vector."""

    dataset_id: str
    img_id: str
    embedding: Sequence[float]


@dataclass(frozen=True)
class ImageSearchResult:
    """A vector-search hit, vendor-agnostic across image embedding stores."""

    dataset_id: str
    img_id: str
    score: float | None
    provider: str = "unknown"
    query_embedding_ms: int = 0
