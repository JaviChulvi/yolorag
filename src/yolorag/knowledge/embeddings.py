from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, Sequence

from openai import OpenAI


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_OPENAI_EMBEDDING_DIMENSIONS = 3072
DEFAULT_EMBEDDING_BATCH_SIZE = 64


class EmbeddingClient(Protocol):
    provider_name: str
    model: str
    dimensions: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text, preserving input order."""


@dataclass(frozen=True)
class OpenAIEmbeddingConfig:
    api_key: str
    model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    dimensions: int = DEFAULT_OPENAI_EMBEDDING_DIMENSIONS
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE

    @classmethod
    def from_env(cls) -> OpenAIEmbeddingConfig:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for OpenAI embeddings.")
        return cls(
            api_key=api_key,
            model=os.getenv(
                "YOLORAG_POSTGRES_EMBEDDING_MODEL",
                DEFAULT_OPENAI_EMBEDDING_MODEL,
            ),
            dimensions=_env_int(
                "YOLORAG_POSTGRES_EMBEDDING_DIMENSIONS",
                DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
            ),
            batch_size=_env_int(
                "YOLORAG_EMBEDDING_BATCH_SIZE",
                DEFAULT_EMBEDDING_BATCH_SIZE,
            ),
        )


class OpenAIEmbeddingClient:
    provider_name = "openai"

    def __init__(
        self,
        config: OpenAIEmbeddingConfig,
        client: OpenAI | None = None,
    ) -> None:
        if config.dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than 0.")
        if config.batch_size <= 0:
            raise ValueError("Embedding batch size must be greater than 0.")
        self.config = config
        self.model = config.model
        self.dimensions = config.dimensions
        self.client = client or OpenAI(api_key=config.api_key)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = list(texts[start : start + self.config.batch_size])
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend([list(item.embedding) for item in ordered])
        return embeddings


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return parsed
