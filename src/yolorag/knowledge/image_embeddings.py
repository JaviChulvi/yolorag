from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


# Mirrors the portal worker's Nomic vision path (apps/alpha/workers/worker.py:4457-4509)
# so backend image embeddings match production byte-for-byte.
DEFAULT_NOMIC_ONNX_MODEL = "/opt/models/nomic-embed-vision-v1.5/onnx/model_int8.onnx"
DEFAULT_NOMIC_EMBEDDING_MODEL = "nomic-embed-vision-v1.5"
DEFAULT_NOMIC_EMBEDDING_DIMENSIONS = 128  # Matryoshka truncation from 768-dim to 128-dim
DEFAULT_NOMIC_IMAGE_SIZE = 224
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# Nomic's CLIPImageProcessor normalization constants.
NOMIC_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
NOMIC_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageEmbeddingClient(Protocol):
    provider_name: str
    model: str
    dimensions: int

    def embed_images(self, images: Sequence[bytes]) -> list[list[float]]:
        """Return one embedding per input image (raw bytes), preserving input order."""


@dataclass(frozen=True)
class NomicImageEmbeddingConfig:
    onnx_model_path: str = DEFAULT_NOMIC_ONNX_MODEL
    model: str = DEFAULT_NOMIC_EMBEDDING_MODEL
    dimensions: int = DEFAULT_NOMIC_EMBEDDING_DIMENSIONS
    image_size: int = DEFAULT_NOMIC_IMAGE_SIZE
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    # 0 => leave onnxruntime's default thread count untouched.
    intra_op_num_threads: int = 0

    @classmethod
    def from_env(cls) -> NomicImageEmbeddingConfig:
        return cls(
            onnx_model_path=os.getenv("NOMIC_ONNX_MODEL", DEFAULT_NOMIC_ONNX_MODEL),
            model=os.getenv(
                "YOLORAG_IMAGE_EMBEDDING_MODEL",
                DEFAULT_NOMIC_EMBEDDING_MODEL,
            ),
            dimensions=_env_int(
                "YOLORAG_IMAGE_EMBEDDING_DIMENSIONS",
                DEFAULT_NOMIC_EMBEDDING_DIMENSIONS,
            ),
            image_size=_env_int(
                "YOLORAG_NOMIC_IMAGE_SIZE",
                DEFAULT_NOMIC_IMAGE_SIZE,
            ),
            batch_size=_env_int(
                "YOLORAG_IMAGE_EMBEDDING_BATCH_SIZE",
                DEFAULT_EMBEDDING_BATCH_SIZE,
            ),
            intra_op_num_threads=_env_int(
                "YOLORAG_NOMIC_INTRA_OP_THREADS",
                0,
                minimum=0,
            ),
        )


class NomicImageEmbeddingClient:
    """Embed images with the pinned Nomic vision ONNX int8 model.

    Replicates the portal worker's `_load_nomic_session`, `_preprocess_nomic_image`,
    and `_embed_nomic_images` so embeddings match production. The session is loaded
    lazily and cached behind a lock so it is built at most once per client.
    """

    provider_name = "nomic-vision"

    def __init__(
        self,
        config: NomicImageEmbeddingConfig,
        session: object | None = None,
    ) -> None:
        if config.dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than 0.")
        if config.batch_size <= 0:
            raise ValueError("Embedding batch size must be greater than 0.")
        if config.image_size <= 0:
            raise ValueError("Image size must be greater than 0.")
        self.config = config
        self.model = config.model
        self.dimensions = config.dimensions
        self._mean = np.array(NOMIC_IMAGE_MEAN, dtype=np.float32)
        self._std = np.array(NOMIC_IMAGE_STD, dtype=np.float32)
        self._session = session
        self._session_lock = threading.Lock()

    def _load_session(self) -> object:
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    model_path = Path(self.config.onnx_model_path)
                    if not model_path.exists():
                        raise RuntimeError(
                            f"Nomic ONNX model not found at {model_path}. "
                            "Set NOMIC_ONNX_MODEL to a valid model_int8.onnx path."
                        )

                    import onnxruntime as ort  # type: ignore

                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                    if self.config.intra_op_num_threads > 0:
                        session_options.intra_op_num_threads = (
                            self.config.intra_op_num_threads
                        )
                    session_options.inter_op_num_threads = 1
                    self._session = ort.InferenceSession(
                        str(model_path),
                        sess_options=session_options,
                        providers=["CPUExecutionProvider"],
                    )
        return self._session

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        import cv2  # type: ignore  # lazy: only needed for image preprocessing

        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("image decode returned None")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Nomic's CLIPImageProcessor uses bicubic resampling; OpenCV keeps this fast without PIL.
        resized = cv2.resize(
            img,
            (self.config.image_size, self.config.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        normalized = (resized.astype(np.float32) / 255.0 - self._mean) / self._std
        return normalized.transpose(2, 0, 1)

    def _embed_pixel_values(self, pixel_values: np.ndarray) -> np.ndarray:
        session = self._load_session()
        last_hidden_state = session.run(
            ["last_hidden_state"], {"pixel_values": pixel_values}
        )[0]
        # CLS token -> layernorm over the full hidden dim -> Matryoshka truncation -> L2 normalize.
        cls = last_hidden_state[:, 0, :].astype(np.float32)
        cls = (cls - cls.mean(axis=1, keepdims=True)) / np.sqrt(
            cls.var(axis=1, keepdims=True) + 1e-5
        )
        cls = cls[:, : self.dimensions]
        return cls / np.maximum(np.linalg.norm(cls, axis=1, keepdims=True), 1e-12)

    def embed_images(self, images: Sequence[bytes]) -> list[list[float]]:
        if not images:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(images), self.config.batch_size):
            batch = list(images[start : start + self.config.batch_size])
            pixel_values = np.stack([self._preprocess(item) for item in batch]).astype(
                np.float32
            )
            embeddings.extend(self._embed_pixel_values(pixel_values).tolist())
        return embeddings


def pack_float32(embedding: Sequence[float]) -> bytes:
    """Pack an embedding as little-endian float32 bytes (the `image_embeddings.e` payload)."""
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}.")
    return parsed
