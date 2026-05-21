from __future__ import annotations

from dataclasses import dataclass

from yolorag.config.model_defaults import default_model_for
from yolorag.config.settings import getenv
from yolorag.core.orchestrator import OrchestratorResult, RAGOrchestrator
from yolorag.providers.base import ResponseMode
from yolorag.providers.deepseek_provider import DeepSeekProvider
from yolorag.providers.openai_provider import OpenAIProvider
from yolorag.retrieval.base import Document
from yolorag.retrieval.in_memory import InMemoryRetriever
from yolorag.review.simple_reviewer import SimpleReviewer


@dataclass
class YoloRAGRuntime:
    orchestrator: RAGOrchestrator
    mode: ResponseMode = "fast"

    async def answer(self, user_message: str, conversation_id: str) -> OrchestratorResult:
        return await self.orchestrator.answer(
            user_message=user_message,
            conversation_id=conversation_id,
            mode=self.mode,
        )

    def message_counts(self, conversation_id: str) -> tuple[int, int]:
        state = self.orchestrator.conversation_store.get(conversation_id)
        total = len(state.turns)
        return total, total


def build_runtime(
    provider_name: str | None = None,
    mode: ResponseMode | None = None,
    api_base: str | None = None,
) -> YoloRAGRuntime:
    selected_provider = provider_name or getenv("YOLORAG_API_PROVIDER", "openai")
    selected_mode = _resolve_mode(mode or getenv("YOLORAG_API_MODE", "fast"))
    selected_model = _resolve_model(
        provider_name=selected_provider,
        mode=selected_mode,
    )
    provider = _build_provider(provider_name=selected_provider, api_base=api_base)

    return YoloRAGRuntime(
        orchestrator=RAGOrchestrator(
            provider=provider,
            model=selected_model,
            retriever=InMemoryRetriever(_sample_documents()),
            reviewer=SimpleReviewer(),
        ),
        mode=selected_mode,
    )


def _build_provider(provider_name: str, api_base: str | None) -> OpenAIProvider | DeepSeekProvider:
    if provider_name == "openai":
        return OpenAIProvider(
            api_key=_require_env("OPENAI_API_KEY"),
            api_base=api_base or getenv("OPENAI_BASE_URL"),
        )
    if provider_name == "deepseek":
        return DeepSeekProvider(
            api_key=_require_env("DEEPSEEK_API_KEY"),
            api_base=api_base or getenv("DEEPSEEK_BASE_URL"),
        )
    raise ValueError(f"Unsupported provider {provider_name!r}.")


def _resolve_model(
    provider_name: str,
    mode: ResponseMode,
) -> str:
    provider_key = provider_name.upper()
    mode_key = "THINKING" if mode == "deep" else "FAST"
    mode_env_name = f"YOLORAG_{provider_key}_{mode_key}_MODEL"
    legacy_env_name = f"YOLORAG_{provider_key}_MODEL"

    configured_model = getenv(mode_env_name) or getenv(legacy_env_name)
    if configured_model:
        return configured_model

    return default_model_for(
        provider_name=provider_name,
        mode=mode,
    )


def _resolve_mode(mode: str | ResponseMode) -> ResponseMode:
    if mode in {"fast", "deep"}:
        return mode
    raise ValueError(f"Unsupported response mode {mode!r}.")


def _require_env(name: str) -> str:
    value = getenv(name)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable {name}.")


def _sample_documents() -> list[Document]:
    return [
        Document(
            id="ultralytics-yolo-export",
            title="Ultralytics YOLO Export",
            content=(
                "Ultralytics YOLO models can be exported to deployment formats. "
                "Export and inference troubleshooting often benefits from focused "
                "retrieval because exact framework versions and target formats matter."
            ),
            metadata={"url": "https://docs.ultralytics.com/modes/export/"},
        ),
        Document(
            id="rag-selective-retrieval",
            title="Selective Retrieval Guidance",
            content=(
                "Retrieval should be skipped for generic chat and used when external "
                "knowledge materially improves answer quality. Repeated turns should "
                "avoid reinjecting the same documents."
            ),
            metadata={"url": "https://docs.ultralytics.com/"},
        ),
    ]
