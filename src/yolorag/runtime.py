from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from yolorag.config.model_defaults import default_model_for
from yolorag.config.settings import getenv
from yolorag.core.agent import DeepAgentOrchestrator
from yolorag.core.conversation_factory import build_conversation_logger
from yolorag.core.orchestrator import OrchestratorResult, RAGOrchestrator
from yolorag.core.routing import SimpleRoutePlanner
from yolorag.knowledge.factory import build_knowledge_store
from yolorag.providers.base import Message, ResponseMode
from yolorag.providers.deepseek_provider import DeepSeekProvider
from yolorag.providers.openai_provider import OpenAIProvider
from yolorag.retrieval.base import Retriever
from yolorag.retrieval.mongodb import MongoReranker, MongoVectorRetriever
from yolorag.tools.factory import build_tool_router


DEFAULT_CHAT_VECTOR_TOP_K = 2
DEFAULT_RETRIEVAL_MIN_SCORE = 0.5
DEFAULT_DEEP_MAX_STEPS = 6
DEFAULT_DEEP_TOOL_TIMEOUT_SECONDS = 20.0


@dataclass
class YoloRAGRuntime:
    orchestrator: RAGOrchestrator
    mode: ResponseMode = "fast"

    async def answer(
        self,
        user_message: str,
        conversation_id: str,
        *,
        conversation_messages: list[Message] | None = None,
        raw_user_message: str | None = None,
        request_id: str | None = None,
        user_message_index: int | None = None,
    ) -> OrchestratorResult:
        return await self.orchestrator.answer(
            user_message=user_message,
            conversation_id=conversation_id,
            mode=self.mode,
            conversation_messages=conversation_messages,
            raw_user_message=raw_user_message,
            request_id=request_id,
            user_message_index=user_message_index,
        )

    def stream_answer(
        self,
        user_message: str,
        conversation_id: str,
        *,
        conversation_messages: list[Message] | None = None,
        raw_user_message: str | None = None,
        request_id: str | None = None,
        user_message_index: int | None = None,
    ) -> AsyncIterator[str]:
        return self.orchestrator.stream_answer(
            user_message=user_message,
            conversation_id=conversation_id,
            mode=self.mode,
            conversation_messages=conversation_messages,
            raw_user_message=raw_user_message,
            request_id=request_id,
            user_message_index=user_message_index,
        )


@dataclass
class YoloRAGAgentRuntime:
    orchestrator: DeepAgentOrchestrator

    def stream_answer(
        self,
        *,
        user_message: str,
        conversation_id: str,
        conversation_messages: list[Message] | None = None,
        raw_user_message: str | None = None,
        request_id: str | None = None,
        user_message_index: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        return self.orchestrator.stream_answer(
            user_message=user_message,
            conversation_id=conversation_id,
            conversation_messages=conversation_messages,
            raw_user_message=raw_user_message,
            request_id=request_id,
            user_message_index=user_message_index,
        )


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
    retriever = _build_retriever()

    return YoloRAGRuntime(
        orchestrator=RAGOrchestrator(
            provider=provider,
            model=selected_model,
            retriever=retriever,
            conversation_logger=build_conversation_logger(),
            route_planner=SimpleRoutePlanner(
                min_relevance_score=_env_float(
                    "YOLORAG_RETRIEVAL_MIN_SCORE",
                    default=DEFAULT_RETRIEVAL_MIN_SCORE,
                ),
            ),
            retrieval_top_k=_env_int("YOLORAG_CHAT_VECTOR_TOP_K", default=DEFAULT_CHAT_VECTOR_TOP_K),
        ),
        mode=selected_mode,
    )


def build_deep_runtime(
    provider_name: str | None = None,
    api_base: str | None = None,
) -> YoloRAGAgentRuntime:
    selected_provider = provider_name or getenv("YOLORAG_API_PROVIDER", "openai")
    selected_model = _resolve_model(selected_provider, "deep")
    provider = _build_provider(provider_name=selected_provider, api_base=api_base)
    retriever = _build_retriever()

    return YoloRAGAgentRuntime(
        orchestrator=DeepAgentOrchestrator(
            provider=provider,
            model=selected_model,
            tool_router=build_tool_router(
                retriever=retriever,
                min_relevance_score=_env_float(
                    "YOLORAG_RETRIEVAL_MIN_SCORE",
                    default=DEFAULT_RETRIEVAL_MIN_SCORE,
                ),
            ),
            conversation_logger=build_conversation_logger(),
            max_steps=_env_int("YOLORAG_DEEP_MAX_STEPS", default=DEFAULT_DEEP_MAX_STEPS),
            tool_timeout_seconds=_env_float(
                "YOLORAG_DEEP_TOOL_TIMEOUT_SECONDS",
                default=DEFAULT_DEEP_TOOL_TIMEOUT_SECONDS,
            ),
        )
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


def _build_retriever() -> Retriever | None:
    store = build_knowledge_store()
    return MongoVectorRetriever(
        store=store,
        reranker=MongoReranker.from_env(),
        candidate_limit=_env_int_or_none("YOLORAG_RERANK_CANDIDATE_LIMIT"),
    )


def _env_int(name: str, default: int) -> int:
    value = getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return parsed


def _env_float(name: str, default: float) -> float:
    value = getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return parsed


def _env_int_or_none(name: str) -> int | None:
    value = getenv(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return parsed
