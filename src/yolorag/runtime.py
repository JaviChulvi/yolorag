from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from yolorag.config.model_defaults import default_model_for
from yolorag.config.settings import getenv
from yolorag.core.agent import DeepAgentOrchestrator
from yolorag.core.conversation_factory import build_conversation_logger
from yolorag.core.orchestrator import (
    FAST_DOCS_TOP_K,
    FAST_TOOL_TIMEOUT_SECONDS,
    OrchestratorResult,
    RAGOrchestrator,
)
from yolorag.core.routing import SimpleRoutePlanner
from yolorag.knowledge.factory import build_knowledge_store
from yolorag.providers.base import Message, ResponseMode
from yolorag.providers.factory import get_llm_provider
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
        measure_ttft: bool = False,
        persist: bool = True,
    ) -> OrchestratorResult:
        return await self.orchestrator.answer(
            user_message=user_message,
            conversation_id=conversation_id,
            mode=self.mode,
            conversation_messages=conversation_messages,
            raw_user_message=raw_user_message,
            request_id=request_id,
            user_message_index=user_message_index,
            measure_ttft=measure_ttft,
            persist=persist,
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
        include_metrics: bool = False,
        persist: bool = True,
    ) -> AsyncIterator[str | dict[str, Any]]:
        return self.orchestrator.stream_answer(
            user_message=user_message,
            conversation_id=conversation_id,
            mode=self.mode,
            conversation_messages=conversation_messages,
            raw_user_message=raw_user_message,
            request_id=request_id,
            user_message_index=user_message_index,
            include_metrics=include_metrics,
            persist=persist,
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
    knowledge_provider: str | None = None,
    conversation_provider: str | None = None,
) -> YoloRAGRuntime:
    selected_provider = provider_name or getenv("YOLORAG_API_PROVIDER", "deepseek")
    selected_mode = _resolve_mode(mode or getenv("YOLORAG_API_MODE", "fast"))
    selected_model = _resolve_model(
        provider_name=selected_provider,
        mode=selected_mode,
    )
    provider = get_llm_provider(provider_name=selected_provider, api_base=api_base)
    retriever = _build_retriever(knowledge_provider=knowledge_provider)

    return YoloRAGRuntime(
        orchestrator=RAGOrchestrator(
            provider=provider,
            model=selected_model,
            retriever=retriever,
            conversation_logger=build_conversation_logger(
                conversation_provider,
                knowledge_provider=knowledge_provider,
            ),
            route_planner=SimpleRoutePlanner(
                min_relevance_score=_env_float(
                    "YOLORAG_RETRIEVAL_MIN_SCORE",
                    default=DEFAULT_RETRIEVAL_MIN_SCORE,
                ),
            ),
            retrieval_top_k=_env_int("YOLORAG_CHAT_VECTOR_TOP_K", default=DEFAULT_CHAT_VECTOR_TOP_K),
            tool_router=build_tool_router(
                retriever=retriever,
                min_relevance_score=_env_float(
                    "YOLORAG_RETRIEVAL_MIN_SCORE",
                    default=DEFAULT_RETRIEVAL_MIN_SCORE,
                ),
                docs_default_top_k=FAST_DOCS_TOP_K,
                docs_max_top_k=FAST_DOCS_TOP_K,
            ),
            fast_tool_timeout_seconds=FAST_TOOL_TIMEOUT_SECONDS,
        ),
        mode=selected_mode,
    )


def build_deep_runtime(
    provider_name: str | None = None,
    api_base: str | None = None,
    knowledge_provider: str | None = None,
    conversation_provider: str | None = None,
) -> YoloRAGAgentRuntime:
    selected_provider = provider_name or getenv("YOLORAG_API_PROVIDER", "deepseek")
    selected_model = _resolve_model(selected_provider, "deep")
    provider = get_llm_provider(provider_name=selected_provider, api_base=api_base)
    retriever = _build_retriever(knowledge_provider=knowledge_provider)

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
            conversation_logger=build_conversation_logger(
                conversation_provider,
                knowledge_provider=knowledge_provider,
            ),
            max_steps=_env_int("YOLORAG_DEEP_MAX_STEPS", default=DEFAULT_DEEP_MAX_STEPS),
            tool_timeout_seconds=_env_float(
                "YOLORAG_DEEP_TOOL_TIMEOUT_SECONDS",
                default=DEFAULT_DEEP_TOOL_TIMEOUT_SECONDS,
            ),
        )
    )


def _resolve_model(
    provider_name: str,
    mode: ResponseMode,
) -> str:
    normalized_provider = provider_name.lower().strip()
    provider_key = normalized_provider.upper()
    mode_key = "THINKING" if mode == "deep" else "FAST"
    mode_env_name = f"YOLORAG_{provider_key}_{mode_key}_MODEL"
    legacy_env_name = f"YOLORAG_{provider_key}_MODEL"

    configured_model = getenv(mode_env_name) or getenv(legacy_env_name)
    if configured_model:
        return configured_model

    return default_model_for(
        provider_name=normalized_provider,
        mode=mode,
    )


def _resolve_mode(mode: str | ResponseMode) -> ResponseMode:
    if mode in {"fast", "deep"}:
        return mode
    raise ValueError(f"Unsupported response mode {mode!r}.")


def _build_retriever(knowledge_provider: str | None = None) -> Retriever | None:
    store = build_knowledge_store(knowledge_provider)
    return MongoVectorRetriever(
        store=store,
        reranker=_build_reranker(),
        candidate_limit=_env_int_or_none("YOLORAG_RERANK_CANDIDATE_LIMIT"),
    )


def _build_reranker() -> MongoReranker:
    return MongoReranker.from_env()


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
