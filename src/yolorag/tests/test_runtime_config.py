from __future__ import annotations

import unittest
from unittest.mock import patch

from yolorag.providers.base import LLMResponse
from yolorag.runtime import (
    DEFAULT_FAST_RERANK_CANDIDATE_LIMIT,
    _build_reranker,
    build_runtime,
)
from yolorag.usage.models import CostBreakdown, TokenUsage


class RuntimeConfigTests(unittest.TestCase):
    def test_reranker_requires_credentials(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                _build_reranker()

    def test_reranker_builds_from_credentials(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "YOLORAG_MONGODB_AI_API_KEY": "test-rerank-key",
                "YOLORAG_RERANK_MODEL": "rerank-test-model",
            },
            clear=True,
        ):
            reranker = _build_reranker()

        self.assertEqual(reranker.api_key, "test-rerank-key")
        self.assertEqual(reranker.model, "rerank-test-model")

    def test_fast_runtime_uses_fast_tool_timeout_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "YOLORAG_MONGODB_AI_API_KEY": "test-rerank-key",
                "YOLORAG_FAST_TOOL_TIMEOUT_SECONDS": "9.5",
            },
            clear=True,
        ):
            with patch("yolorag.runtime.get_llm_provider", return_value=FakeProvider()):
                with patch("yolorag.runtime.build_knowledge_store", return_value=FakeStore()):
                    with patch("yolorag.runtime.build_conversation_logger", return_value=None):
                        runtime = build_runtime(provider_name="deepseek", mode="fast")

        self.assertEqual(runtime.orchestrator.fast_tool_timeout_seconds, 9.5)

    def test_fast_runtime_uses_smaller_default_rerank_candidate_limit(self) -> None:
        with patch.dict(
            "os.environ",
            {"YOLORAG_MONGODB_AI_API_KEY": "test-rerank-key"},
            clear=True,
        ):
            with patch("yolorag.runtime.get_llm_provider", return_value=FakeProvider()):
                with patch("yolorag.runtime.build_knowledge_store", return_value=FakeStore()):
                    with patch("yolorag.runtime.build_conversation_logger", return_value=None):
                        runtime = build_runtime(provider_name="deepseek", mode="fast")

        assert runtime.orchestrator.retriever is not None
        self.assertEqual(
            runtime.orchestrator.retriever.candidate_limit,
            DEFAULT_FAST_RERANK_CANDIDATE_LIMIT,
        )

    def test_shared_rerank_candidate_limit_overrides_fast_default(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "YOLORAG_MONGODB_AI_API_KEY": "test-rerank-key",
                "YOLORAG_RERANK_CANDIDATE_LIMIT": "24",
            },
            clear=True,
        ):
            with patch("yolorag.runtime.get_llm_provider", return_value=FakeProvider()):
                with patch("yolorag.runtime.build_knowledge_store", return_value=FakeStore()):
                    with patch("yolorag.runtime.build_conversation_logger", return_value=None):
                        runtime = build_runtime(provider_name="deepseek", mode="fast")

        assert runtime.orchestrator.retriever is not None
        self.assertEqual(runtime.orchestrator.retriever.candidate_limit, 24)


class FakeProvider:
    provider_name = "test"

    async def complete(self, request):
        return LLMResponse(
            content="ok",
            provider=self.provider_name,
            model=request.model,
            usage=TokenUsage(),
            cost=CostBreakdown(),
            latency_ms=1,
            raw_response={},
        )

    async def stream_complete(self, request):
        if False:
            yield None


class FakeStore:
    provider_name = "fake"


if __name__ == "__main__":
    unittest.main()
