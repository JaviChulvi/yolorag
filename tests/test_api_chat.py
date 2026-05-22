from __future__ import annotations

import unittest
from asyncio import run
from decimal import Decimal

from fastapi.testclient import TestClient

from yolorag.api.app import create_app
from yolorag.core.orchestrator import RAGOrchestrator
from yolorag.providers.base import LLMRequest, LLMResponse, LLMStreamEvent
from yolorag.retrieval.base import Document, RetrievalResult
from yolorag.runtime import YoloRAGRuntime
from yolorag.usage.models import CostBreakdown, TokenUsage


class RecordingProvider:
    provider_name = "test"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_calls += 1
        self.requests.append(request)
        user_message = request.messages[-1]["content"]
        return LLMResponse(
            content=f"Echo: {user_message}",
            provider=self.provider_name,
            model=request.model,
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            cost=CostBreakdown(total_usd=Decimal("0.000001"), pricing_source="test"),
            latency_ms=1,
            raw_response={"test": True},
        )

    async def stream_complete(self, request: LLMRequest):
        self.stream_calls += 1
        self.requests.append(request)
        user_message = request.messages[-1]["content"]
        yield LLMStreamEvent(content="Echo: ")
        yield LLMStreamEvent(content=user_message)


class ChatApiTests(unittest.TestCase):
    def test_chat_streams_sse_and_returns_session_headers(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app(runtime=runtime))

        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertTrue(response.headers["X-Session-ID"])
        self.assertEqual(response.headers["X-Total-User-Messages"], "1")
        self.assertEqual(provider.stream_calls, 1)
        self.assertEqual(provider.complete_calls, 0)
        self.assertIn('data: {"content": "Echo: "}', response.text)
        self.assertIn('data: {"content": "hello"}', response.text)
        self.assertIn("data: [DONE]", response.text)

    def test_chat_reuses_session_history_for_follow_up_turns(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app(runtime=runtime))

        first = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "first"}]},
        )
        session_id = first.headers["X-Session-ID"]
        second = client.post(
            "/api/chat",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "second"}],
            },
        )

        self.assertEqual(second.headers["X-Session-ID"], session_id)
        self.assertEqual(second.headers["X-Total-User-Messages"], "2")
        second_request_messages = provider.requests[1].messages
        self.assertIn({"role": "user", "content": "first"}, second_request_messages)
        self.assertIn({"role": "assistant", "content": "Echo: first"}, second_request_messages)
        self.assertEqual(second_request_messages[-1], {"role": "user", "content": "second"})

    def test_chat_includes_page_context_when_present(self) -> None:
        provider = RecordingProvider()
        runtime = YoloRAGRuntime(
            orchestrator=RAGOrchestrator(provider=provider, model="test-model")
        )
        client = TestClient(create_app(runtime=runtime))

        client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "what page is this?"}],
                "context": {
                    "title": "YOLO Export",
                    "url": "https://docs.ultralytics.com/modes/export/",
                    "path": "/modes/export/",
                    "description": "Export docs",
                },
            },
        )

        sent_message = provider.requests[0].messages[-1]["content"]
        self.assertIn("Page context:", sent_message)
        self.assertIn("Title: YOLO Export", sent_message)
        self.assertIn("User message:\nwhat page is this?", sent_message)


class StaticRetriever:
    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                document=Document(
                    id="doc-1",
                    title="Training Docs",
                    content="Use model.train(data='coco8.yaml', epochs=100).",
                ),
                score=0.9,
                reason=f"Test retriever top_k={top_k}",
            )
        ]


class OrchestratorRetrievalTests(unittest.TestCase):
    def test_main_prompt_does_not_expose_internal_mode_or_budget(self) -> None:
        provider = RecordingProvider()
        orchestrator = RAGOrchestrator(provider=provider, model="test-model")

        run(orchestrator.answer("hello", mode="deep"))

        main_prompt = provider.requests[0].messages[0]["content"]
        self.assertIn("YoloRAG", main_prompt)
        self.assertIn("Ultralytics YOLO documentation", main_prompt)
        self.assertNotIn("Mode=", main_prompt)
        self.assertNotIn("reasoning_budget", main_prompt)
        self.assertEqual(provider.requests[0].mode, "deep")

    def test_forced_retrieval_adds_context_to_llm_request(self) -> None:
        provider = RecordingProvider()
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=StaticRetriever(),
            force_retrieval=True,
            retrieval_top_k=5,
        )

        result = run(orchestrator.answer("hello"))

        sent_messages = provider.requests[0].messages
        context_messages = [
            message["content"]
            for message in sent_messages
            if message["role"] == "system" and "Relevant retrieved context" in message["content"]
        ]
        self.assertEqual(len(context_messages), 1)
        self.assertIn("Document ID: doc-1", context_messages[0])
        self.assertIn("Use model.train", context_messages[0])
        self.assertTrue(result.trace.retrieval_used)
        self.assertIn("Retrieval was forced", result.trace.route_reason)
        self.assertGreaterEqual(result.trace.total_ms, result.trace.llm_ms)
        self.assertGreaterEqual(result.trace.retrieval_ms, 0)
        self.assertEqual(result.trace.llm_ms, 1)

    def test_forced_retrieval_reinjects_context_on_follow_up_turns(self) -> None:
        provider = RecordingProvider()
        orchestrator = RAGOrchestrator(
            provider=provider,
            model="test-model",
            retriever=StaticRetriever(),
            force_retrieval=True,
            retrieval_top_k=5,
        )

        run(orchestrator.answer("hello", conversation_id="thread-1"))
        run(orchestrator.answer("thanks", conversation_id="thread-1"))

        second_request_messages = provider.requests[1].messages
        self.assertTrue(
            any(
                message["role"] == "system"
                and "Relevant retrieved context" in message["content"]
                for message in second_request_messages
            )
        )


if __name__ == "__main__":
    unittest.main()
