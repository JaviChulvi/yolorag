from __future__ import annotations

import unittest
from decimal import Decimal

from fastapi.testclient import TestClient

from yolorag.api.app import create_app
from yolorag.core.orchestrator import RAGOrchestrator
from yolorag.providers.base import LLMRequest, LLMResponse
from yolorag.runtime import YoloRAGRuntime
from yolorag.usage.models import CostBreakdown, TokenUsage


class RecordingProvider:
    provider_name = "test"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
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
        self.assertIn('data: {"content": "Echo: hello"}', response.text)
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


if __name__ == "__main__":
    unittest.main()
