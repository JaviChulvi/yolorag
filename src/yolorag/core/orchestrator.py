from __future__ import annotations

from dataclasses import dataclass

from yolorag.core.conversation import (
    ConversationTurn,
    InMemoryConversationStore,
)
from yolorag.core.routing import SimpleRoutePlanner
from yolorag.core.tracing import OrchestrationTrace
from yolorag.providers.base import LLMProvider, LLMRequest, ResponseMode
from yolorag.retrieval.base import RetrievalResult, Retriever
from yolorag.review.base import ReviewResult, Reviewer


@dataclass(frozen=True)
class OrchestratorResult:
    answer: str
    trace: OrchestrationTrace
    retrieved_context: list[RetrievalResult]
    review: ReviewResult | None


class RAGOrchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        retriever: Retriever | None = None,
        reviewer: Reviewer | None = None,
        conversation_store: InMemoryConversationStore | None = None,
        route_planner: SimpleRoutePlanner | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.retriever = retriever
        self.reviewer = reviewer
        self.conversation_store = conversation_store or InMemoryConversationStore()
        self.route_planner = route_planner or SimpleRoutePlanner()

    async def answer(
        self,
        user_message: str,
        conversation_id: str = "default",
        mode: ResponseMode = "fast",
    ) -> OrchestratorResult:
        state = self.conversation_store.get(conversation_id)
        plan = self.route_planner.plan(user_message=user_message, requested_mode=mode)
        retrieved_context = await self._retrieve_if_needed(
            user_message=user_message,
            existing_document_ids=state.retrieved_document_ids,
            should_retrieve=plan.should_retrieve,
            top_k=plan.top_k,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a provider-agnostic RAG assistant. "
                    f"Mode={plan.mode}; reasoning_budget={plan.reasoning_budget}. "
                    "Use retrieved context only when it is relevant."
                ),
            },
            *state.recent_messages(),
        ]
        if retrieved_context:
            messages.append(
                {
                    "role": "system",
                    "content": self._format_context(retrieved_context),
                }
            )
        messages.append({"role": "user", "content": user_message})

        response = await self.provider.complete(
            LLMRequest(
                messages=messages,
                model=self.model,
                mode=plan.mode,
                max_tokens=512 if plan.mode == "fast" else 1600,
            )
        )

        review = None
        if plan.should_review and self.reviewer:
            review = await self.reviewer.review(
                question=user_message,
                answer=response.content,
                context=retrieved_context,
            )

        retrieved_ids = [item.document.id for item in retrieved_context]
        state.add_turn(
            ConversationTurn(
                user_message=user_message,
                assistant_message=response.content,
                retrieved_document_ids=retrieved_ids,
            )
        )

        trace = OrchestrationTrace(
            provider=response.provider,
            model=response.model,
            mode=plan.mode,
            reasoning_budget=plan.reasoning_budget,
            retrieval_used=bool(retrieved_context),
            retrieved_document_ids=retrieved_ids,
            review_used=review is not None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            estimated_cost_usd=response.cost.total_usd,
            pricing_source=response.cost.pricing_source,
            latency_ms=response.latency_ms,
            route_reason=plan.reason,
            review_confidence=review.confidence if review else None,
            review_notes=review.notes if review else [],
        )

        return OrchestratorResult(
            answer=response.content,
            trace=trace,
            retrieved_context=retrieved_context,
            review=review,
        )

    async def _retrieve_if_needed(
        self,
        user_message: str,
        existing_document_ids: set[str],
        should_retrieve: bool,
        top_k: int,
    ) -> list[RetrievalResult]:
        if not should_retrieve or self.retriever is None:
            return []

        results = await self.retriever.retrieve(user_message, top_k=top_k)
        return [
            result
            for result in results
            if result.document.id not in existing_document_ids
        ]

    def _format_context(self, context: list[RetrievalResult]) -> str:
        blocks = []
        for item in context:
            blocks.append(
                "\n".join(
                    [
                        f"Document ID: {item.document.id}",
                        f"Title: {item.document.title}",
                        f"Selection reason: {item.reason}",
                        item.document.content,
                    ]
                )
            )
        return "Relevant retrieved context:\n\n" + "\n\n---\n\n".join(blocks)

