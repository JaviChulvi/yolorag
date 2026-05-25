from __future__ import annotations

from dataclasses import dataclass

from yolorag.providers.base import ResponseMode


@dataclass(frozen=True)
class OrchestrationPlan:
    mode: ResponseMode
    should_retrieve: bool
    top_k: int
    min_relevance_score: float | None
    reason: str


class SimpleRoutePlanner:
    def __init__(
        self,
        *,
        fast_top_k: int = 2,
        deep_top_k: int = 4,
        min_relevance_score: float | None = 0.5,
    ) -> None:
        self.fast_top_k = fast_top_k
        self.deep_top_k = deep_top_k
        self.min_relevance_score = min_relevance_score

    def plan(self, user_message: str, requested_mode: ResponseMode) -> OrchestrationPlan:
        should_retrieve = bool(user_message.strip())

        if requested_mode == "deep":
            return OrchestrationPlan(
                mode="deep",
                should_retrieve=should_retrieve,
                top_k=self.deep_top_k,
                min_relevance_score=self.min_relevance_score,
                reason="Deep mode uses retrieval when configured and filters low-confidence context.",
            )

        return OrchestrationPlan(
            mode="fast",
            should_retrieve=should_retrieve,
            top_k=self.fast_top_k,
            min_relevance_score=self.min_relevance_score,
            reason="Fast mode uses lightweight retrieval and filters low-confidence context.",
        )
