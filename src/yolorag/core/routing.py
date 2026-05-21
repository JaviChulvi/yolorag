from __future__ import annotations

from dataclasses import dataclass

from yolorag.providers.base import ResponseMode


@dataclass(frozen=True)
class OrchestrationPlan:
    mode: ResponseMode
    should_retrieve: bool
    should_review: bool
    top_k: int
    reasoning_budget: str
    reason: str


class SimpleRoutePlanner:
    domain_keywords = {
        "ultralytics",
        "yolo",
        "training",
        "dataset",
        "export",
        "inference",
        "model",
        "github",
        "issue",
        "error",
        "debug",
    }

    generic_starters = {
        "hello",
        "hi",
        "thanks",
        "thank you",
        "who are you",
        "what can you do",
    }

    def plan(self, user_message: str, requested_mode: ResponseMode) -> OrchestrationPlan:
        lowered = user_message.lower().strip()
        looks_generic = lowered in self.generic_starters
        looks_domain_specific = any(keyword in lowered for keyword in self.domain_keywords)

        if requested_mode == "deep":
            return OrchestrationPlan(
                mode="deep",
                should_retrieve=looks_domain_specific,
                should_review=True,
                top_k=4,
                reasoning_budget="high",
                reason=(
                    "Deep mode requested; retrieval is enabled only because the "
                    "question appears domain-specific."
                    if looks_domain_specific
                    else "Deep mode requested, but retrieval skipped because the question is generic."
                ),
            )

        return OrchestrationPlan(
            mode="fast",
            should_retrieve=looks_domain_specific and not looks_generic,
            should_review=False,
            top_k=2,
            reasoning_budget="low",
            reason=(
                "Fast mode with targeted retrieval because the question appears domain-specific."
                if looks_domain_specific and not looks_generic
                else "Fast mode skipped retrieval to minimize latency and context noise."
            ),
        )

