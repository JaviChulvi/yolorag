from __future__ import annotations

from yolorag.review.base import ReviewResult
from yolorag.retrieval.base import RetrievalResult


class SimpleReviewer:
    async def review(
        self,
        question: str,
        answer: str,
        context: list[RetrievalResult],
    ) -> ReviewResult:
        notes: list[str] = []
        confidence = 0.82

        if not answer.strip():
            notes.append("Answer is empty.")
            confidence = 0.0

        if self._looks_domain_specific(question) and not context:
            notes.append("Domain-specific question answered without retrieved context.")
            confidence -= 0.25

        if context:
            notes.append(f"Reviewed against {len(context)} retrieved context item(s).")

        return ReviewResult(
            passed=confidence >= 0.6,
            confidence=max(confidence, 0.0),
            notes=notes or ["No obvious verification issues found."],
        )

    def _looks_domain_specific(self, question: str) -> bool:
        keywords = {"yolo", "ultralytics", "training", "export", "inference", "dataset"}
        lowered = question.lower()
        return any(keyword in lowered for keyword in keywords)

