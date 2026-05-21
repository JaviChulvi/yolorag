from __future__ import annotations

from decimal import Decimal
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from yolorag.usage.cost_calculator import CostCalculator
from yolorag.usage.extractors import GeminiUsageExtractor, OpenAIUsageExtractor
from yolorag.usage.models import TokenUsage
from yolorag.usage.pricing_registry import PricingRegistry


class UsageExtractorTests(unittest.TestCase):
    def test_openai_usage_extracts_cached_and_reasoning_tokens(self) -> None:
        usage = OpenAIUsageExtractor().extract(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 25},
                    "completion_tokens_details": {"reasoning_tokens": 5},
                }
            }
        )

        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.cached_input_tokens, 25)
        self.assertEqual(usage.reasoning_tokens, 5)

    def test_gemini_usage_extracts_camel_case_metadata(self) -> None:
        usage = GeminiUsageExtractor().extract(
            {
                "usageMetadata": {
                    "promptTokenCount": 70,
                    "candidatesTokenCount": 14,
                    "totalTokenCount": 84,
                    "thoughtsTokenCount": 9,
                }
            }
        )

        self.assertEqual(usage.input_tokens, 70)
        self.assertEqual(usage.output_tokens, 14)
        self.assertEqual(usage.total_tokens, 84)
        self.assertEqual(usage.reasoning_tokens, 9)


class CostCalculatorTests(unittest.TestCase):
    def test_local_pricing_override_wins_before_genai_prices(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pricing_path = Path(temp_dir) / "pricing.json"
            pricing_path.write_text(
                """
                {
                  "openai/local-test-model": {
                    "input_per_1m": "1.00",
                    "output_per_1m": "2.00",
                    "source": "test-local"
                  }
                }
                """
            )

            cost = CostCalculator(
                PricingRegistry(pricing_path=pricing_path)
            ).calculate(
                provider="openai",
                model="local-test-model",
                usage=TokenUsage(input_tokens=1000, output_tokens=500),
            )

        self.assertEqual(cost.total_usd, Decimal("0.002000"))
        self.assertEqual(cost.pricing_source, "test-local")
        self.assertIsNone(cost.unavailable_reason)

    def test_unknown_model_returns_unavailable_cost(self) -> None:
        cost = CostCalculator().calculate(
            provider="unknown",
            model="missing-model",
            usage=TokenUsage(input_tokens=100, output_tokens=25),
        )

        self.assertEqual(str(cost.total_usd), "0")
        self.assertIsNotNone(cost.unavailable_reason)

    def test_known_model_falls_back_to_genai_prices(self) -> None:
        cost = CostCalculator().calculate(
            provider="openai",
            model="gpt-4o-mini",
            usage=TokenUsage(input_tokens=1000, output_tokens=200),
        )

        self.assertGreater(cost.total_usd, Decimal("0"))
        self.assertEqual(cost.pricing_source, "genai-prices")
        self.assertIsNone(cost.unavailable_reason)

    def test_deepseek_model_falls_back_to_genai_prices(self) -> None:
        cost = CostCalculator().calculate(
            provider="deepseek",
            model="deepseek-v4-flash",
            usage=TokenUsage(input_tokens=1000, output_tokens=200),
        )

        self.assertGreater(cost.total_usd, Decimal("0"))
        self.assertEqual(cost.pricing_source, "genai-prices")
        self.assertIsNone(cost.unavailable_reason)


if __name__ == "__main__":
    unittest.main()
