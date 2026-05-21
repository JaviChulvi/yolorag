from __future__ import annotations

from decimal import Decimal
from typing import Any

from genai_prices import Usage as GenAIUsage
from genai_prices import calc_price as genai_calc_price

from yolorag.usage.models import CostBreakdown, ModelPricing, TokenUsage
from yolorag.usage.pricing_registry import PricingRegistry


class CostCalculator:
    def __init__(self, pricing_registry: PricingRegistry | None = None) -> None:
        self.pricing_registry = pricing_registry or PricingRegistry()

    def calculate(self, provider: str, model: str, usage: TokenUsage) -> CostBreakdown:
        local_price = self.pricing_registry.get(provider=provider, model=model)
        if local_price:
            return self._from_local_pricing(local_price, usage)

        genai_price = self._from_genai_prices(provider=provider, model=model, usage=usage)
        if genai_price:
            return genai_price

        return CostBreakdown.unavailable(
            f"No pricing found for provider={provider!r}, model={model!r}"
        )

    def _from_local_pricing(
        self,
        pricing: ModelPricing,
        usage: TokenUsage,
    ) -> CostBreakdown:
        uncached_input_tokens = max(usage.input_tokens - usage.cached_input_tokens, 0)

        input_usd = pricing.price_tokens(uncached_input_tokens, pricing.input_per_1m)
        output_usd = pricing.price_tokens(usage.output_tokens, pricing.output_per_1m)
        cache_read_usd = pricing.price_tokens(
            usage.cached_input_tokens,
            pricing.cached_input_per_1m,
        )
        cache_write_usd = pricing.price_tokens(
            usage.cache_write_tokens,
            pricing.cache_write_per_1m,
        )

        return CostBreakdown(
            input_usd=input_usd,
            output_usd=output_usd,
            cache_read_usd=cache_read_usd,
            cache_write_usd=cache_write_usd,
            total_usd=input_usd + output_usd + cache_read_usd + cache_write_usd,
            pricing_source=pricing.source,
            estimated=True,
        )

    def _from_genai_prices(
        self,
        provider: str,
        model: str,
        usage: TokenUsage,
    ) -> CostBreakdown | None:
        try:
            price = genai_calc_price(
                GenAIUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                ),
                model_ref=model,
                provider_id=provider,
            )
        except Exception:
            return None

        return CostBreakdown(
            input_usd=self._decimal_attr(price, "input_price"),
            output_usd=self._decimal_attr(price, "output_price"),
            total_usd=self._decimal_attr(price, "total_price"),
            pricing_source="genai-prices",
            estimated=True,
        )

    def _decimal_attr(self, obj: Any, attr: str) -> Decimal:
        return Decimal(str(getattr(obj, attr, "0")))
