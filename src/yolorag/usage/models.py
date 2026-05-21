from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


ZERO = Decimal("0")
ONE_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def normalized_total(self) -> int:
        if self.total_tokens:
            return self.total_tokens
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class CostBreakdown:
    input_usd: Decimal = ZERO
    output_usd: Decimal = ZERO
    cache_read_usd: Decimal = ZERO
    cache_write_usd: Decimal = ZERO
    reasoning_usd: Decimal = ZERO
    total_usd: Decimal = ZERO
    pricing_source: str = "unknown"
    estimated: bool = True
    unavailable_reason: str | None = None

    @classmethod
    def unavailable(cls, reason: str) -> "CostBreakdown":
        return cls(unavailable_reason=reason)


@dataclass(frozen=True)
class ModelPricing:
    provider: str
    model: str
    input_per_1m: Decimal
    output_per_1m: Decimal
    cached_input_per_1m: Decimal = ZERO
    cache_write_per_1m: Decimal = ZERO
    source: str = "local"

    def price_tokens(self, tokens: int, rate_per_1m: Decimal) -> Decimal:
        return (Decimal(tokens) / ONE_MILLION) * rate_per_1m

