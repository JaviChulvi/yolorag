from __future__ import annotations

import json
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any

from yolorag.usage.models import ModelPricing


class PricingRegistry:
    def __init__(self, pricing_path: Path | None = None) -> None:
        self.pricing_path = pricing_path
        self._prices = self._load_prices()

    def get(self, provider: str, model: str) -> ModelPricing | None:
        provider_key = provider.lower()
        model_key = model.lower()
        candidates = [
            f"{provider_key}/{model_key}",
            model_key,
        ]

        for candidate in candidates:
            raw = self._prices.get(candidate)
            if raw:
                return self._to_model_pricing(provider_key, model_key, raw)
        return None

    def _load_prices(self) -> dict[str, dict[str, Any]]:
        if self.pricing_path:
            if not self.pricing_path.exists():
                return {}
            return self._normalize_keys(json.loads(self.pricing_path.read_text()))

        package_file = resources.files("yolorag.usage").joinpath("pricing.json")
        return self._normalize_keys(json.loads(package_file.read_text()))

    def _normalize_keys(self, raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {key.lower(): value for key, value in raw.items()}

    def _to_model_pricing(
        self,
        provider: str,
        model: str,
        raw: dict[str, Any],
    ) -> ModelPricing:
        return ModelPricing(
            provider=provider,
            model=model,
            input_per_1m=Decimal(str(raw["input_per_1m"])),
            output_per_1m=Decimal(str(raw["output_per_1m"])),
            cached_input_per_1m=Decimal(str(raw.get("cached_input_per_1m", "0"))),
            cache_write_per_1m=Decimal(str(raw.get("cache_write_per_1m", "0"))),
            source=str(raw.get("source", "local")),
        )

