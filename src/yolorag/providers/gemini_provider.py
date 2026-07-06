from __future__ import annotations

import base64
import time
from typing import Any

from google import genai
from google.genai import errors, types

from yolorag.providers.base import (
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    Message,
    ProviderError,
)
from yolorag.usage.cost_calculator import CostCalculator
from yolorag.usage.models import TokenUsage


class GeminiProvider:
    """LLM provider for Google's Gemini models via the ``google-genai`` SDK.

    Implements the same ``LLMProvider`` contract as the OpenAI/DeepSeek adapters
    (async ``complete`` + ``stream_complete``). Chat ``messages`` are translated
    into Gemini ``contents``; OpenAI-style ``image_url`` parts carrying ``data:``
    URLs become inline image parts, so a vision request is just a normal chat
    request that happens to include images.

    Pricing note: genai-prices identifies Gemini models under the ``google``
    provider id, so cost is looked up with ``pricing_provider`` rather than the
    ``gemini`` display name.
    """

    provider_name = "gemini"
    pricing_provider = "google"

    def __init__(
        self,
        api_key: str,
        api_base: str | None = None,
        cost_calculator: CostCalculator | None = None,
        timeout: float = 60.0,
    ) -> None:
        http_kwargs: dict[str, Any] = {"timeout": int(timeout * 1000)}
        if api_base:
            http_kwargs["base_url"] = api_base.rstrip("/")
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(**http_kwargs),
        )
        self.cost_calculator = cost_calculator or CostCalculator()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        contents, system_instruction = self.build_contents(request.messages)
        config = self._config(request, system_instruction)

        started = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=config,
            )
        except errors.APIError as exc:
            raise self._api_error(exc) from exc
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - network/SDK faults -> upstream error
            raise ProviderError(f"Could not reach Gemini ({exc}).", status=502) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = self.extract_usage(response) or TokenUsage()
        cost = self.cost_calculator.calculate(
            provider=self.pricing_provider,
            model=request.model,
            usage=usage,
        )
        return LLMResponse(
            content=self.extract_text(response),
            provider=self.provider_name,
            model=request.model,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            raw_response=self._safe_dump(response),
        )

    async def stream_complete(self, request: LLMRequest):
        contents, system_instruction = self.build_contents(request.messages)
        config = self._config(request, system_instruction)

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config=config,
            )
        except errors.APIError as exc:
            raise self._api_error(exc) from exc
        except Exception as exc:  # noqa: BLE001 - network/SDK faults -> upstream error
            raise ProviderError(f"Could not reach Gemini ({exc}).", status=502) from exc

        usage: TokenUsage | None = None
        async for chunk in stream:
            chunk_usage = self.extract_usage(chunk)
            if chunk_usage is not None:
                usage = chunk_usage
            text = self._chunk_text(chunk)
            if text:
                yield LLMStreamEvent(content=text)

        final_usage = usage or TokenUsage()
        cost = self.cost_calculator.calculate(
            provider=self.pricing_provider,
            model=request.model,
            usage=final_usage,
        )
        yield LLMStreamEvent(usage=final_usage, cost=cost)

    def _config(
        self,
        request: LLMRequest,
        system_instruction: str | None,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens or None,
            system_instruction=system_instruction or None,
        )

    # --- message <-> Gemini content translation ---------------------------

    @staticmethod
    def build_contents(messages: list[Message]) -> tuple[list[types.Content], str | None]:
        """Translate chat ``messages`` into ``(contents, system_instruction)``.

        ``system`` messages are hoisted into a single system instruction; every
        other turn becomes a ``Content`` with role ``user``/``model``. Within a
        turn, parts keep their given order (so images-then-text stays that way).
        """
        contents: list[types.Content] = []
        system_texts: list[str] = []
        for message in messages:
            role = (message.get("role") or "user").lower()
            parts = GeminiProvider._parts(message.get("content"))
            if role == "system":
                system_texts.extend(part.text for part in parts if getattr(part, "text", None))
                continue
            if not parts:
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=gemini_role, parts=parts))
        system_instruction = "\n\n".join(text for text in system_texts if text) or None
        return contents, system_instruction

    @staticmethod
    def _parts(content: Any) -> list[types.Part]:
        if content is None:
            return []
        if isinstance(content, str):
            return [types.Part.from_text(text=content)] if content else []

        parts: list[types.Part] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append(types.Part.from_text(text=item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text") or ""
                if text:
                    parts.append(types.Part.from_text(text=text))
            elif item_type == "image_url":
                data, mime_type = GeminiProvider._decode_image_url(item.get("image_url"))
                parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        return parts

    @staticmethod
    def _decode_image_url(image_url: Any) -> tuple[bytes, str]:
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(url, str) or not url.startswith("data:"):
            raise ProviderError(
                "Gemini images must be provided as inline data: URLs.", status=400
            )
        header, _, encoded = url.partition(",")
        if not encoded:
            raise ProviderError("Malformed image data URL.", status=400)
        mime_type = "image/jpeg"
        meta = header[len("data:") :]
        if meta:
            candidate = meta.split(";", 1)[0].strip()
            if candidate:
                mime_type = candidate
        try:
            data = base64.b64decode(encoded)
        except Exception as exc:  # noqa: BLE001 - bad base64 is a client error
            raise ProviderError("Could not decode image data URL.", status=400) from exc
        return data, mime_type

    # --- response extraction ----------------------------------------------

    @staticmethod
    def extract_text(response: Any) -> str:
        text = GeminiProvider._chunk_text(response)
        if text and text.strip():
            return text.strip()

        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None) if feedback else None
        if block_reason:
            raise ProviderError(f"Gemini blocked the request ({block_reason}).", status=400)

        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        raise ProviderError(
            f"Gemini returned an empty response ({finish or 'no text'}).", status=502
        )

    @staticmethod
    def _chunk_text(chunk: Any) -> str | None:
        try:
            return chunk.text
        except Exception:  # noqa: BLE001 - .text can raise when a reply was blocked
            return None

    @staticmethod
    def extract_usage(response: Any) -> TokenUsage | None:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return None
        return TokenUsage(
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            total_tokens=getattr(meta, "total_token_count", 0) or 0,
            cached_input_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
        )

    @staticmethod
    def _safe_dump(response: Any) -> dict[str, Any]:
        try:
            return response.model_dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001 - keep a light, serializable trace
            return {"provider": "gemini"}

    @staticmethod
    def _api_error(exc: errors.APIError) -> ProviderError:
        code = getattr(exc, "code", 0) or 0
        status = 400 if 400 <= code < 500 else 502
        message = getattr(exc, "message", None) or str(exc)
        return ProviderError(f"Gemini: {message}", status=status)
