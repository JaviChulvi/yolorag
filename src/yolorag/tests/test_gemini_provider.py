from __future__ import annotations

import base64
import unittest
from decimal import Decimal
from types import SimpleNamespace

from fastapi import HTTPException

from yolorag.api import datasets as ds
from yolorag.providers.base import LLMResponse, ProviderError
from yolorag.providers.gemini_provider import GeminiProvider
from yolorag.usage.models import CostBreakdown, TokenUsage


def _image_message(*data_urls: str, prompt: str = "Describe.") -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": url}} for url in data_urls]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _encode_test_image(width: int = 8, height: int = 8) -> bytes:
    """A small, real PNG so the cv2 mosaic path can actually decode it."""
    import cv2
    import numpy as np

    return cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))[1].tobytes()


class GeminiContentTests(unittest.TestCase):
    def test_build_contents_keeps_images_first_then_prompt(self) -> None:
        messages = _image_message(
            _data_url(b"\xff\xd8\xff", "image/jpeg"),
            _data_url(b"RIFF", "image/webp"),
        )
        contents, system = GeminiProvider.build_contents(messages)

        self.assertIsNone(system)
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        parts = contents[0].parts
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0].inline_data.mime_type, "image/jpeg")
        self.assertEqual(parts[1].inline_data.mime_type, "image/webp")
        self.assertEqual(parts[2].text, "Describe.")

    def test_build_contents_hoists_system_and_maps_roles(self) -> None:
        messages = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        contents, system = GeminiProvider.build_contents(messages)

        self.assertEqual(system, "Be terse.")
        self.assertEqual([c.role for c in contents], ["user", "model"])
        self.assertEqual(contents[0].parts[0].text, "Hi")

    def test_decode_image_url_rejects_non_data_url(self) -> None:
        with self.assertRaises(ProviderError) as ctx:
            GeminiProvider._decode_image_url({"url": "https://cdn.ul.run/a.jpg"})
        self.assertEqual(ctx.exception.status, 400)

    def test_decode_image_url_reads_mime_and_bytes(self) -> None:
        data, mime = GeminiProvider._decode_image_url(
            {"url": _data_url(b"hello", "image/png")}
        )
        self.assertEqual(mime, "image/png")
        self.assertEqual(data, b"hello")

    def test_extract_text_returns_stripped_text(self) -> None:
        response = SimpleNamespace(text="  A cat and a dog.  ")
        self.assertEqual(GeminiProvider.extract_text(response), "A cat and a dog.")

    def test_extract_text_raises_on_block(self) -> None:
        response = SimpleNamespace(
            text=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"), candidates=[]
        )
        with self.assertRaises(ProviderError) as ctx:
            GeminiProvider.extract_text(response)
        self.assertEqual(ctx.exception.status, 400)

    def test_extract_text_raises_on_empty(self) -> None:
        response = SimpleNamespace(
            text=None,
            prompt_feedback=None,
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
        )
        with self.assertRaises(ProviderError):
            GeminiProvider.extract_text(response)

    def test_extract_usage_reads_metadata(self) -> None:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1290, candidates_token_count=62, total_token_count=1352
            )
        )
        usage = GeminiProvider.extract_usage(response)
        self.assertEqual(usage.input_tokens, 1290)
        self.assertEqual(usage.output_tokens, 62)
        self.assertEqual(usage.total_tokens, 1352)

    def test_extract_usage_none_when_absent(self) -> None:
        self.assertIsNone(GeminiProvider.extract_usage(SimpleNamespace()))


class VisionProviderListingTests(unittest.TestCase):
    def _gemini(self) -> dict:
        from yolorag.providers.factory import list_vision_providers

        return next(p for p in list_vision_providers() if p["name"] == "gemini")

    def test_dropdown_models_come_from_model_defaults(self) -> None:
        from yolorag.config.model_defaults import model_matrix

        gemini = self._gemini()
        expected = list(dict.fromkeys(model_matrix()["gemini"].values()))
        self.assertEqual(gemini["models"], expected)
        self.assertEqual(gemini["default_model"], model_matrix()["gemini"]["fast"])

    def test_every_listed_gemini_model_is_priced(self) -> None:
        from yolorag.providers.factory import PROVIDER_INFO
        from yolorag.usage.cost_calculator import CostCalculator

        gemini = self._gemini()
        calc = CostCalculator()
        usage = TokenUsage(input_tokens=1000, output_tokens=100, total_tokens=1100)
        self.assertTrue(gemini["models"], "dropdown has no models")
        for model in gemini["models"]:
            cost = calc.calculate(
                provider=PROVIDER_INFO["gemini"].pricing_provider, model=model, usage=usage
            )
            self.assertIsNone(cost.unavailable_reason, f"{model} has no pricing")
            self.assertGreater(float(cost.total_usd), 0.0, f"{model} priced at 0")


class DescribeProvidersListingTests(unittest.TestCase):
    def test_providers_listing_has_gemini(self) -> None:
        names = [p["name"] for p in ds.dataset_describe_providers()["providers"]]
        self.assertIn("gemini", names)


class FetchImageAllowlistTests(unittest.TestCase):
    def test_rejects_disallowed_host(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            ds._fetch_image_bytes("https://evil.example.com/a.jpg")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            ds._fetch_image_bytes("ftp://cdn.ul.run/a.jpg")
        self.assertEqual(ctx.exception.status_code, 400)


class MosaicComposeTests(unittest.TestCase):
    def test_compose_mosaic_makes_one_decodable_grid(self) -> None:
        import cv2
        import numpy as np

        images = [(_encode_test_image(), "image/png") for _ in range(4)]
        out = ds._compose_mosaic(images)
        decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        # 2x2 grid: two cells + three gaps on each side.
        side = 2 * ds._MOSAIC_CELL_PX + 3 * ds._MOSAIC_GAP_PX
        self.assertEqual(decoded.shape[:2], (side, side))

    def test_compose_mosaic_raises_when_undecodable(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            ds._compose_mosaic([(b"not-an-image", "image/png")])
        self.assertEqual(ctx.exception.status_code, 502)


class DrawBoxesTests(unittest.TestCase):
    def test_draw_boxes_marks_a_blank_image(self) -> None:
        import numpy as np

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        ds._draw_boxes(image, [ds.BoxAnnotation(bbox=[0.5, 0.5, 0.4, 0.4], className="cat", classId=0)])
        # The rectangle + label tag leave non-black pixels behind.
        self.assertGreater(int(np.count_nonzero(image)), 0)

    def test_draw_boxes_skips_malformed_bbox(self) -> None:
        import numpy as np

        image = np.zeros((50, 50, 3), dtype=np.uint8)
        ds._draw_boxes(image, [ds.BoxAnnotation(bbox=[0.5, 0.5], classId=0)])
        self.assertEqual(int(np.count_nonzero(image)), 0)

    def test_annotated_data_url_reencodes_only_when_boxes_present(self) -> None:
        raw = _encode_test_image(32, 32)
        with_boxes = ds._annotated_data_url(
            raw, "image/png", [ds.BoxAnnotation(bbox=[0.5, 0.5, 0.5, 0.5], classId=1)]
        )
        without = ds._annotated_data_url(raw, "image/png", [])
        self.assertTrue(with_boxes.startswith("data:image/jpeg;base64,"))
        self.assertTrue(without.startswith("data:image/png;base64,"))

    def test_annotated_data_url_passthrough_on_undecodable(self) -> None:
        out = ds._annotated_data_url(
            b"not-an-image", "image/webp", [ds.BoxAnnotation(bbox=[0.5, 0.5, 0.2, 0.2])]
        )
        self.assertTrue(out.startswith("data:image/webp;base64,"))


def _count_image_parts(messages: list[dict]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            count += sum(
                1 for part in content if isinstance(part, dict) and part.get("type") == "image_url"
            )
    return count


class _FakeProvider:
    provider_name = "gemini"

    def __init__(self) -> None:
        self.last_request = None

    async def complete(self, request) -> LLMResponse:
        self.last_request = request
        return LLMResponse(
            content=f"{request.model}:{_count_image_parts(request.messages)}",
            provider="gemini",
            model=request.model,
            usage=TokenUsage(input_tokens=1000, output_tokens=100, total_tokens=1100),
            cost=CostBreakdown(total_usd=Decimal("0.0004")),
            latency_ms=1234,
            raw_response={},
        )


class DescribeEndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_get = ds.get_llm_provider
        self._orig_fetch = ds._fetch_image_bytes
        self.fake = _FakeProvider()
        self._img = _encode_test_image()
        ds.get_llm_provider = lambda name, api_base=None: self.fake
        ds._fetch_image_bytes = lambda url: (self._img, "image/webp")

    def tearDown(self) -> None:
        ds.get_llm_provider = self._orig_get
        ds._fetch_image_bytes = self._orig_fetch

    async def test_describe_uses_provider_default_model(self) -> None:
        out = await ds.dataset_describe(
            ds.DescribeBody(prompt="Describe.", provider="gemini", images=["https://cdn.ul.run/1.webp"])
        )
        self.assertEqual(out["model"], "gemini-3.1-flash-lite")
        self.assertEqual(out["imageCount"], 1)
        self.assertTrue(out["description"].startswith("gemini-3.1-flash-lite"))

    async def test_describe_builds_image_message(self) -> None:
        await ds.dataset_describe(
            ds.DescribeBody(prompt="Describe.", provider="gemini", images=["https://cdn.ul.run/1.webp"])
        )
        content = self.fake.last_request.messages[0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/webp;base64,"))
        self.assertEqual(content[-1], {"type": "text", "text": "Describe."})

    async def test_separate_mode_sends_all_images(self) -> None:
        out = await ds.dataset_describe(
            ds.DescribeBody(
                prompt="hi", provider="gemini",
                images=[f"https://cdn.ul.run/{i}.webp" for i in range(4)],
            )
        )
        self.assertFalse(out["mosaic"])
        self.assertEqual(out["imageCount"], 4)
        image_parts = [p for p in self.fake.last_request.messages[0]["content"] if p.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 4)

    async def test_mosaic_sends_single_combined_image(self) -> None:
        out = await ds.dataset_describe(
            ds.DescribeBody(
                prompt="hi", provider="gemini", mosaic=True,
                images=[f"https://cdn.ul.run/{i}.webp" for i in range(4)],
            )
        )
        self.assertTrue(out["mosaic"])
        self.assertEqual(out["imageCount"], 1)
        self.assertEqual(out["sourceImageCount"], 4)
        image_parts = [p for p in self.fake.last_request.messages[0]["content"] if p.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue(image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    async def test_mosaic_ignored_for_single_image(self) -> None:
        out = await ds.dataset_describe(
            ds.DescribeBody(prompt="hi", provider="gemini", mosaic=True, images=["https://cdn.ul.run/1.webp"])
        )
        self.assertFalse(out["mosaic"])
        self.assertEqual(out["imageCount"], 1)

    async def test_boxes_are_burned_into_separate_images(self) -> None:
        out = await ds.dataset_describe(
            ds.DescribeBody(
                prompt="hi",
                provider="gemini",
                images=[
                    {
                        "url": "https://cdn.ul.run/1.webp",
                        "boxes": [{"bbox": [0.5, 0.5, 0.4, 0.4], "className": "cat", "classId": 0}],
                    }
                ],
            )
        )
        self.assertFalse(out["mosaic"])
        parts = [p for p in self.fake.last_request.messages[0]["content"] if p.get("type") == "image_url"]
        self.assertEqual(len(parts), 1)
        # Re-encoded as JPEG because a box was drawn in (raw passthrough would be webp).
        self.assertTrue(parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    async def test_describe_returns_usage_cost_and_latency(self) -> None:
        out = await ds.dataset_describe(
            ds.DescribeBody(prompt="Describe.", provider="gemini", images=["https://cdn.ul.run/1.webp"])
        )
        self.assertEqual(out["latencyMs"], 1234)
        self.assertEqual(out["usage"]["inputTokens"], 1000)
        self.assertEqual(out["usage"]["outputTokens"], 100)
        self.assertEqual(out["costUsd"], 0.0004)

    async def test_describe_requires_prompt_and_images(self) -> None:
        with self.assertRaises(HTTPException):
            await ds.dataset_describe(ds.DescribeBody(prompt=" ", images=["https://cdn.ul.run/1.webp"]))
        with self.assertRaises(HTTPException):
            await ds.dataset_describe(ds.DescribeBody(prompt="hi", images=[]))

    async def test_describe_caps_image_count(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await ds.dataset_describe(
                ds.DescribeBody(prompt="hi", images=[f"https://cdn.ul.run/{i}.webp" for i in range(9)])
            )
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_provider_error_maps_to_http_status(self) -> None:
        class _Raising:
            provider_name = "gemini"

            async def complete(self, request):
                raise ProviderError("no key", status=400)

        ds.get_llm_provider = lambda name, api_base=None: _Raising()
        with self.assertRaises(HTTPException) as ctx:
            await ds.dataset_describe(
                ds.DescribeBody(prompt="hi", provider="gemini", images=["https://cdn.ul.run/1.webp"])
            )
        self.assertEqual(ctx.exception.status_code, 400)

    async def test_unknown_provider_maps_to_400(self) -> None:
        # Fall back to the real factory, which raises ValueError for unknown names.
        ds.get_llm_provider = self._orig_get
        with self.assertRaises(HTTPException) as ctx:
            await ds.dataset_describe(
                ds.DescribeBody(
                    prompt="hi", provider="nope", model="x", images=["https://cdn.ul.run/1.webp"]
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
