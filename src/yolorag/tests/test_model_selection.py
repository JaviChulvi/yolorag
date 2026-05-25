from __future__ import annotations

from asyncio import run
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from yolorag.config.model_defaults import default_model_for, model_matrix
from yolorag.providers.base import LLMRequest
from yolorag.providers.deepseek_provider import DeepSeekProvider
from yolorag.providers.factory import get_llm_provider
from yolorag.providers.openai_provider import OpenAIProvider
from yolorag.runtime import _resolve_model


class ModelSelectionTests(unittest.TestCase):
    def test_defaults_are_provider_and_mode_specific(self) -> None:
        self.assertEqual(default_model_for("openai", "fast"), "gpt-5.4-mini")
        self.assertEqual(default_model_for("openai", "deep"), "gpt-5.5")
        self.assertEqual(default_model_for("deepseek", "fast"), "deepseek-v4-flash")
        self.assertEqual(default_model_for("deepseek", "deep"), "deepseek-v4-pro")

    def test_model_matrix_is_static_builtin_defaults(self) -> None:
        matrix = model_matrix()
        self.assertEqual(matrix["openai"]["fast"], "gpt-5.4-mini")
        self.assertEqual(matrix["openai"]["deep"], "gpt-5.5")
        self.assertEqual(matrix["deepseek"]["fast"], "deepseek-v4-flash")
        self.assertEqual(matrix["deepseek"]["deep"], "deepseek-v4-pro")

    def test_env_override_uses_mode_specific_key(self) -> None:
        with patch.dict(
            "os.environ",
            {"YOLORAG_OPENAI_FAST_MODEL": "custom-fast"},
            clear=False,
        ):
            self.assertEqual(_resolve_model("openai", "fast"), "custom-fast")

    def test_legacy_provider_env_is_supported_as_fallback(self) -> None:
        with patch.dict(
            "os.environ",
            {"YOLORAG_DEEPSEEK_MODEL": "custom-deepseek"},
            clear=True,
        ):
            self.assertEqual(_resolve_model("deepseek", "deep"), "custom-deepseek")

    def test_model_resolution_normalizes_provider_name(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_resolve_model(" OpenAI ", "fast"), "gpt-5.4-mini")


class ProviderFactoryTests(unittest.TestCase):
    def test_builds_openai_provider_from_registry(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-openai"}, clear=True):
            provider = get_llm_provider(" openai ")

        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(provider.provider_name, "openai")

    def test_builds_deepseek_provider_from_registry(self) -> None:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-deepseek"}, clear=True):
            provider = get_llm_provider("deepseek")

        self.assertIsInstance(provider, DeepSeekProvider)
        self.assertEqual(provider.provider_name, "deepseek")

    def test_rejects_unsupported_provider_with_supported_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported provider 'gemini'. Supported providers: deepseek, openai.",
        ):
            get_llm_provider("gemini")

    def test_requires_provider_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "Missing required environment variable OPENAI_API_KEY.",
            ):
                get_llm_provider("openai")


class ProviderRequestOptionTests(unittest.TestCase):
    def test_openai_fast_sets_low_reasoning_and_low_verbosity(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        kwargs = provider._completion_kwargs(
            LLMRequest(
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-5.4-mini",
                mode="fast",
            )
        )

        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["verbosity"], "low")
        self.assertNotIn("temperature", kwargs)

    def test_openai_uses_max_completion_tokens(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        kwargs = provider._completion_kwargs(
            LLMRequest(
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-5.4-mini",
                mode="fast",
                max_tokens=512,
            )
        )

        self.assertEqual(kwargs["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", kwargs)

    def test_openai_tool_requests_omit_chat_completion_reasoning_controls(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        kwargs = provider._completion_kwargs(
            LLMRequest(
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-5.5",
                mode="deep",
                max_tokens=512,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "docs_search",
                            "description": "Search docs.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        )

        self.assertEqual(kwargs["tools"][0]["function"]["name"], "docs_search")
        self.assertEqual(kwargs["max_completion_tokens"], 512)
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("verbosity", kwargs)

    def test_openai_streaming_requests_usage_in_final_chunk(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        kwargs = provider._stream_completion_kwargs(
            LLMRequest(
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-5.4-mini",
                mode="fast",
            )
        )

        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})

    def test_openai_legacy_chat_models_keep_sampling_controls(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        kwargs = provider._completion_kwargs(
            LLMRequest(
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4o-mini",
                mode="fast",
                max_tokens=512,
            )
        )

        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["max_tokens"], 512)
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertNotIn("verbosity", kwargs)
        self.assertNotIn("max_completion_tokens", kwargs)

    def test_deepseek_fast_disables_thinking(self) -> None:
        provider = DeepSeekProvider(api_key="test-key")
        kwargs = provider._completion_kwargs(
            LLMRequest(
                messages=[{"role": "user", "content": "hello"}],
                model="deepseek-v4-flash",
                mode="fast",
            )
        )

        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", kwargs)

    def test_deepseek_streaming_requests_usage_in_final_chunk(self) -> None:
        provider = DeepSeekProvider(api_key="test-key")
        kwargs = provider._stream_completion_kwargs(
            LLMRequest(
                messages=[{"role": "user", "content": "hello"}],
                model="deepseek-v4-flash",
                mode="fast",
            )
        )

        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_deepseek_deep_enables_high_effort_thinking(self) -> None:
        provider = DeepSeekProvider(api_key="test-key")
        kwargs = provider._completion_kwargs(
            LLMRequest(
                messages=[{"role": "user", "content": "hard problem"}],
                model="deepseek-v4-pro",
                mode="deep",
            )
        )

        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("temperature", kwargs)

    def test_deepseek_preserves_reasoning_content_from_tool_response(self) -> None:
        provider = DeepSeekProvider(api_key="test-key")
        tool_call_payload = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "docs_search", "arguments": '{"query": "export"}'},
        }
        raw_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "I should search the docs.",
                        "tool_calls": [tool_call_payload],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        raw_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(model_dump=lambda: tool_call_payload)
                        ],
                    )
                )
            ],
            model_dump=lambda: raw_payload,
        )

        with patch.object(
            provider.client.chat.completions,
            "create",
            new=AsyncMock(return_value=raw_response),
        ):
            response = run(
                provider.complete(
                    LLMRequest(
                        messages=[{"role": "user", "content": "how do I export?"}],
                        model="deepseek-v4-pro",
                        mode="deep",
                    )
                )
            )

        self.assertEqual(response.reasoning_content, "I should search the docs.")
        self.assertEqual(response.tool_calls, [tool_call_payload])


if __name__ == "__main__":
    unittest.main()
