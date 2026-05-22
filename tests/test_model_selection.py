from __future__ import annotations

import unittest
from unittest.mock import patch

from yolorag.config.model_defaults import default_model_for, model_matrix
from yolorag.providers.base import LLMRequest
from yolorag.providers.deepseek_provider import DeepSeekProvider
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

    def test_deepseek_streaming_keeps_openai_compat_kwargs_minimal(self) -> None:
        provider = DeepSeekProvider(api_key="test-key")
        kwargs = provider._stream_completion_kwargs(
            LLMRequest(
                messages=[{"role": "user", "content": "hello"}],
                model="deepseek-v4-flash",
                mode="fast",
            )
        )

        self.assertTrue(kwargs["stream"])
        self.assertNotIn("stream_options", kwargs)
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


if __name__ == "__main__":
    unittest.main()
