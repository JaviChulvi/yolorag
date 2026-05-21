from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from yolorag.cli import _resolve_model
from yolorag.config.model_defaults import default_model_for, model_matrix
from yolorag.providers.base import LLMRequest
from yolorag.providers.deepseek_provider import DeepSeekProvider
from yolorag.providers.openai_provider import OpenAIProvider


class ModelSelectionTests(unittest.TestCase):
    def test_defaults_are_provider_and_mode_specific(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_config_path = Path(temp_dir) / "missing-models.json"
            self.assertEqual(
                default_model_for("openai", "fast", config_path=missing_config_path),
                "gpt-5.4-mini",
            )
            self.assertEqual(
                default_model_for("openai", "deep", config_path=missing_config_path),
                "gpt-5.5",
            )
            self.assertEqual(
                default_model_for("deepseek", "fast", config_path=missing_config_path),
                "deepseek-v4-flash",
            )
            self.assertEqual(
                default_model_for("deepseek", "deep", config_path=missing_config_path),
                "deepseek-v4-pro",
            )

    def test_models_json_config_is_easy_to_swap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "models.json"
            config_path.write_text(
                """
                {
                  "openai": {
                    "fast": { "model": "openai-fast-alt" },
                    "deep": { "model": "openai-thinking-alt" }
                  },
                  "deepseek": {
                    "fast": { "model": "deepseek-fast-alt" },
                    "deep": { "model": "deepseek-thinking-alt" }
                  }
                }
                """
            )

            matrix = model_matrix(config_path=config_path)

        self.assertEqual(matrix["openai"]["fast"], "openai-fast-alt")
        self.assertEqual(matrix["openai"]["deep"], "openai-thinking-alt")
        self.assertEqual(matrix["deepseek"]["fast"], "deepseek-fast-alt")
        self.assertEqual(matrix["deepseek"]["deep"], "deepseek-thinking-alt")

    def test_resolve_model_uses_models_config_after_env(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "models.json"
            config_path.write_text(
                """
                {
                  "openai": {
                    "fast": { "model": "file-fast" },
                    "deep": { "model": "file-thinking" }
                  }
                }
                """
            )

            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(
                    _resolve_model(
                        "openai",
                        "fast",
                        None,
                        models_config=str(config_path),
                    ),
                    "file-fast",
                )

    def test_env_override_uses_mode_specific_key(self) -> None:
        with patch.dict(
            "os.environ",
            {"YOLORAG_OPENAI_FAST_MODEL": "custom-fast"},
            clear=False,
        ):
            self.assertEqual(_resolve_model("openai", "fast", None), "custom-fast")

    def test_cli_override_wins_over_env_and_default(self) -> None:
        with patch.dict(
            "os.environ",
            {"YOLORAG_DEEPSEEK_THINKING_MODEL": "custom-thinking"},
            clear=False,
        ):
            self.assertEqual(
                _resolve_model("deepseek", "deep", "explicit-model"),
                "explicit-model",
            )


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
