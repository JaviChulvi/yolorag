from __future__ import annotations

import unittest
from unittest.mock import patch

from yolorag.runtime import _build_reranker


class RuntimeConfigTests(unittest.TestCase):
    def test_reranker_requires_credentials(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                _build_reranker()

    def test_reranker_builds_from_credentials(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "YOLORAG_MONGODB_AI_API_KEY": "test-rerank-key",
                "YOLORAG_RERANK_MODEL": "rerank-test-model",
            },
            clear=True,
        ):
            reranker = _build_reranker()

        self.assertEqual(reranker.api_key, "test-rerank-key")
        self.assertEqual(reranker.model, "rerank-test-model")


if __name__ == "__main__":
    unittest.main()
