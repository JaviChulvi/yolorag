from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from yolorag.ingestion.docs_chunker import chunk_markdown_docs, has_unclosed_fence


class DocsChunkerTests(unittest.TestCase):
    def test_expands_macros_and_creates_url_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "macros").mkdir()
            (root / "en" / "modes").mkdir(parents=True)
            (root / "macros" / "args.md").write_text(
                "Macro paragraph with enough detail to be indexed as useful retrieval content.",
                encoding="utf-8",
            )
            (root / "en" / "modes" / "export.md").write_text(
                """---
description: Export docs
---

# Export

Intro paragraph.

{% include "macros/args.md" %}
""",
                encoding="utf-8",
            )

            chunks = chunk_markdown_docs(docs_root=root, max_chars=1000)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].doc_id, "modes/export")
        self.assertEqual(chunks[0].url, "https://docs.ultralytics.com/modes/export/")
        self.assertIn("Macro paragraph with enough detail", chunks[0].text)

    def test_reference_docs_are_tagged_and_can_be_excluded(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "en" / "reference" / "nn").mkdir(parents=True)
            (root / "en" / "reference" / "nn" / "tasks.md").write_text(
                "# Reference\n\n## ::: ultralytics.nn.tasks.torch_safe_load\n",
                encoding="utf-8",
            )

            chunks = chunk_markdown_docs(docs_root=root, max_chars=1000)
            excluded = chunk_markdown_docs(docs_root=root, max_chars=1000, include_reference=False)

        self.assertEqual(chunks[0].kind, "reference")
        self.assertEqual(chunks[0].reference_symbols, ["ultralytics.nn.tasks.torch_safe_load"])
        self.assertEqual(excluded, [])

    def test_large_sections_split_with_overlap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "en" / "guides").mkdir(parents=True)
            paragraphs = "\n\n".join(f"Paragraph {index} " + ("word " * 30) for index in range(8))
            (root / "en" / "guides" / "long.md").write_text(f"# Long\n\n{paragraphs}", encoding="utf-8")

            chunks = chunk_markdown_docs(docs_root=root, max_chars=500, overlap_chars=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.char_count <= 650 for chunk in chunks))

    def test_headings_inside_code_blocks_do_not_split_sections(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "en" / "usage").mkdir(parents=True)
            (root / "en" / "usage" / "python.md").write_text(
                """# Python Usage

This section includes a code block with comments that look like markdown headings.

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

# Run inference
results = model("image.jpg")
```

The closing explanation should remain in the same section.
""",
                encoding="utf-8",
            )

            chunks = chunk_markdown_docs(docs_root=root, max_chars=1200)

        self.assertEqual(len(chunks), 1)
        self.assertFalse(has_unclosed_fence(chunks[0].content))
        self.assertIn("# Load a model", chunks[0].content)
        self.assertIn("The closing explanation", chunks[0].content)

    def test_mkdocs_snippets_do_not_break_code_fences(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "en" / "datasets").mkdir(parents=True)
            (root / "en" / "datasets" / "sample.md").write_text(
                """# Dataset

## Dataset YAML

The dataset configuration is shown below.

```yaml
--8<-- "ultralytics/cfg/datasets/Sample.yaml"
```
""",
                encoding="utf-8",
            )

            chunks = chunk_markdown_docs(docs_root=root, max_chars=1200)

        self.assertEqual(len(chunks), 1)
        self.assertFalse(has_unclosed_fence(chunks[0].content))
        self.assertIn('--8<-- "ultralytics/cfg/datasets/Sample.yaml"', chunks[0].content)

    def test_large_code_blocks_stay_balanced(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "en" / "guides").mkdir(parents=True)
            code = "\n".join(f"print({index})" for index in range(350))
            (root / "en" / "guides" / "code.md").write_text(
                f"""# Code Guide

## Long Example

```python
{code}
```

The code above should remain syntactically fenced.
""",
                encoding="utf-8",
            )

            chunks = chunk_markdown_docs(docs_root=root, max_chars=500)

        self.assertTrue(any(chunk.content.count("```") == 2 for chunk in chunks))
        self.assertTrue(all(not has_unclosed_fence(chunk.content) for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
