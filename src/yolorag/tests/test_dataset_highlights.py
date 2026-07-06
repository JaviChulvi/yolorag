from __future__ import annotations

import unittest
from collections import Counter

from fastapi import HTTPException

from yolorag.api.datasets import (
    _image_class_ids,
    _select_diverse_images,
    parse_dataset_ref,
)


def _image(name: str, class_ids: list[int]) -> dict:
    return {"name": name, "labels": [{"classId": cid} for cid in class_ids]}


class ImageClassIdsTests(unittest.TestCase):
    def test_distinct_class_ids(self) -> None:
        self.assertEqual(_image_class_ids(_image("a", [0, 0, 1])), {0, 1})

    def test_no_labels(self) -> None:
        self.assertEqual(_image_class_ids({"name": "a"}), set())


class SelectDiverseImagesTests(unittest.TestCase):
    def test_prefers_image_covering_multiple_classes_first(self) -> None:
        images = [
            _image("cat_only", [0]),
            _image("dog_only", [1]),
            _image("both", [0, 1]),
        ]
        selected, covered = _select_diverse_images(images, 1)
        self.assertEqual([img["name"] for img in selected], ["both"])
        self.assertEqual(covered, {0, 1})

    def test_covers_all_classes_when_no_image_has_both(self) -> None:
        # All images are single-class; the selection must still span both.
        images = [_image(f"cat{i}", [0]) for i in range(6)]
        images += [_image(f"dog{i}", [1]) for i in range(6)]
        selected, covered = _select_diverse_images(images, 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(covered, {0, 1})

    def test_selection_is_balanced_across_classes(self) -> None:
        images = [_image(f"cat{i}", [0]) for i in range(6)]
        images += [_image(f"dog{i}", [1]) for i in range(6)]
        selected, _ = _select_diverse_images(images, 4)
        mix = Counter(next(iter(_image_class_ids(img))) for img in selected)
        self.assertEqual(mix[0], 2)
        self.assertEqual(mix[1], 2)

    def test_returns_fewer_when_pool_is_small(self) -> None:
        selected, _ = _select_diverse_images([_image("only", [0])], 4)
        self.assertEqual(len(selected), 1)


class ParseDatasetRefTests(unittest.TestCase):
    def test_accepts_full_url(self) -> None:
        self.assertEqual(
            parse_dataset_ref("https://platform.ultralytics.com/ddxy/datasets/dogs-cats"),
            ("ddxy", "dogs-cats"),
        )

    def test_accepts_username_slug(self) -> None:
        self.assertEqual(parse_dataset_ref("ddxy/dogs-cats"), ("ddxy", "dogs-cats"))

    def test_accepts_username_datasets_slug(self) -> None:
        self.assertEqual(parse_dataset_ref("ddxy/datasets/dogs-cats"), ("ddxy", "dogs-cats"))

    def test_rejects_empty(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            parse_dataset_ref("")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejects_single_token(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            parse_dataset_ref("justoneword")
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
