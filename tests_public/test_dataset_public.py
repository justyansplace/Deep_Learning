from __future__ import annotations

from PIL import Image

from hw.dataset import MathVQADataset, MathVQASample, sanitize_question


def test_sanitize_question_removes_visual_tokens() -> None:
    text = "<image_start> <image> Найдите x <image_end>"
    assert sanitize_question(text) == "Найдите x"


def test_dataset_loads_train_split(toy_manifest) -> None:
    ds = MathVQADataset(toy_manifest, split="train", max_samples=3)
    assert len(ds) == 3
    sample = ds[0]
    assert isinstance(sample, MathVQASample)
    assert isinstance(sample.image, Image.Image)
    assert sample.image.mode == "RGB"
    assert sample.question
    assert sample.options and all(isinstance(x, str) for x in sample.options)
    assert sample.answer in {"A", "B", "C", "D"}
    assert "<image>" not in sample.question


def test_dataset_loads_dev_split(toy_manifest) -> None:
    ds = MathVQADataset(toy_manifest, split="dev")
    assert len(ds) >= 2
    assert all(ds[i].subject for i in range(len(ds)))
