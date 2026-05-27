from __future__ import annotations

import torch

from hw.dataset import MathVQADataset
from hw.processor import MathVLMProcessor, ProcessorConfig


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<eos>": 1, "<image>": 2}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for token in text.replace("\n", " ").split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False, max_length: int | None = None):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_preprocess_image_shape(toy_manifest) -> None:
    sample = MathVQADataset(toy_manifest, split="train", max_samples=1)[0]
    processor = MathVLMProcessor(DummyTokenizer(), ProcessorConfig(image_size=64, num_tiles=1))
    pixels = processor.preprocess_image(sample.image)
    assert isinstance(pixels, torch.Tensor)
    assert pixels.shape == (1, 3, 64, 64)
    assert pixels.dtype in {torch.float32, torch.float16, torch.bfloat16}


def test_tokenize_labels_mask_prompt(toy_manifest) -> None:
    sample = MathVQADataset(toy_manifest, split="train", max_samples=1)[0]
    processor = MathVLMProcessor(DummyTokenizer(), ProcessorConfig(max_length=128, num_image_tokens=4))
    item = processor.tokenize_sample(sample)
    assert set(item) >= {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == item["attention_mask"].shape == item["labels"].shape
    assert (item["labels"] != processor.config.ignore_index).any(), "answer tokens must contribute to loss"
    assert (item["labels"] == processor.config.ignore_index).any(), "prompt tokens must be masked"


def test_collate_pads_batch(toy_manifest) -> None:
    ds = MathVQADataset(toy_manifest, split="train", max_samples=2)
    processor = MathVLMProcessor(DummyTokenizer(), ProcessorConfig(image_size=64, max_length=128))
    batch = [processor(ds[i]) for i in range(2)]
    out = processor.collate(batch)
    assert out["input_ids"].ndim == 2
    assert out["labels"].shape == out["input_ids"].shape
    assert out["pixel_values"].shape[:3] == (2, 1, 3)
