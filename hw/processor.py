from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        image = image.convert("RGB")
        size = self.config.image_size
        image = image.resize((size, size))

        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        tensor = tensor.unsqueeze(0)
        if self.config.num_tiles > 1:
            tensor = tensor.repeat(self.config.num_tiles, 1, 1, 1)
        return tensor

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_part = IMAGE_START_TOKEN + IMAGE_TOKEN * self.config.num_image_tokens + IMAGE_END_TOKEN
        prompt = image_part + "\n"
        prompt += "Вопрос: " + sample.question + "\n"
        prompt += "Варианты:\n" + "\n".join(sample.options) + "\n"
        prompt += "Ответ:"
        if include_answer:
            prompt += " " + sample.answer
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        prompt_text = self.build_prompt(sample, include_answer=False) + " "
        answer_text = sample.answer

        prompt_ids = self.tokenizer(prompt_text)["input_ids"]
        answer_ids = self.tokenizer(answer_text)["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [self.config.ignore_index] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)

        max_length = self.config.max_length
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = attention_mask[:max_length]

        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)}


    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        pad_id = self.tokenizer.pad_token_id
        ignore_index = self.config.ignore_index
        max_len = max(b["input_ids"].shape[0] for b in batch)

        input_ids = []
        attention_mask = []
        labels = []
        for b in batch:
            n = b["input_ids"].shape[0]
            pad_n = max_len - n
            input_ids.append(torch.cat([b["input_ids"], torch.full((pad_n,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_n, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((pad_n,), ignore_index, dtype=torch.long)]))

        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        return {"input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_mask),
                "labels": torch.stack(labels),
                "pixel_values": pixel_values}
