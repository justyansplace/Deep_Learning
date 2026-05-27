from __future__ import annotations

import argparse
import functools
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from hw.constants import IGNORE_INDEX, IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import MathVLM, ModelConfig
from hw.processor import MathVLMProcessor, ProcessorConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """
    model.train()
    out = model(batch)
    loss = out["loss"] if isinstance(out, dict) else out.loss
    assert torch.isfinite(loss), "Loss is not finite"
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return float(loss.detach().cpu())


class _DummyTok:
    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<eos>": 1, IMAGE_START_TOKEN: 2, IMAGE_TOKEN: 3, IMAGE_END_TOKEN: 4}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.specials = (IMAGE_START_TOKEN, IMAGE_TOKEN, IMAGE_END_TOKEN)
        self.pattern = re.compile("(" + "|".join(re.escape(t) for t in self.specials) + ")")

    def encode(self, text):
        ids = []
        for part in self.pattern.split(text):
            if not part:
                continue
            if part in self.specials:
                ids.append(self.vocab[part])
            else:
                for word in part.split():
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
                    ids.append(self.vocab[word])
        return ids

    def __call__(self, text, **kwargs):
        ids = self.encode(text)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens=True):
        words = {i: t for t, i in self.vocab.items()}
        result = []
        for i in ids:
            token = words.get(int(i), "")
            if skip_special_tokens and token in ("<pad>", "<eos>", *self.specials):
                continue
            if token:
                result.append(token)
        return " ".join(result)


class _VisionStub(nn.Module):
    def __init__(self, image_size, hidden=32, seq_len=16):
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(3 * 4 * 4, seq_len * hidden)

    def forward(self, pixel_values):
        x = self.pool(pixel_values)
        x = x.flatten(1)
        x = self.proj(x)
        return x.view(x.shape[0], self.seq_len, self.hidden)


class _LLMOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class _LLMStub(nn.Module):
    def __init__(self, hidden=64, vocab_size=32000):
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        logits = self.lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, self.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
        return _LLMOutput(loss, logits)

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask=None, max_new_tokens=16, **kwargs):
        last = self.lm_head(inputs_embeds[:, -1:, :])
        next_id = last.argmax(dim=-1)
        return next_id.repeat(1, max_new_tokens)


def build_model_from_config(config):
    model_cfg = config["model"]
    proc_cfg = config["processor"]
    is_tiny = "tiny" in str(model_cfg["language_model"])

    if is_tiny:
        tokenizer = _DummyTok()
        vision_encoder = _VisionStub(image_size=proc_cfg["image_size"])
        language_model = _LLMStub()
        vision_hidden = vision_encoder.hidden
        text_hidden = language_model.hidden
        image_token_id = tokenizer.vocab[IMAGE_TOKEN]
    else:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["language_model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_tokens([IMAGE_START_TOKEN, IMAGE_TOKEN, IMAGE_END_TOKEN], special_tokens=True)
        vision_encoder = AutoModel.from_pretrained(model_cfg["vision_encoder"])
        language_model = AutoModelForCausalLM.from_pretrained(model_cfg["language_model"])
        language_model.resize_token_embeddings(len(tokenizer))
        vision_hidden = vision_encoder.config.hidden_size
        text_hidden = language_model.config.hidden_size
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    processor = MathVLMProcessor(tokenizer, ProcessorConfig(**proc_cfg))
    model = MathVLM(
        vision_encoder,
        language_model,
        ModelConfig(
            vision_hidden_size=vision_hidden,
            text_hidden_size=text_hidden,
            num_image_tokens=proc_cfg["num_image_tokens"],
            image_token_id=image_token_id,
        ),
    )
    if model_cfg.get("freeze_vision", True) and model_cfg.get("freeze_llm", True):
        model.freeze_backbones()
    return model, processor, tokenizer


def collate_samples(batch, processor):
    return processor.collate([processor(s) for s in batch])


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """
    data_cfg = config["data"]
    trainer_cfg = config["trainer"]

    dataset = MathVQADataset(
        data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )

    model, processor, tokenizer = build_model_from_config(config)

    device = trainer_cfg.get("device", "cpu")
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=trainer_cfg["local_batch_size"],
        shuffle=True,
        num_workers=trainer_cfg.get("num_workers", 0),
        collate_fn=functools.partial(collate_samples, processor=processor))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=trainer_cfg["learning_rate"],
        weight_decay=trainer_cfg.get("weight_decay", 0.0))

    max_steps = 3 if fast_train else trainer_cfg["max_steps"]
    accum_steps = max(1, trainer_cfg.get("global_batch_size", 1) // trainer_cfg.get("local_batch_size", 1))

    model.train()
    optimizer.zero_grad()
    step = 0
    micro_step = 0
    running_loss = 0.0
    while step < max_steps:
        stop = False
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss = out["loss"] if isinstance(out, dict) else out.loss
            assert torch.isfinite(loss), "Loss is not finite"
            (loss / accum_steps).backward()
            running_loss += float(loss.detach().cpu())
            micro_step += 1
            if micro_step % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                print(f"step {step}/{max_steps} loss={running_loss / accum_steps:.4f}")
                running_loss = 0.0
                if step >= max_steps:
                    stop = True
                    break
        if stop:
            break

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"adapter": model.adapter.state_dict()}, save_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
