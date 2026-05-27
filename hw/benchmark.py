from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from hw.constants import CHOICES, IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN
from hw.dataset import MathVQADataset

def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output."""

    if not text:
        return None
    matches = re.findall(r"[ABCDE]", text.upper())
    if not matches:
        return None

    letter = matches[-1]
    if letter in choices:
        return letter
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop."""
    from hw.train import build_model_from_config, set_seed

    set_seed(int(config.get("seed", 42)))

    data_cfg = config["data"]
    proc_cfg = config["processor"]
    inf_cfg = config.get("inference", {})

    dataset = MathVQADataset(
        data_cfg["eval_manifest"],
        split=data_cfg.get("split", "dev"),
        max_samples=data_cfg.get("max_samples"))

    model, processor, tokenizer = build_model_from_config(config)

    # если есть сохранённый adapter то будем его подгружать
    adapter_path = config["model"].get("adapter_path")
    if adapter_path and Path(adapter_path).exists():
        state = torch.load(adapter_path, map_location="cpu")
        model.adapter.load_state_dict(state["adapter"])

    device = inf_cfg.get("device", "cpu")
    model.to(device)
    model.eval()

    num_image_tokens = proc_cfg["num_image_tokens"]
    max_new_tokens = inf_cfg.get("max_new_tokens", 16)

    rows = []
    for i in range(len(dataset)):
        sample = dataset[i]
        prompt_text = build_benchmark_prompt(sample.question, sample.options)
        full_prompt = (IMAGE_START_TOKEN + IMAGE_TOKEN * num_image_tokens + IMAGE_END_TOKEN + "\n" + prompt_text)
        ids = tokenizer(full_prompt)["input_ids"]
        pixel_values = processor.preprocess_image(sample.image).unsqueeze(0)
        batch = {"input_ids": torch.tensor([ids], dtype=torch.long).to(device),
                "attention_mask": torch.tensor([[1] * len(ids)], dtype=torch.long).to(device),
                "pixel_values": pixel_values.to(device)}
        out_ids = model.generate(batch, max_new_tokens=max_new_tokens)
        if hasattr(tokenizer, "decode"):
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        else:
            text = str(out_ids)
        pred = parse_mc_answer(text)
        rows.append({"id": sample.id,
                    "prediction": pred,
                    "answer": sample.answer,
                    "subject": sample.subject})

    output_path = inf_cfg.get("output_path")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
