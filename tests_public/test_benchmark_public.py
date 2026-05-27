from __future__ import annotations

from hw.benchmark import build_benchmark_prompt, compute_accuracy, parse_mc_answer


def test_parse_mc_answer() -> None:
    assert parse_mc_answer("A") == "A"
    assert parse_mc_answer("(B)") == "B"
    assert parse_mc_answer("Answer: C") == "C"
    assert parse_mc_answer("The correct answer is D.") == "D"
    assert parse_mc_answer("не знаю") is None


def test_build_benchmark_prompt_contains_options() -> None:
    prompt = build_benchmark_prompt("Чему равен x?", ["A) 1", "B) 2"])
    assert "Чему равен x?" in prompt
    assert "A) 1" in prompt
    assert "B) 2" in prompt
    assert "Ответ" in prompt


def test_compute_accuracy() -> None:
    rows = [
        {"prediction": "A", "answer": "A", "subject": "geometry"},
        {"prediction": "B", "answer": "A", "subject": "geometry"},
        {"prediction": "C", "answer": "C", "subject": "plots"},
    ]
    metrics = compute_accuracy(rows)
    assert metrics["overall"] == 2 / 3
    assert metrics["subject/geometry"] == 0.5
    assert metrics["subject/plots"] == 1.0
