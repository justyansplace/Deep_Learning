# MathVista evaluation

MathVista используется в этом задании как внешний benchmark качества, а не как обязательный датасет для public tests.

## Когда он нужен

```text
Track A, CPU-only:
  не нужен

Track B, small GPU:
  можно использовать для отчёта / sanity-check качества

Track C, A100-20GB:
  рекомендуется для quality evaluation / bonus
```

## Подготовка

```bash
python -m pip install -e ".[ml]"
python scripts/prepare_mathvista_testmini.py --out assets/mathvista_testmini --max-samples 200
```

Для полного `testmini`:

```bash
python scripts/prepare_mathvista_testmini.py --out assets/mathvista_testmini --max-samples 1000
```

## Запуск benchmark

```bash
python -m hw.benchmark --config configs/eval_mathvista_testmini.yaml
```

## Важные правила

- Не коммитьте `assets/mathvista_testmini/` в GitHub.
- Не используйте MathVista как training dataset.
- В отчёте укажите, сколько примеров MathVista вы использовали: 50, 200 или 1000.
- Сравнивайте качество с baseline, а не только с абсолютным числом.

## Рекомендуемые режимы

```text
quick check: 50 примеров
report: 200 примеров
bonus/full eval: 1000 примеров testmini
```
