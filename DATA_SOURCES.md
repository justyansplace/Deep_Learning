# Данные в задании

## Что лежит в репозитории

В репозитории лежат только маленькие и безопасные для GitHub наборы:

```text
assets/toy_math_vqa/       # smoke-check для public tests
assets/math_vqa_medium/    # синтетическая практика побольше
```

## Что НЕ лежит в репозитории

MathVista, MATH-Vision, MAVIS и другие внешние датасеты не нужно коммитить в GitHub. Они подключаются отдельно через Hugging Face / локальный кеш / преподавательский сервер.

## Роль каждого набора

### `toy_math_vqa`

Используется для public tests и smoke-запусков. Этот набор отвечает только на вопрос:

```text
"Работает ли пайплайн технически?"
```

Он не отвечает на вопрос:

```text
"Хорошая ли модель?"
```

### `math_vqa_medium`

Синтетический набор побольше. Подходит для:

```text
- локального train loop;
- отчёта;
- sanity-check перед GPU-треком;
- демонстрации, что loss/accuracy считаются на более чем 3 примерах.
```

### MathVista

MathVista — основной внешний benchmark для проверки качества. Он используется для вопроса:

```text
"Насколько модель справляется с настоящими визуально-математическими задачами?"
```

Для подготовки локальной копии используйте:

```bash
python -m pip install -e ".[ml]"
python scripts/prepare_mathvista_testmini.py --out assets/mathvista_testmini --max-samples 200
```

Для оценки:

```bash
python -m hw.benchmark --config configs/eval_mathvista_testmini.yaml
```

Папка `assets/mathvista_testmini/` добавлена в `.gitignore` и не должна попадать в GitHub.

## Почему MathVista не кладём в repo

1. Это внешний benchmark, его правильнее получать из официального источника.
2. В датасете есть изображения и вопросы из разных source datasets; условия и источники указаны в metadata.
3. GitHub repo задания должен оставаться маленьким и быстро клонироваться.
4. MathVista предназначен прежде всего для evaluation, а не для обучения.

## Профильные источники для математиков

### MathVista

Benchmark для visual mathematical reasoning: графики, схемы, таблицы, математические изображения и вопросы.

### MAVIS

Данные для mathematical visual instruction tuning: caption/alignment и instruction-style задачи по визуальной математике.

### MATH-Vision

Визуальные математические задачи из реальных математических соревнований.

### We-Math

Benchmark визуального математического рассуждения с разбиением по математическим понятиям и reasoning steps.

### MMMU STEM/math subset

Можно использовать только STEM/math-подмножества. Не используйте `Marketing` как основной benchmark для студентов-математиков.

## Что не является профильным основным источником

```text
Flickr30k
TextVQA
VQAv2
MMMU Marketing
```

Их можно использовать только как техническую демонстрацию, но не как основную цель оценки.
