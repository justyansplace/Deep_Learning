# Постановка задания

## Цель

Реализовать минимальную multimodal language model. Модель получает изображение и вопрос, а возвращает ответ или вариант ответа.

Примеры задач:

- найти значение функции по графику;
- определить длину стороны в геометрической схеме;
- прочитать величину из столбчатой диаграммы;
- выбрать правильный ответ по визуальной формуле или таблице.

## Как понимать датасеты в этом задании

В задании есть три уровня данных.

### 1. `toy_math_vqa`: smoke-check, а не оценка качества

```text
assets/toy_math_vqa/
```

Toy-набор нужен, чтобы быстро проверить, что всё технически работает:

```text
- dataset читает manifest;
- PIL-картинки открываются;
- processor строит input_ids / labels / pixel_values;
- model делает forward;
- train loop делает несколько шагов;
- benchmark умеет парсить ответы.
```

Он слишком маленький, поэтому по нему **нельзя делать вывод о качестве модели**.

### 2. `math_vqa_medium`: локальная практика

```text
assets/math_vqa_medium/
```

Это синтетический набор побольше. Его можно использовать для отчёта и локальных экспериментов без GPU и без скачивания внешних данных.

### 3. MathVista: проверка качества

MathVista используется как профильный benchmark для визуально-математического reasoning. Он **не лежит в репозитории** и скачивается отдельно только для расширенного трека / бонуса.

Команда подготовки:

```bash
python -m pip install -e ".[ml]"
python scripts/prepare_mathvista_testmini.py --out assets/mathvista_testmini --max-samples 200
```

Команда оценки:

```bash
python -m hw.benchmark --config configs/eval_mathvista_testmini.yaml
```

Важно: MathVista используется как evaluation dataset. Не используйте его как обучающий датасет.

## Базовая архитектура

Рекомендуемая архитектура:

```text
image -> ViT encoder -> trainable adapter -> visual embeddings
text question -> tokenizer -> text embeddings
visual embeddings + text embeddings -> frozen/instruct LLM -> answer
```

В обязательной части можно считать, что:

```text
vision encoder заморожен
LLM заморожен
обучается только adapter
LoRA используется только в Track C или как бонус
```

## Обязательные компоненты

### 1. `dataset.py`

Нужно реализовать загрузку примеров из `manifest.jsonl`.

Ожидаемый формат примера:

```json
{
  "id": "toy_train_000",
  "split": "train",
  "image": "images/line_plot_0.png",
  "question": "На графике дана прямая. Чему равен y при x=2?",
  "options": ["A) 3", "B) 5", "C) 6", "D) 7"],
  "answer": "B",
  "subject": "algebra",
  "source": "toy_math_vqa"
}
```

### 2. `processor.py`

Нужно реализовать:

- приведение изображения к RGB;
- resize/crop/pad до `image_size`;
- разбиение на `num_tiles` тайлов;
- нормализацию изображения;
- построение prompt с visual special tokens;
- токенизацию question/options/answer;
- `labels`, где prompt замаскирован `IGNORE_INDEX`, а loss считается только на ответе;
- `collate_fn` для батча.

### 3. `model.py`

Нужно реализовать:

- adapter из hidden states vision encoder в размерность LLM embeddings;
- функцию вставки visual embeddings на позиции `<image>`-токенов;
- forward pass с loss;
- generate/inference wrapper;
- корректную заморозку vision encoder и LLM в Track A/B.

### 4. `train.py`

Нужно реализовать:

- загрузку YAML-конфига;
- создание датасета, processor, модели, optimizer;
- gradient accumulation;
- `fast_train` режим для smoke-тестов;
- сохранение adapter/checkpoint;
- проверку, что loss конечный.

### 5. `benchmark.py`

Нужно реализовать:

- построение benchmark prompt;
- запуск `generate`;
- извлечение ответа `A/B/C/D` или normalised text answer;
- подсчёт accuracy по subject и overall.


## Оценивание

Задание оценивается в **10 баллов**. Основные критерии вынесены в файл [`GRADING.md`](GRADING.md). Базовые 10 баллов можно получить без GPU и без скачивания MathVista: нужно реализовать пайплайн, пройти public tests и заполнить отчёт.

## Ограничения

- Все параметры должны задаваться через YAML-конфиги.
- Код должен быть воспроизводимым: seed, config, логирование.
- В обязательной части не требуется скачивать внешние датасеты.
- MathVista нужен только для quality evaluation / bonus.
- Не коммитьте `assets/mathvista_testmini/`, checkpoints и другие большие артефакты.

## Профильные источники

- MathVista — visual mathematical reasoning benchmark;
- MAVIS — mathematical visual instruction tuning datasets;
- MATH-Vision — visual math tasks from real math competitions;
- We-Math — benchmark with hierarchical visual mathematical reasoning concepts.

В student-template включены только toy/medium synthetic datasets. Реальные источники подключаются отдельно.
