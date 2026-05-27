# Hardware-треки

## Track A — CPU-only

Подходит студентам без GPU. Это основной обязательный трек.

### Требования

```text
pytest -q tests_public проходит
train.py запускается в --fast-train режиме
benchmark.py запускается на toy-dev
report.md заполнен
```

### Не требуется

```text
полное обучение VLM
качество на MathVista/MATH-Vision/We-Math
LoRA
чекпойнт большой модели
```

### Оценивание

```text
dataset.py + benchmark.py        20%
processor.py                     25%
model.py                         25%
train.py                         15%
воспроизводимость и report.md    15%
```

## Track B — Small GPU

Подходит для домашней видеокарты 6–12 GB VRAM.

### Требования

```text
все требования Track A
adapter-only обучение на маленьком math subset
сохранён artifacts/adapter.pt или artifacts/adapter.safetensors
loss конечный и не взрывается
```

### Рекомендуемые параметры

```text
num_tiles: 1
image_size: 224
max_length: 384 или 512
local_batch_size: 1
global_batch_size: 8
LoRA: off
vision encoder: frozen
LLM: frozen
```

## Track C — A100-20GB

Подходит для 1/4 A100 или похожего GPU с примерно 20 GB VRAM.

### Требования

```text
все требования Track A
alignment/pretrain adapter
SFT adapter + LoRA
benchmark на hidden math-dev или public math-dev
report с ресурсами и ошибками модели
```

### Рекомендуемые параметры

```text
num_tiles: 4
image_size: 224
max_length: 768
LoRA rank: 32 или 64
LoRA rank 256: только bonus/leaderboard
vision encoder: frozen
LLM base: frozen
```

## Почему так

CPU-only студенты могут получить максимум за корректную реализацию, а студенты с GPU могут сделать более глубокий ML-проект и сравнить качество.
