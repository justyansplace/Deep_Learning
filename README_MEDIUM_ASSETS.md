# Patch: medium public dataset для VLM Math Homework

Этот patch добавляет в шаблон более весомый, но всё ещё лёгкий synthetic math-VQA набор:

```text
assets/math_vqa_medium/
  manifest.jsonl
  images/*.png
configs/
  track_a_cpu_medium.yaml
  track_b_small_gpu_medium.yaml
```

Как использовать:

1. Распакуйте архив в корень `vlm-math-hw-template`.
2. Сделайте commit + push.
3. В LMS напишите, что обязательная часть всё ещё не требует внешних датасетов.
4. Для public tests можно оставить старый toy-набор; medium-набор использовать для локального smoke/dev запуска и отчёта.

Размер набора:

- train: 180
- dev: 40
- test_public: 40

Он не заменяет hidden tests преподавателя и не требует GPU.
