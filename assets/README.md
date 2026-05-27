# Assets

В репозитории лежат только маленькие датасеты, которые можно безопасно хранить в GitHub.

```text
toy_math_vqa       — маленький синтетический набор для public tests и CPU smoke
math_vqa_medium    — синтетический набор побольше для локальной практики и отчёта
```

Внешние benchmark-датасеты, например MathVista, не лежат в GitHub. Их нужно скачивать отдельно:

```bash
python -m pip install -e ".[ml]"
python scripts/prepare_mathvista_testmini.py --out assets/mathvista_testmini --max-samples 200
```

Папка `assets/mathvista_testmini/` добавлена в `.gitignore`.
