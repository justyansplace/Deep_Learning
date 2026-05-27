from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def toy_manifest() -> Path:
    return Path("assets/toy_math_vqa/manifest.jsonl")
