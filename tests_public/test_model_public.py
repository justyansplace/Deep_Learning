from __future__ import annotations

import torch

from hw.model import VisionToTextAdapter, merge_visual_embeddings


def test_adapter_output_shape() -> None:
    adapter = VisionToTextAdapter(vision_hidden_size=8, text_hidden_size=16, num_image_tokens=4)
    x = torch.randn(2, 6, 8)
    y = adapter(x)
    assert y.shape == (2, 4, 16)
    assert torch.isfinite(y).all()


def test_merge_visual_embeddings_exact_positions() -> None:
    image_token_id = 99
    input_ids = torch.tensor([
        [10, 99, 99, 20],
        [99, 30, 99, 40],
    ])
    input_embeds = torch.zeros(2, 4, 3)
    visual = torch.tensor([
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
    ])
    merged = merge_visual_embeddings(input_embeds, input_ids, visual, image_token_id)
    assert torch.allclose(merged[0, 1], visual[0, 0])
    assert torch.allclose(merged[0, 2], visual[0, 1])
    assert torch.allclose(merged[1, 0], visual[1, 0])
    assert torch.allclose(merged[1, 2], visual[1, 1])
    assert torch.allclose(merged[0, 0], torch.zeros(3))
