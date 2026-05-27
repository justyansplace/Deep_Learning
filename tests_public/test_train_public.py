from __future__ import annotations

import torch

from hw.train import load_config, set_seed, train_one_step


class TinyTrainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, batch):
        pred = self.linear(batch["x"])
        loss = torch.nn.functional.mse_loss(pred, batch["y"])
        return {"loss": loss}


def test_load_config() -> None:
    cfg = load_config("configs/track_a_cpu.yaml")
    assert cfg["track"] == "A_cpu_only"
    assert cfg["trainer"]["max_steps"] == 3


def test_set_seed_reproducible() -> None:
    set_seed(123)
    a = torch.randn(3)
    set_seed(123)
    b = torch.randn(3)
    assert torch.allclose(a, b)


def test_train_one_step_updates_weights() -> None:
    model = TinyTrainModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.linear.weight.detach().clone()
    batch = {"x": torch.randn(4, 2), "y": torch.randn(4, 1)}
    loss = train_one_step(model, batch, opt)
    assert isinstance(loss, float)
    assert loss >= 0
    assert not torch.allclose(before, model.linear.weight.detach())
