"""Example model handoff pattern for same-host same-GPU torch IPC."""

from __future__ import annotations

import torch

from shared_tensor import SharedTensorProvider

provider = SharedTensorProvider(execution_mode="client")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")


@provider.share(name="load_linear_model")
def load_linear_model(input_dim: int = 16, output_dim: int = 4) -> torch.nn.Module:
    _require_cuda()
    layer = torch.nn.Linear(input_dim, output_dim, device="cuda")
    with torch.no_grad():
        layer.weight.fill_(0.75)
        layer.bias.zero_()
    return layer


@provider.share(name="scale_tensor")
def scale_tensor(tensor: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    _require_cuda()
    return tensor * factor
