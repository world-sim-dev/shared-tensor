"""Canonical synchronous service definition for CUDA torch IPC."""

from __future__ import annotations

import torch

from shared_tensor import SharedTensorProvider

provider = SharedTensorProvider(execution_mode="client")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")


@provider.share(name="build_linear_model")
def build_linear_model() -> torch.nn.Module:
    _require_cuda()
    layer = torch.nn.Linear(4, 2, device="cuda")
    with torch.no_grad():
        layer.weight.fill_(1.25)
        layer.bias.fill_(0.5)
    return layer


@provider.share(name="echo_tensor")
def echo_tensor(tensor: torch.Tensor) -> torch.Tensor:
    _require_cuda()
    return tensor
