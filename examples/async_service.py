"""Canonical task-oriented service definition for CUDA torch IPC."""

from __future__ import annotations

import time

import torch

from shared_tensor import SharedTensorProvider

provider = SharedTensorProvider(execution_mode="server")


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")


@provider.share(execution="task")
def build_delayed_model(delay: float = 0.1) -> torch.nn.Module:
    _require_cuda()
    time.sleep(delay)
    return torch.nn.Linear(8, 4, device="cuda")


@provider.share
def clone_tensor_async(tensor: torch.Tensor) -> torch.Tensor:
    _require_cuda()
    return tensor.clone()
