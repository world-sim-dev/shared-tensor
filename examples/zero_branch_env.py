"""One file, two processes, zero branching, native CUDA IPC."""

from __future__ import annotations
import logging

logging.basicConfig()

import torch

from shared_tensor import SharedTensorProvider, SharedObjectHandle

provider = SharedTensorProvider(verbose_debug=True)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")


@provider.share(execution="task", managed=True, concurrency="serialized", cache_format_key="model:{hidden_size}")
def load_model(hidden_size: int = 4) -> torch.nn.Module:
    _require_cuda()
    layer = torch.nn.Linear(hidden_size, 2, device="cuda")
    with torch.no_grad():
        layer.weight.fill_(1.5)
        layer.bias.fill_(0.25)
    return layer

@provider.share(execution="task", managed=False, concurrency="serialized", cache_format_key="model:{hidden_size}")
def load_model2(hidden_size: int = 4) -> torch.nn.Module:
    _require_cuda()
    layer = torch.nn.Linear(hidden_size, 2, device="cuda")
    with torch.no_grad():
        layer.weight.fill_(1.5)
        layer.bias.fill_(0.25)
    return layer


@provider.share
def get_shared_tensor():
    return torch.range(1, 4, device="cuda")


if __name__ == "__main__":
    x = torch.ones(1, 4, device="cuda")
    result = load_model(hidden_size=4)
    print(f"model loaded from {provider.execution_mode}, type {type(result)}")
    if isinstance(result, SharedObjectHandle):
        result = result.value
    y = result(x)
    print(f"y from {provider.execution_mode}, value {y},  enter to continue")

    result2 = load_model2(hidden_size=4)
    y2 = result2(x)
    input(f"model2 loaded from {provider.execution_mode}, type {type(result)}")
    input(f"y2 from {provider.execution_mode}, value {y}")

    shared = get_shared_tensor()
    input(f"shared tensor from {provider.execution_mode}, value {shared}, type {type(shared)}, enter to continue")

    if provider.execution_mode == "server":
        # change first to 4
        shared[0] = 4

    input(f"shared tensor changed from {provider.execution_mode}, value {shared}, type {type(shared)}, enter to continue")
