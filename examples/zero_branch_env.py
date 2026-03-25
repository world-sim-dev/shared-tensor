"""One file, two processes, zero branching, native CUDA IPC."""

from __future__ import annotations

import torch

from shared_tensor import SharedObjectHandle, SharedTensorProvider

provider = SharedTensorProvider()


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


if __name__ == "__main__":
    x = torch.ones(1, 4, device="cuda")
    result = load_model(hidden_size=4)
    if isinstance(result, SharedObjectHandle):
        with result as handle:
            y = handle.value(x)
    else:
        y = result(x)

    print("y=", y)
