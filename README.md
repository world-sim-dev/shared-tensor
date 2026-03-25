# Shared Tensor

`shared_tensor` is a localhost-only RPC layer for one thing: passing CUDA `torch.Tensor` and CUDA `torch.nn.Module` objects between processes on the same machine and the same GPU with native PyTorch IPC semantics.

## What It Supports

- same-host, trusted-process deployment
- same-GPU CUDA object handoff
- native `torch` tensors and modules
- explicit endpoint registration
- sync calls and async task polling

## What It Does Not Support

- CPU tensor transport
- generic Python object RPC
- cross-machine transport
- macOS `mps`
- silent CPU fallback or implicit device migration

## Payload Contract

Allowed payloads:

- CUDA `torch.Tensor`
- CUDA `torch.nn.Module`
- `tuple`, `list`, and `dict[str, ...]` containers built from those values for `args` and `kwargs`
- empty `args` / `kwargs` through the control path for no-argument calls only

Rejected payloads:

- CPU tensors and CPU modules
- plain Python values such as `int`, `str`, `dict`, and `list`
- `mps` tensors and modules

## Install

Use Python `3.10+` and a CUDA-enabled PyTorch build.

```bash
pip install shared-tensor
```

For development:

```bash
conda create -y -n shared-tensor-dev python=3.11
conda activate shared-tensor-dev
pip install -e ".[dev,test]"
```

## Typical Example

Provider process:

```python
import torch

from shared_tensor import SharedTensorProvider

provider = SharedTensorProvider(execution_mode="server")


@provider.share(name="load_model")
def load_model() -> torch.nn.Module:
    return torch.nn.Linear(4, 2, device="cuda")


@provider.share(name="identity")
def identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor
```

Run the server:

```bash
shared-tensor-server --provider my_service:provider --host 127.0.0.1 --port 2537
```

Consumer process:

```python
import torch

from shared_tensor import SharedTensorClient

with SharedTensorClient(port=2537) as client:
    model = client.call("load_model")
    x = torch.ones(1, 4, device="cuda")
    y = model(x)

    shared = client.call("identity", x)
```

## Test Matrix

Default local run:

```bash
python -m pytest -m "not gpu"
```

CUDA run:

```bash
python -m pytest -m gpu
```

`skipped` means the test was intentionally not run because its precondition was missing. In this repo that usually means a `gpu` test was executed on a machine where `torch.cuda.is_available()` was false. It is not a failure.

Current validation target:

- local non-GPU suite passes
- H100 CUDA suite passes

## Operational Notes

- This library assumes a trusted same-host environment.
- The server process must be a separate process from the client when using CUDA IPC.
- If you need cross-machine transport or CPU object RPC, use a different tool.

## Repo Notes

- `CLAUDE.md` captures repo maintenance rules.
- `examples/basic_service.py` shows the minimal sync flow.
- `examples/model_service.py` shows model handoff.
