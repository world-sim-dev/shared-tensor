# Shared Tensor

`shared_tensor` is a narrow library for one job: sharing CUDA `torch.Tensor` and CUDA `torch.nn.Module` objects across processes on the same host and the same GPU with native PyTorch IPC semantics.

The control plane is a local Unix Domain Socket RPC channel. The data plane is native `torch` CUDA IPC serialization. CPU fallback is intentionally out of scope.

## Scope

Supported:
- same-host trusted processes
- same-GPU CUDA tensors and modules
- explicit endpoint registration
- sync `call` and task-backed `submit`
- managed object handles with explicit release
- server-side caching, `cache_format_key`, and singleflight
- zero-branch auto mode gated by `SHARED_TENSOR_ENABLED=1`

Not supported:
- CPU tensor or CPU module transport
- generic Python object RPC
- cross-host transport
- `mps`
- implicit device migration

## Install

Use Python `3.10+` and a CUDA-enabled PyTorch build.

```bash
pip install shared-tensor
```

For local development:

```bash
conda create -y -n shared-tensor-dev python=3.11
conda activate shared-tensor-dev
pip install -e ".[dev,test]"
```

## Example: Same Code, Two Processes

See [examples/zero_branch_env.py](./examples/zero_branch_env.py).

```python
import torch

from shared_tensor import SharedObjectHandle, SharedTensorProvider

provider = SharedTensorProvider()


@provider.share(
    execution="task",
    managed=True,
    concurrency="serialized",
    cache_format_key="model:{hidden_size}",
)
def load_model(hidden_size: int = 4) -> torch.nn.Module:
    return torch.nn.Linear(hidden_size, 2, device="cuda")


x = torch.ones(1, 4, device="cuda")
result = load_model(hidden_size=4)
if isinstance(result, SharedObjectHandle):
    with result as handle:
        y = handle.value(x)
else:
    y = result(x)
```

Server process:

```bash
SHARED_TENSOR_ENABLED=1 SHARED_TENSOR_ROLE=server python demo.py
```

Client process with the exact same file:

```bash
SHARED_TENSOR_ENABLED=1 python demo.py
```

What changes is only the environment:

```text
same code

server process                      client process
------------------------------      ------------------------------
provider auto-starts UDS daemon     provider builds client wrappers
shared function runs locally        shared function becomes RPC call
CUDA object stays on same GPU       CUDA object is reopened via torch IPC
```

## Example: Reusable Model Registry

See [examples/model_service.py](./examples/model_service.py).

```python
@provider.share(
    execution="task",
    managed=True,
    concurrency="serialized",
    cache_format_key="model:{input_dim}:{output_dim}",
)
def load_linear_model(input_dim: int = 16, output_dim: int = 4) -> torch.nn.Module:
    ...
```

Recommended settings for expensive reusable models:
- `execution="task"`
- `managed=True`
- `concurrency="serialized"`
- `singleflight=True`
- explicit `cache_format_key`

This gives one build per cache key, shared handles for identical requests, and explicit release semantics.
Task submission uses the same server-side cache as sync `call`: repeated `submit` for the same cache key reuses the cached result instead of rebuilding the CUDA object.

## Example: Direct Tensor Path

See [examples/basic_service.py](./examples/basic_service.py).

```python
@provider.share(execution="direct", cache=False)
def echo_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor
```

Use this for short-lived request-scoped CUDA transforms. The main production path is still task-backed model construction.

## Configuration

`SharedTensorProvider()` defaults to safe local mode unless shared-tensor behavior is explicitly enabled.

Environment gate:

```bash
export SHARED_TENSOR_ENABLED=1
```

Per-provider override:

```python
SharedTensorProvider(enabled=True)
SharedTensorProvider(enabled=False)
SharedTensorProvider(enabled=None)
```

Provider runtime controls:

```python
SharedTensorProvider(server_process_start_method="fork")
SharedTensorProvider(server_startup_timeout=30.0)
provider.get_runtime_info()
```

Use `server_process_start_method="fork"` when you explicitly want POSIX fork behavior.
Leave it as `None` to let the library choose a safer default for the current entrypoint.

`execution_mode="auto"` behaves as follows:
- disabled: local mode
- enabled + `SHARED_TENSOR_ROLE=server`: auto-start local server and execute endpoints locally
- enabled + role unset: build client wrappers

Socket selection is per CUDA device:
- base path comes from `SHARED_TENSOR_BASE_PATH` or `/tmp/shared-tensor`
- runtime socket path is `<base_path>-<device_index>.sock`
- `device_index=None` means probe lazily from the current CUDA device when needed

## Payload Contract

Allowed result payloads:
- CUDA `torch.Tensor`
- CUDA `torch.nn.Module`

Allowed call payloads:
- CUDA tensors and modules
- scalar control values in `args` and `kwargs`
- `tuple`, `list`, and `dict[str, ...]` wrappers
- empty `args` and `kwargs` through the control path

Rejected:
- CPU tensors or modules
- plain Python result payloads
- `mps`

## Managed Objects

When `managed=True`, the client receives a `SharedObjectHandle`.

```python
handle = load_model(hidden_size=4096)
with handle as model_handle:
    y = model_handle.value(x)
```

You can also release explicitly:

```python
handle.release()
```

Use managed mode for cached models or other reusable long-lived CUDA objects.

## Runtime Introspection

`client.get_server_info()` now returns readiness and process metadata in addition to endpoint and capability data.
In client mode, `provider.get_runtime_info()` wraps that into a provider-oriented view.

```python
info = provider.get_runtime_info()
# execution_mode, server_socket_path, server_running, server_ready, server_info...
```

## Testing

Default suite:

```bash
python -m pytest -m "not gpu"
```

GPU suite:

```bash
python -m pytest -m gpu
```
