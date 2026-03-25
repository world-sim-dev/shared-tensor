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
- manual two-process deployment as the primary production path
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

## Example: Manual Two-Process Deployment

Production should prefer two explicitly started processes: one server process that owns CUDA objects, and one or more client processes that reopen them through torch IPC.

See [examples/model_service.py](./examples/model_service.py) for endpoint definitions.

Server process:

```python
from shared_tensor import SharedTensorProvider, SharedTensorServer

provider = SharedTensorProvider(execution_mode="server")

@provider.share(execution="task", managed=True, concurrency="serialized", cache_format_key="model:{hidden_size}")
def load_model(hidden_size: int = 4):
    ...

server = SharedTensorServer(provider)
server.start(blocking=True)
```

Client process:

```python
import torch

from shared_tensor import SharedObjectHandle, SharedTensorClient

client = SharedTensorClient()
x = torch.ones(1, 4, device="cuda")
result = client.call("load_model", hidden_size=4)
if isinstance(result, SharedObjectHandle):
    with result as handle:
        y = handle.value(x)
```

This keeps the contract explicit:

```text
server process                      client process
------------------------------      ------------------------------
owns CUDA allocations               issues local UDS RPC requests
executes endpoint functions         reopens CUDA objects via torch IPC
manages cache and refcounts         releases managed handles explicitly
```

## Example: Same Code, Two Processes

See [examples/zero_branch_env.py](./examples/zero_branch_env.py). This is a convenience mode for environments that want one file and environment-controlled behavior.

```bash
SHARED_TENSOR_ENABLED=1 SHARED_TENSOR_ROLE=server python demo.py
SHARED_TENSOR_ENABLED=1 python demo.py
```

What changes is only the environment:

```text
same code

server process                      client process
------------------------------      ------------------------------
provider auto-starts local thread   provider builds client wrappers
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
SharedTensorProvider(server_startup_timeout=30.0)
provider.get_runtime_info()
```

Non-blocking provider autostart runs the UDS server in a background thread inside the current process.

`execution_mode="auto"` behaves as follows:
- disabled: local mode
- enabled + `SHARED_TENSOR_ROLE=server`: auto-start a local background server thread and execute endpoints locally
- enabled + role unset: build client wrappers

For production deployment, prefer explicit `SharedTensorServer(...).start(blocking=True)` in a dedicated server process.

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
