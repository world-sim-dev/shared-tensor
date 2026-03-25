# Shared Tensor

`shared_tensor` is a localhost-only RPC layer for one thing: passing CUDA `torch.Tensor` and CUDA `torch.nn.Module` objects between processes on the same machine and the same GPU with native PyTorch IPC semantics.

## What It Supports

- same-host, trusted-process deployment
- same-GPU CUDA object handoff
- native `torch` tensors and modules
- explicit endpoint registration
- one endpoint model with sync `call` and async `submit`
- task-backed slow object construction
- endpoint-level serialization and cache-key singleflight
- zero-branch auto mode driven by `SHARED_TENSOR_ROLE`
- port routing by `base_port + cuda_device_index`

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

## Zero-Branch Example

One file:

```python
import torch

from shared_tensor import SharedObjectHandle, SharedTensorProvider

provider = SharedTensorProvider()


@provider.share(execution="task", managed=True, cache_format_key="model:{hidden_size}")
def load_model(hidden_size: int = 4) -> torch.nn.Module:
    return torch.nn.Linear(hidden_size, 2, device="cuda")


@provider.share(cache=False)
def identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


if __name__ == "__main__":
    x = torch.ones(1, 4, device="cuda")
    maybe_handle = load_model(hidden_size=4)
    if isinstance(maybe_handle, SharedObjectHandle):
        with maybe_handle as handle:
            y = handle.value(x)
    else:
        y = maybe_handle(x)
    echoed = identity(x)
    print(y, echoed)
```

Run process A as the auto server:

```bash
SHARED_TENSOR_ROLE=server python demo.py
```

Run process B as the client with the exact same file:

```bash
python demo.py
```

Behavior:

- `SHARED_TENSOR_ROLE=server` makes the provider auto-start a background localhost daemon
- in the server process, shared functions still execute locally
- in the client process, the same function names become RPC wrappers
- no `SHARED_TENSOR_HOST` is used; transport is fixed to `127.0.0.1`
- the final port is `SHARED_TENSOR_BASE_PORT + current_cuda_device_index`

## Endpoint Semantics

Each endpoint is registered once and then supports two client-side call styles.

- `fn(...)` or `provider.call(name, ...)`
  - synchronous
  - blocks until the result is ready
- `fn.submit(...)` or `provider.submit(name, ...)`
  - asynchronous
  - returns a task id

Endpoint options:

- `execution="direct"`
  - sync calls run the function directly on the server
  - best for fast tensor transforms
- `execution="task"`
  - sync calls still block, but they block on the task system
  - async submit is the natural path for slow model construction
- `concurrency="parallel"`
  - multiple server executions may run at once
- `concurrency="serialized"`
  - only one execution of that endpoint runs at a time
- `singleflight=True`
  - identical in-flight cache keys collapse to one execution
  - this is the recommended model-loading default

## Common Scenarios

### 1. Fast Tensor Transform

Use this for cheap operations such as clone, view-like transforms, elementwise scaling, or lightweight preprocessing.

```python
@provider.share(execution="direct", cache=False)
def scale_tensor(tensor: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return tensor * factor
```

Recommended combination:

- `execution="direct"`
- `cache=False`
- `managed=False`
- `concurrency="parallel"`

Why:

- direct execution has the lowest overhead
- these calls are request-scoped, so caching is usually wrong
- parallel execution is usually fine because the work is short-lived

### 2. Slow Model Construction

Use this for loading or building a CUDA model that may take hundreds of milliseconds or multiple seconds.

```python
@provider.share(
    execution="task",
    managed=True,
    concurrency="serialized",
    cache_format_key="model:{hidden_size}",
)
def load_model(hidden_size: int) -> torch.nn.Module:
    return torch.nn.Linear(hidden_size, 2, device="cuda")
```

Recommended combination:

- `execution="task"`
- `managed=True`
- `concurrency="serialized"`
- `singleflight=True`
- explicit `cache_format_key`

Why:

- task mode gives you both blocking sync calls and true async submission
- managed handles let the client release the remote object explicitly
- serialized execution avoids multiple concurrent heavy loads on one GPU
- singleflight prevents duplicate in-flight construction for the same model key

### 3. Reusable Shared Model Service

Use this when the model should be built once and reused by many client calls.

```python
@provider.share(
    execution="task",
    managed=True,
    cache_format_key="model:{model_name}:{dtype}",
)
def load_model(model_name: str, dtype: str) -> torch.nn.Module:
    ...
```

Recommended combination:

- `cache=True`
- `managed=True`
- `singleflight=True`
- explicit stable cache key

Why:

- caching makes the endpoint act like a model registry
- managed handles keep explicit lifecycle control
- stable cache keys prevent accidental duplication from argument shape changes

### 4. Fire-and-Poll Background Warmup

Use this when the caller should not block, for example prewarming a model or allocating a large reusable tensor in the background.

```python
task_id = load_model.submit(hidden_size=8192)
model_handle = provider.wait_for_task(task_id)
```

Recommended combination:

- endpoint uses `execution="task"`
- caller uses `.submit(...)`
- optionally add `managed=True` for long-lived objects

Why:

- the endpoint stays declarative
- the caller decides whether to block now or poll later

### 5. Strictly Non-Reusable Per-Request Work

Use this when every request must create a fresh result and reuse is wrong.

```python
@provider.share(execution="task", cache=False, singleflight=False)
def build_request_tensor(template: torch.Tensor) -> torch.Tensor:
    return template.clone()
```

Recommended combination:

- `cache=False`
- `singleflight=False`
- choose `execution="direct"` or `execution="task"` based on runtime cost

Why:

- disabling cache avoids cross-request reuse
- disabling singleflight ensures independent requests stay independent

### 6. Endpoint That Must Run One At A Time

Use this when the endpoint mutates shared state, temporarily spikes memory, or must not overlap with itself.

```python
@provider.share(execution="task", concurrency="serialized", cache=False, singleflight=False)
def compact_memory(tensor: torch.Tensor) -> torch.Tensor:
    ...
```

Recommended combination:

- `execution="task"`
- `concurrency="serialized"`
- usually `cache=False`

Why:

- serialization is endpoint-wide, not just per cache key
- useful for fragile GPU-heavy paths where overlap is unsafe or wasteful

## Parameter Guide

### `execution`

- `"direct"`: run immediately on the server request path
- `"task"`: run via the task system

Choose `direct` for fast transforms. Choose `task` for slow construction, warmup, or any path where async submit matters.

### `cache`

- `True`: reuse previously computed results
- `False`: every call computes a new result

Use `True` for reusable models and long-lived shared objects. Use `False` for request-scoped tensors.

### `cache_format_key`

A format string built from bound function arguments.

Example:

```python
@provider.share(cache_format_key="model:{name}:{dtype}")
def load_model(name: str, dtype: str) -> torch.nn.Module:
    ...
```

Use it whenever the default function-level cache identity is too coarse.

### `managed`

- `False`: return the value directly
- `True`: return a `SharedObjectHandle` on the client side

Use `managed=True` for long-lived models or tensors that need deterministic release.

### `concurrency`

- `"parallel"`: allow multiple executions at once
- `"serialized"`: only one execution of that endpoint runs at a time

Use `serialized` for heavy model loads, stateful operations, or anything that should not overlap.

### `singleflight`

- `True`: same in-flight cache key shares one execution
- `False`: concurrent identical requests execute independently

This matters most when `cache=True` and model construction is expensive.

## Port Routing

This repo is same-host and same-GPU only, so the effective port is derived from the CUDA device index.

- `base_port` is the base port
- runtime port is `base_port + device_index`
- `SHARED_TENSOR_BASE_PORT` overrides the base port from the environment
- `device_index=None` means probe lazily when the client/server is actually created
- you can force a specific device with `SharedTensorProvider(device_index=3)`

Example:

```python
provider = SharedTensorProvider(base_port=2537, device_index=1)
```

This binds or connects to port `2538`.

## Cache Behavior

Caching is on by default.

- default `cache_format_key` is the function `__qualname__`
- that means repeated calls to the same shared function reuse the first result unless you override the key
- set `cache=False` for endpoints that should never reuse results
- set `cache_format_key="{arg_name}"` or similar to build cache keys from bound function arguments
- `singleflight=True` deduplicates concurrent in-flight work for the same cache key

Example:

```python
@provider.share(execution="task", cache_format_key="{hidden_size}")
def load_model(hidden_size: int = 4) -> torch.nn.Module:
    return torch.nn.Linear(hidden_size, 2, device="cuda")
```

## Managed Object Release

For long-lived shared CUDA objects, use `managed=True`. Client-side calls return a `SharedObjectHandle` with `.value`, `.object_id`, and `.release()`.

```python
with load_model(hidden_size=8) as handle:
    model = handle.value
    y = model(torch.ones(1, 8, device="cuda"))
```

Rules:

- `managed=True` only changes remote/client results; server-mode direct calls still return the local object itself
- managed endpoints still use cache by default
- when cache hits, repeated calls reuse the same remote object id and increment its server-side refcount
- call `handle.release()` when you want deterministic teardown
- `client.release_many([...])` and `client.get_object_info(object_id)` are available for batch release and inspection

## Manual Mode

Manual mode is programmatic only.

```python
from shared_tensor import SharedTensorProvider, SharedTensorServer
from shared_tensor.utils import resolve_runtime_port

provider = SharedTensorProvider(execution_mode="server", base_port=2537)
server = SharedTensorServer(
    provider,
    port=resolve_runtime_port(provider.base_port, provider.device_index),
)
server.start(blocking=True)
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

## Operational Notes

- This library assumes a trusted same-host environment.
- The server process must be a separate process from the client when using CUDA IPC.
- If you need cross-machine transport or CPU object RPC, use a different tool.

## Repo Notes

- `CLAUDE.md` captures repo maintenance rules.
- `examples/zero_branch_env.py` shows the main production-oriented example.
- `examples/model_service.py` shows explicit model handoff.
