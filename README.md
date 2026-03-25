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
- auto mode is gated by `SHARED_TENSOR_ENABLED=1`
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

## Enabling Auto Mode

`SharedTensorProvider()` now defaults to safe local mode unless you explicitly enable shared-tensor behavior.

Global default:

```bash
export SHARED_TENSOR_ENABLED=1
```

Per-provider override:

```python
provider = SharedTensorProvider(enabled=True)
provider = SharedTensorProvider(enabled=False)
provider = SharedTensorProvider(enabled=None)
```

`enabled=None` means do not override and keep using the environment variable.

Then `execution_mode="auto"` behaves like this:

- `enabled=False`: provider stays in local mode
- `enabled=True` and `SHARED_TENSOR_ROLE=server`: auto-start server and execute locally on the server side
- `enabled=True` and no role set: provider becomes a client wrapper
- `enabled=None`: fall back to `SHARED_TENSOR_ENABLED`

This makes accidental opt-in much less likely in scripts that import shared endpoints but did not intend to start RPC behavior.

## Example 1: Zero-Branch Auto Mode

See [examples/zero_branch_env.py](./examples/zero_branch_env.py).

One file, two processes, no branch in user code:

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
SHARED_TENSOR_ENABLED=1 SHARED_TENSOR_ROLE=server python demo.py
```

Run process B as the client with the exact same file:

```bash
SHARED_TENSOR_ENABLED=1 python demo.py
```

Equivalent stepwise form:

```bash
export SHARED_TENSOR_ENABLED=1
SHARED_TENSOR_ROLE=server python demo.py
python demo.py
```

Behavior:

- `SHARED_TENSOR_ENABLED=1` enables shared-tensor auto behavior for providers that keep `enabled=None`
- `SHARED_TENSOR_ROLE=server` makes the provider auto-start a background localhost daemon
- in the server process, shared functions still execute locally
- in the client process, the same function names become RPC wrappers
- no `SHARED_TENSOR_HOST` is used; transport is fixed to `127.0.0.1`
- the final port is `SHARED_TENSOR_BASE_PORT + current_cuda_device_index`

Why this works:

```text
same code file

Process A                               Process B
SHARED_TENSOR_ENABLED=1                 SHARED_TENSOR_ENABLED=1
SHARED_TENSOR_ROLE=server               SHARED_TENSOR_ROLE unset
----------------------------------      ----------------------------------
provider.share(...)                     provider.share(...)
provider auto-starts localhost daemon   provider builds RPC wrappers
shared fn executes locally              shared fn becomes RPC call

load_model(...)                         load_model(...)
  -> local CUDA model                     -> JSON-RPC to localhost daemon
identity(x)                              -> receives CUDA IPC-backed result
  -> local tensor return
```

Use this mode when you want the cleanest operator experience: one script, one env var difference, server side stays local, client side becomes remote automatically.

## Example 2: Fast Tensor Transform

See [examples/model_service.py](./examples/model_service.py).

```python
@provider.share(execution="direct", cache=False)
def scale_tensor(tensor: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return tensor * factor
```

What happens on the wire:

```text
client tensor -> direct RPC -> server runs function immediately -> CUDA result back
```

Use this for cheap tensor math, lightweight preprocessing, and request-scoped outputs.

Recommended combination:

- `execution="direct"`
- `cache=False`
- `managed=False`
- `concurrency="parallel"`

## Example 3: Reusable Model Service

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

What happens when two clients ask for the same model key:

```text
Client A                      Server                         Client B
------------------------      ------------------------       ------------------------
call("load_model", k) -----> cache miss                    call("load_model", k)
                              build object once              -------------> same key in flight
                              object_id = obj-123                         wait on same future
<-------------------------     return handle(obj-123)       <------------- return handle(obj-123)

release(obj-123) ---------->  refcount 2 -> 1
release(obj-123) ------------------------------------------> refcount 1 -> 0 -> destroy
```

Use this for big reusable models. The important mix is:

- `execution="task"`
- `managed=True`
- `concurrency="serialized"`
- `singleflight=True`
- explicit `cache_format_key`

`managed=True` gives explicit lifecycle control. `cache_format_key` turns the endpoint into a model registry. `singleflight=True` ensures duplicate in-flight loads collapse to one build.

## Example 4: Fire-And-Poll Warmup

This is the same task-backed endpoint style, but the caller chooses async use:

```python
task_id = load_model.submit(hidden_size=8192)
model_handle = provider.wait_for_task(task_id)
```

Runtime shape:

```text
submit now -> task queue -> slow build on server -> poll later -> consume handle/result
```

Use this when the build is slow enough that the caller should not block immediately.

## Example 5: Serialized Fragile Path

```python
@provider.share(execution="task", concurrency="serialized", cache=False, singleflight=False)
def compact_memory(tensor: torch.Tensor) -> torch.Tensor:
    ...
```

Execution model:

```text
request A -> lock -> run -> unlock
request B -> wait -> lock -> run -> unlock
```

Use this for GPU-heavy paths that must not overlap with themselves.

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
  - use this for fast tensor transforms
- `execution="task"`
  - sync calls still block, but they block on the task system
  - use this for slow construction, warmup, and reusable model loading
- `concurrency="parallel"`
  - multiple server executions may run at once
- `concurrency="serialized"`
  - only one execution of that endpoint runs at a time
- `singleflight=True`
  - identical in-flight cache keys collapse to one execution
  - this is the recommended model-loading default

## Scenario Map

- Fast tensor transform:
  use `execution="direct"`, `cache=False`, `managed=False`
- Slow model construction:
  use `execution="task"`, `managed=True`, `concurrency="serialized"`
- Reusable model registry:
  add stable `cache_format_key` and keep `singleflight=True`
- Background warmup:
  keep endpoint as task-backed and use `.submit(...)`
- Fragile non-overlapping GPU path:
  use `concurrency="serialized"`
- Fresh per-request work:
  disable cache and usually disable singleflight

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
