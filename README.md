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
- server-side caching, `cache_format_key`, singleflight, and explicit cache invalidation
- manual two-process deployment as the primary production path
- zero-branch auto mode gated by `SHARED_TENSOR_ENABLED=1`

Not supported:
- CPU tensor or CPU module transport
- generic Python object RPC
- cross-host transport
- `mps`
- implicit device migration

## Install

Use Python `3.9+`. Install a compatible PyTorch build first, then install `shared-tensor`.

```bash
pip install torch
pip install shared-tensor
```

For local development:

```bash
conda create -y -n shared-tensor-dev python=3.11
conda activate shared-tensor-dev
pip install -e ".[dev,test]"
```

If you want to share Hugging Face `transformers` models, install both `torch` and `transformers` in the server and client environments. `shared-tensor` no longer installs `torch` for you.

## Docs

Read the examples first, then the design notes:

- `docs/overview.md`
- `docs/patterns.md`
- `docs/architecture.md`
- `docs/lifecycle.md`
- `docs/diagrams.md`

## Example: Manual Two-Process Deployment

Production should prefer two explicitly started processes: one server process that owns CUDA objects, and one or more client processes that reopen them through torch IPC.

See [examples/model_service.py](./examples/model_service.py) for endpoint definitions.

The server-oriented example modules construct providers with explicit `execution_mode="server"` so importing the module already reflects the intended deployment role.

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

## Example: Transformers Models

`shared_tensor` also supports CUDA `transformers.PreTrainedModel` instances.

See:
- `examples/transformers_two_proc_demo.py`: minimal same-code two-process demo using `AutoModel`-style loading
- `examples/transformers_mutation_check.py`: proves client-side in-place parameter mutation is visible on the server
- `examples/transformers_ipc_benchmark.py`: measures reopen latency and client GPU memory delta

Usage:

```bash
TRANSFORMERS_MODEL_ROOT=/path/to/model-or-hf-cache/models--bert-base-uncased \
SHARED_TENSOR_ENABLED=1 SHARED_TENSOR_ROLE=server \
python examples/transformers_two_proc_demo.py

TRANSFORMERS_MODEL_ROOT=/path/to/model-or-hf-cache/models--bert-base-uncased \
SHARED_TENSOR_ENABLED=1 \
python examples/transformers_two_proc_demo.py
```

Notes:
- `TRANSFORMERS_MODEL_ROOT` may point either to a resolved local model directory or to a Hugging Face cache root like `models--...`; the example resolves the newest snapshot automatically
- `TRANSFORMERS_AUTO_CLASS` defaults to `AutoModel` and can be overridden to another `Auto*` class that exposes `from_pretrained`
- for custom `transformers` code paths, the library stages the required module source files before reopening the shared module on the client
- transport remains same-host same-GPU torch CUDA IPC; the client should not allocate a second full model copy just to reconstruct parameters
- in a fresh client Python process, the first reopen may still look slow because `transformers` import/module resolution is often much slower than the shared-tensor IPC restore path itself; a second reopen in the same process should be much faster

## Lifetime And Failure Contract

`shared_tensor` follows native PyTorch CUDA IPC semantics. It does not virtualize or harden producer lifetime.

Core assumption:
- the server process that owns the original CUDA allocation must stay alive while clients are still using reopened CUDA tensors or modules
- handle health checks can detect some stale-object conditions, but they do not remove the producer-liveness requirement

If the server exits, crashes, or is killed before the client is done with the shared CUDA object, behavior is no longer guaranteed by this library. Depending on PyTorch and CUDA runtime state, the client may see CUDA runtime errors, invalid resource handle failures, broken module execution, or process-level instability.

So the production contract is:
- client-side handles are only valid while the producer process remains alive
- `handle.release()` is explicit lifecycle cleanup, not durability
- this library does not promise survivability across producer death

Treat producer liveness as a hard requirement, not a soft optimization.

## Example: Same Code, Two Processes

See [examples/zero_branch_env.py](./examples/zero_branch_env.py). This is a convenience mode for environments that want one file and environment-controlled behavior.

Resolution rule:
- `SHARED_TENSOR_ENABLED` unset or false: provider stays local
- `SHARED_TENSOR_ENABLED=1` and `SHARED_TENSOR_ROLE=server`: provider resolves to server and auto-starts the thread-backed local server
- `SHARED_TENSOR_ENABLED=1` and role unset or `client`: provider resolves to client

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

## Example: Task Submission And Wait

See [examples/async_service.py](./examples/async_service.py).

```python
from shared_tensor import AsyncSharedTensorClient, SharedTensorProvider

provider = SharedTensorProvider(execution_mode="server")

@provider.share(execution="task")
def build_delayed_model(delay: float = 0.1):
    ...

client = AsyncSharedTensorClient()
task_id = client.submit("build_delayed_model", delay=0.1)
model = client.wait_for_task(task_id, timeout=30)
```

Use `SharedTensorProvider(execution="task")` for task-backed endpoints.
Use `AsyncSharedTensorClient` when you want a task-oriented waiting interface.

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
Managed object introspection now includes `created_at` and `last_accessed_at` timestamps through `get_object_info()`.

## Cache Invalidation

The library now exposes explicit cache invalidation instead of forcing process restarts when a cached object becomes stale.

```python
provider.invalidate_call_cache("load_model", hidden_size=4096)
provider.invalidate_endpoint_cache("load_model")
```

Client-side equivalents are also available:

```python
client.invalidate_call_cache("load_model", hidden_size=4096)
client.invalidate_endpoint_cache("load_model")
```

Use call-level invalidation when you want to evict one cache key.
Use endpoint-level invalidation when you want to drop all cached variants for the endpoint.
Invalidation removes cache lookup entries; it does not guarantee that already-issued client handles remain valid after producer death.

For cached `transformers` model endpoints, keep `cache=True` unless you explicitly want every request to rebuild and re-share the model.


## Handle Health Checks

Managed handles now carry the producer `server_id` and support lightweight liveness probes:

```python
handle = client.call("load_model", hidden_size=4096)
info = handle.get_object_info()
client.ensure_handle_live(handle)
```

If the producer no longer owns the object, `client.ensure_handle_live(handle)` raises `SharedTensorStaleHandleError`.
This is still advisory, not a durability guarantee: it helps detect stale handles earlier, but it cannot make producer death safe.

## Runtime Introspection

`client.get_server_info()` now returns readiness, stable `server_id`, cache/task counters, and process metadata in addition to endpoint and capability data.
In client mode, `provider.get_runtime_info()` wraps that into a provider-oriented view.
`AsyncSharedTensorClient` exposes the same runtime, cache invalidation, release, and handle-health helper methods as `SharedTensorClient`; the async surface is task-oriented, not capability-reduced.

```python
info = provider.get_runtime_info()
# execution_mode, server_socket_path, server_running, server_ready, server_info...
```

## Logging

`shared_tensor` now installs a default package logger on import and enables detailed logs by default.

- default level: `INFO`
- logger name: `shared_tensor`
- override level with `SHARED_TENSOR_LOG_LEVEL`, for example `INFO`, `WARNING`, or `ERROR`
- pass `verbose_debug=False` to `SharedTensorProvider`, `SharedTensorClient`, `AsyncSharedTensorClient`, or `SharedTensorServer` if you want to suppress detailed request-level logs

## Client Retry And Timeout Defaults

The client now retries initial connection setup for up to `60s` when the server socket is not ready yet, covering the common server-startup race where the client starts slightly earlier.

Default request timeout is now `600s` for:
- `SharedTensorClient`
- `AsyncSharedTensorClient`
- `SharedTensorProvider`

You can still override these per instance:

```python
client = SharedTensorClient(timeout=120.0)
provider = SharedTensorProvider(timeout=120.0)
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
