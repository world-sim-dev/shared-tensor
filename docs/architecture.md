# Architecture

## Component Map

```text
SharedTensorProvider
    registers endpoints
    resolves runtime mode
    owns local-mode cache
    may auto-start SharedTensorServer

SharedTensorServer
    owns endpoint execution for server mode
    owns task manager
    owns server-side cache
    owns managed object registry
    owns singleflight coordination

SharedTensorClient
    sends UDS RPC requests
    reopens CUDA payloads via torch IPC
    wraps managed results as SharedObjectHandle

AsyncSharedTensorClient
    task-oriented facade over SharedTensorClient
    preserves runtime introspection, cache invalidation,
    and managed-handle lifecycle helpers from the sync client

TaskManager
    in-process thread pool
    tracks task state and retained results

ManagedObjectRegistry
    object_id allocation
    cache-key to object mapping
    reference counting
```

## Control Plane And Data Plane

### Control Plane

The control plane is local Unix Domain Socket RPC.

Responsibilities:

- endpoint lookup
- task submission
- task polling / waiting
- object release
- object info lookup
- server introspection

### Data Plane

The data plane is PyTorch CUDA IPC serialization.

Responsibilities:

- encode CUDA tensors / modules in the producer process
- reopen them in a consumer process on the same host and GPU

Important constraint:

```text
CUDA IPC is a cross-process mechanism.
It is not a same-process object transport.
```

That is why the thread-backed local server path must short-circuit locally instead of serializing and reopening its own CUDA objects.

## Socket Selection

Runtime socket path is device-scoped:

```text
<base_path>-<device_index>.sock
```

Where:

- `base_path` comes from provider construction or `SHARED_TENSOR_BASE_PATH`
- `device_index` comes from explicit provider config or lazy CUDA probe

This means one logical server namespace per GPU device index.

## Endpoint Execution Modes

Per endpoint, execution is one of:

- `direct`: execute immediately in the request path
- `task`: execute through `TaskManager`

Per endpoint, concurrency is one of:

- `parallel`: no endpoint-level lock
- `serialized`: one endpoint-level lock around materialization

Per endpoint, deduplication is controlled by:

- `singleflight=True`: concurrent requests for the same cache key join one in-flight build
- `singleflight=False`: every request builds independently unless cache is already populated

## Same-Process Thread-Backed Autostart

When auto mode resolves to `server`, the provider starts a background server thread in the same process.

The implementation now uses an in-process runtime registry:

```text
socket_path -> SharedTensorServer instance
```

When a client in the same process targets that socket path, it does not go through UDS + CUDA IPC reopen. It directly calls the registered server object and returns the original in-process result, while still preserving remote-style error mapping.

That gives two important properties:

- same-process convenience mode shares the same server-owned cache, task manager, and managed registry
- same-process convenience mode does not trigger invalid CUDA IPC reopen attempts
