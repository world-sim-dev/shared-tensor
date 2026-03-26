# Lifecycle

This is the most important part of the design.

`shared_tensor` has four different lifecycles that interact:

- endpoint call lifecycle
- task lifecycle
- managed object lifecycle
- cache lifecycle

## 1. Endpoint Call Lifecycle

Direct call path:

```text
client.call()
  -> server resolves endpoint
  -> server computes cache key if enabled
  -> cache hit returns immediately
  -> otherwise endpoint executes
  -> result is encoded and returned
```

Task-backed path:

```text
client.submit()
  -> server creates task record
  -> TaskManager executes endpoint
  -> client waits / polls
  -> client fetches task result
```

In same-process thread-backed autostart mode, the control flow is similar, but the client short-circuits into the local server object instead of going through UDS and torch IPC reopen.

## 2. Task Lifecycle

Task states:

```text
pending -> running -> completed
pending -> cancelled
running -> failed
```

The server stores:

- task metadata
- encoded payload bytes for remote retrieval
- local in-process result for same-process short-circuit retrieval

Why both forms exist:

- remote consumers need encoded payload bytes
- same-process local consumers must not reopen their own CUDA IPC payloads

### Task Retention

Completed tasks stay in the task manager until:

- TTL cleanup removes them, or
- bounded capacity eviction removes the oldest finished task

This retention window matters because late callers may still request `get_task_result`.

## 3. Managed Object Lifecycle

Managed mode means the server owns the long-lived object and the client receives a handle.

Server-side sequence:

```text
endpoint builds CUDA object
  -> registry allocates object_id
  -> cache_key may point to that object_id
  -> client receives SharedObjectHandle(object_id, value)
```

Reference counting rules:

- new managed object starts at refcount `1`
- cache hits add one ref
- singleflight joiners add one ref when they receive the shared managed result
- `handle.release()` decrements one ref
- when refcount reaches zero, registry entry is destroyed
- if that object owned the cache key, cache index entry is also removed

### Important Design Constraint

```text
managed object lifetime is not the same as task lifetime
```

Tasks retain completion metadata.
Managed registry retains reusable object ownership.

These are separate stores because one task may create a reusable object whose lifetime extends beyond the task completion moment.

### Producer Lifetime Constraint

```text
managed handle lifetime is still bounded by producer process lifetime
```

Even after a client reopens a CUDA tensor or module through torch IPC, `shared_tensor` does not guarantee that the object remains usable after the server process dies.

This library depends on native PyTorch CUDA IPC behavior:
- if the producer process exits too early, client-side use may fail later
- explicit release controls registry ownership, not post-crash durability
- producer liveness is part of the runtime contract

## 4. Cache Lifecycle

There are two cache categories.

### Provider Local Cache

Used only by `SharedTensorProvider.invoke_local()`.

This is the simple in-process local-mode cache.

### Server-Owned Cache

Used when a real `SharedTensorServer` exists.

This is the authoritative cache for:

- remote client calls
- same-process local calls routed through server mode
- task-backed execution

Server-owned cache has two forms:

- `_local_cache`: original in-process values for direct reuse
- managed registry cache index: cache key to `object_id`

The separation matters because managed objects need refcount tracking, while non-managed direct results only need value reuse.

## 5. `cache_format_key` Lifecycle Semantics

`cache_format_key` determines cache identity, not function identity.

Examples:

```text
cache_format_key="model:{hidden_size}"
cache_format_key="weights:{name}:{dtype}"
```

Formatting uses bound function arguments.

Implications:

- arguments omitted from the format string do not affect cache identity
- incompatible format strings fail fast during cache-key construction
- default behavior is function `__qualname__`, which means one cache entry per endpoint without argument variation

## 6. Release Semantics

Client release is explicit:

```python
handle.release()
```

Or context-managed:

```python
with handle as managed:
    ...
```

Current contract:

- first successful release returns `True`
- releasing an already released local handle returns `False`
- releasing an unknown object id on the server is a no-op with `released=False`

## 7. Failure-Prone Edges

These are the lifecycle edges most likely to cause bugs:

- forgetting to add a managed refcount on singleflight join
- treating same-process thread-backed mode like cross-process CUDA IPC
- coupling task retention to managed object retention
- clearing server-owned cache before handles are released
- using overly broad `cache_format_key` values and accidentally aliasing distinct resources

Those are the edges the current test suite should continue to guard.
