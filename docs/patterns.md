# Patterns

## Pattern 1: Dedicated Model Server

Use for:

- expensive model construction
- many clients
- explicit operational boundaries

Recommended endpoint settings:

```python
@provider.share(
    execution="task",
    managed=True,
    concurrency="serialized",
    singleflight=True,
    cache_format_key="model:{hidden_size}",
)
def load_model(hidden_size: int) -> torch.nn.Module:
    ...
```

Why:

- `task`: long model build does not block the immediate request thread model
- `managed`: client gets explicit handle and release control
- `serialized`: only one materialization path enters the builder at a time
- `singleflight`: concurrent identical requests coalesce
- `cache_format_key`: cache identity is explicit and stable

## Pattern 2: Short-Lived Tensor Transform

Use for:

- request-scoped tensor transforms
- no long-lived ownership on server

Recommended endpoint settings:

```python
@provider.share(execution="direct", cache=False)
def transform(tensor: torch.Tensor) -> torch.Tensor:
    ...
```

Why:

- no need for task retention
- no need for managed registry
- no risk of stale cache aliasing

## Pattern 3: Zero-Branch Two-Process Same Code

Use for:

- one code file
- role chosen only by env vars

Typical launch:

```bash
SHARED_TENSOR_ENABLED=1 SHARED_TENSOR_ROLE=server python demo.py
SHARED_TENSOR_ENABLED=1 python demo.py
```

Important detail:

- server role executes locally and owns the objects
- client role turns the same function call into RPC

## Pattern 4: Embedded Same-Process Convenience Mode

Use for:

- local embedding
- development
- integration inside one existing process tree

Key expectation:

```text
same-process thread-backed mode is convenience mode, not fake cross-process mode
```

So:

- state is shared through the local server object
- same-process clients do not reopen CUDA IPC payloads

## Choosing `cache_format_key`

Good:

```text
model:{hidden_size}
model:{name}:{revision}
tensorizer:{dtype}:{layout}
```

Bad:

```text
fixed
model
```

unless you truly want one global shared object for the endpoint.

## Choosing `managed`

Use `managed=True` when:

- the object is expensive to build
- the object should be reused
- explicit release matters

Keep `managed=False` when:

- result is request-scoped
- caller should not own lifecycle explicitly
- cache identity is irrelevant or disabled

## Choosing `serialized`

Use `concurrency="serialized"` when:

- the endpoint builds one shared expensive resource
- builder code is not safe under concurrent construction
- duplicate concurrent materialization is wasteful

Keep `parallel` when:

- requests are independent
- there is no shared mutable build path

## Choosing `singleflight`

Use `singleflight=True` when:

- multiple callers can race on the same cache key
- only one real build should win

Disable it when:

- every request should independently execute
- you intentionally want concurrent uncached work

## Release Discipline

For managed results, always prefer one of these:

```python
with handle as managed:
    ...
```

or:

```python
try:
    ...
finally:
    handle.release()
```

That is the simplest way to avoid lifetime ambiguity.
