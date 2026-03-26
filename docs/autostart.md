# Autostart And Same-Process Behavior

## Why This Needs Special Documentation

Auto-start looks simple from the API surface, but it is where the easiest design mistakes happen.

The dangerous wrong assumption is:

```text
"If the server is in another thread, I can treat it like another process."
```

That assumption is false for CUDA IPC.

## Auto Mode Resolution

`SharedTensorProvider(execution_mode="auto")` resolves as follows:

```text
enabled false  -> local
enabled true + SHARED_TENSOR_ROLE=server -> server
enabled true + SHARED_TENSOR_ROLE=client or unset -> client
```

Where `enabled` comes from:

- provider argument `enabled`, if not `None`
- otherwise `SHARED_TENSOR_ENABLED`

## What Happens In Auto Server Mode

When auto mode resolves to `server`:

```text
provider.register(endpoint)
  -> provider restarts/starts SharedTensorServer
  -> server runs in a background thread
  -> server registers itself in local runtime registry by socket path
```

That registry is the key mechanism that makes same-process behavior safe.

## Same-Process Short-Circuit Path

When `SharedTensorClient` targets a socket path that belongs to a locally registered thread-backed server in the same process:

```text
client.call()
  -> runtime registry lookup by socket_path
  -> direct call into SharedTensorServer object
  -> return original in-process object
  -> still map failures as SharedTensorRemoteError
```

The same applies to:

- `submit`
- `get_task_result`
- `wait_task`
- `release_object`
- `release_objects`
- `get_object_info`
- `list_tasks`

## Why Encode/Decode Is Wrong Here

Cross-process path:

```text
producer process serializes CUDA payload
consumer process reopens CUDA payload
```

Same-process thread path must not do that:

```text
same process
  -> no reopen step
  -> no CUDA IPC self-deserialization
```

If the same process serializes and then reopens its own CUDA IPC payload as if it were a remote consumer, H100-class systems can fail with invalid resource handle errors.

That is the exact class of bug this design now avoids.

## Cache Ownership In Auto Server Mode

Once a real server object exists, it must be the owner of shared state.

That means:

- local provider calls in server mode route through server-owned state
- remote client calls use the same server-owned state
- autostart is not allowed to create a second independent cache universe

Shared state includes:

- managed object registry
- server `_local_cache`
- server singleflight table
- task manager state

## Practical Rule

Use this rule when reasoning about behavior:

```text
same process + thread-backed server:
    behave like one process with one authoritative server state

different process:
    behave like normal RPC + torch CUDA IPC
```

## Production Guidance

Auto-start is valid for:

- embedding shared_tensor into one long-running process
- development
- zero-branch environment-controlled startup

Prefer explicit dedicated server processes for:

- production model-serving topologies
- clearer operational ownership
- easier resource isolation
- fewer shutdown edge cases
