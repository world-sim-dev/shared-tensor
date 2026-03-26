# Diagrams

## End-To-End Request Shapes

### Dedicated Two-Process Direct Call

```text
client process                         server process
--------------                        ---------------------------
client.call(endpoint)                 |
  -> UDS RPC request ---------------->| decode args/kwargs
                                      | lookup endpoint
                                      | cache / singleflight / lock policy
                                      | execute endpoint
                                      | serialize CUDA result via torch IPC
  <- RPC response with payload -------|
deserialize payload
reopen CUDA object in client process
```

### Dedicated Two-Process Managed Object

```text
client process                         server process
--------------                        ---------------------------
client.call(load_model)               |
  -> UDS RPC request ---------------->| build model
                                      | register object_id
                                      | cache_key -> object_id
                                      | serialize payload + include object_id
  <- managed payload -----------------|
wrap as SharedObjectHandle
use handle.value(...)
handle.release()
  -> UDS RPC release ---------------->| refcount -= 1
                                      | destroy registry entry at refcount 0
```

### Same-Process Thread-Backed Autostart

```text
same process

provider auto-starts SharedTensorServer in a thread
      |
runtime registry maps socket_path -> server object
      |
client.call(endpoint)
      |
lookup local server by socket_path
      |
direct call into SharedTensorServer object
      |
return original in-process object

No UDS round trip.
No CUDA IPC reopen.

### Zero-Branch Env Resolution

```text
same Python file
      |
      v
SHARED_TENSOR_ENABLED unset/false? ---- yes ---> provider resolves to local
      |
      no
      v
SHARED_TENSOR_ROLE=server? ---------- yes ---> provider resolves to server
      |
      no
      v
provider resolves to client
```
```

## Server Materialization Policy

```text
resolve endpoint
    |
compute cache key if cache=True
    |
cache hit? -------------------- yes -> return cached result
    |
    no
    |
singleflight enabled and same cache key in-flight?
    |                         |
    | yes                     | no
    |                         v
    |                   concurrency=serialized?
    |                         |
    |                         +-- yes -> endpoint lock -> build
    |                         |
    |                         +-- no  -> build
    |
join owner future
    |
managed result? add_ref on shared object id
    |
return result
```

## Managed Object Refcount Lifecycle

```text
initial build
  refcount = 1

cache hit / joined managed result
  refcount += 1

client handle release
  refcount -= 1

refcount == 0
  -> remove object entry
  -> remove cache_key index if it points to this object
```

## Task Lifecycle

```text
submit
  -> pending
  -> running
  -> completed | failed | cancelled

completed task retains:
  - task metadata
  - encoded remote payload
  - local in-process result
```

## Cache Ownership Model

```text
local mode
  provider._cache owns local results

real server exists
  server._local_cache owns direct cached values
  managed registry owns managed cached values
  provider local wrappers should route through server-owned state
```
