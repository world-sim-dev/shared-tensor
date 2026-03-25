# Examples

The examples directory only shows supported same-host CUDA torch IPC patterns.

- [zero_branch_env.py](/Users/mapix/workspace/shared-tensor/examples/zero_branch_env.py): one file, two processes, env-controlled auto mode
- [model_service.py](/Users/mapix/workspace/shared-tensor/examples/model_service.py): cached managed model registry
- [basic_service.py](/Users/mapix/workspace/shared-tensor/examples/basic_service.py): minimal direct endpoint examples
- [async_service.py](/Users/mapix/workspace/shared-tensor/examples/async_service.py): task submission and wait flow

Auto mode:

```bash
SHARED_TENSOR_ENABLED=1 SHARED_TENSOR_ROLE=server python examples/zero_branch_env.py
SHARED_TENSOR_ENABLED=1 python examples/zero_branch_env.py
```

Without `SHARED_TENSOR_ENABLED=1`, providers stay in local mode by default.

Socket selection is device-aware:
- base path defaults to `/tmp/shared-tensor`
- runtime socket path is `<base_path>-<device_index>.sock`

Manual mode is programmatic only: construct `SharedTensorServer(provider, socket_path=...)` and call `start()`.
