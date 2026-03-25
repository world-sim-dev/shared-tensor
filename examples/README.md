# Examples

The examples directory shows only supported same-host CUDA torch IPC patterns.

- [zero_branch_env.py](/Users/mapix/workspace/shared-tensor/examples/zero_branch_env.py): one file, two processes, `SHARED_TENSOR_ROLE=server` auto-daemon mode with task-backed model loading
- [basic_service.py](/Users/mapix/workspace/shared-tensor/examples/basic_service.py): direct synchronous tensor and module endpoints
- [async_service.py](/Users/mapix/workspace/shared-tensor/examples/async_service.py): async task endpoints returning CUDA objects
- [model_service.py](/Users/mapix/workspace/shared-tensor/examples/model_service.py): serialized model construction with managed handles

Port selection is `base_port + cuda_device_index`. By default the base port is `2537`, or `SHARED_TENSOR_BASE_PORT` when set.

Auto mode:

```bash
SHARED_TENSOR_ROLE=server python examples/zero_branch_env.py
python examples/zero_branch_env.py
```

Manual mode is programmatic only. Construct `SharedTensorServer(provider, port=...)` and call `start()`.
