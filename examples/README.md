# Examples

The examples directory only shows supported same-host CUDA torch IPC patterns.
Modules that are meant to back a dedicated server process now construct providers in explicit `execution_mode="server"` so the example import state matches the documented production topology.

- [model_service.py](/Users/mapix/workspace/shared-tensor/examples/model_service.py): primary production-style server endpoint definitions
- [zero_branch_env.py](/Users/mapix/workspace/shared-tensor/examples/zero_branch_env.py): one file, two processes, env-controlled auto mode
- [basic_service.py](/Users/mapix/workspace/shared-tensor/examples/basic_service.py): minimal direct endpoint examples
- [async_service.py](/Users/mapix/workspace/shared-tensor/examples/async_service.py): task submission and wait flow using `SharedTensorProvider(execution="task")` and `AsyncSharedTensorClient`
- [transformers_two_proc_demo.py](/Users/mapix/workspace/shared-tensor/examples/transformers_two_proc_demo.py): smallest two-process CUDA `transformers` model sharing demo
- [transformers_mutation_check.py](/Users/mapix/workspace/shared-tensor/examples/transformers_mutation_check.py): verifies client-side parameter mutation is observed by the server for a shared `transformers` model
- [transformers_ipc_benchmark.py](/Users/mapix/workspace/shared-tensor/examples/transformers_ipc_benchmark.py): synthetic benchmark for `transformers` IPC reopen latency and client GPU memory delta

Recommended production pattern: start a dedicated server process around `model_service.py`, then connect from clients with `SharedTensorClient`.

Auto mode:

```bash
SHARED_TENSOR_ENABLED=1 SHARED_TENSOR_ROLE=server python examples/zero_branch_env.py
SHARED_TENSOR_ENABLED=1 python examples/zero_branch_env.py
```

Without `SHARED_TENSOR_ENABLED=1`, providers stay in local mode by default.

For the `transformers` demos:

```bash
TRANSFORMERS_MODEL_ROOT=/path/to/models--bert-base-uncased \
SHARED_TENSOR_ENABLED=1 SHARED_TENSOR_ROLE=server \
python examples/transformers_two_proc_demo.py

TRANSFORMERS_MODEL_ROOT=/path/to/models--bert-base-uncased \
SHARED_TENSOR_ENABLED=1 \
python examples/transformers_two_proc_demo.py
```

`TRANSFORMERS_MODEL_ROOT` may point either at a concrete model directory or a Hugging Face cache root containing `snapshots/`; the example resolves the newest snapshot automatically. `TRANSFORMERS_AUTO_CLASS` defaults to `AutoModel`.

The default client behavior now tolerates normal startup races: connection setup retries for up to `60s`, and request timeout defaults to `600s`.

Socket selection is device-aware:
- base path defaults to `/tmp/shared-tensor`
- runtime socket path is `<base_path>-<device_index>.sock`

Manual mode is programmatic only: construct `SharedTensorServer(provider, socket_path=...)` and call `start()`.
For production-style startup control, you can inspect `provider.get_runtime_info()` / `client.get_server_info()`. Background autostart uses a thread inside the current process; explicit production servers should still use a dedicated process.
