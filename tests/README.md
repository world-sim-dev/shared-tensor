# Tests

`pytest` is the only supported test runner.

Main coverage buckets:

- `test_protocol.py`: JSON-RPC parsing and validation
- `test_provider.py`: endpoint registration and wrapper behavior
- `test_serialization.py`: CUDA payload rules and capability boundaries
- `test_sync_integration.py`: real server/client sync flow for empty control-path RPC
- `test_async_integration.py`: task lifecycle behavior for empty control-path RPC
- `test_torch_integration.py`: rejected CPU payloads and optional CUDA RPC

Run the default suite:

```bash
python -m pytest -m "not gpu"
```

Run GPU coverage when CUDA is available:

```bash
python -m pytest -m gpu
```
