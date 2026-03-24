# Examples

The examples directory shows only supported same-host CUDA torch IPC patterns.

- [basic_service.py](/Users/mapix/workspace/shared-tensor/examples/basic_service.py): synchronous CUDA model and tensor handoff
- [async_service.py](/Users/mapix/workspace/shared-tensor/examples/async_service.py): async task endpoints returning CUDA objects
- [model_service.py](/Users/mapix/workspace/shared-tensor/examples/model_service.py): minimal model/tensor workflow on one GPU

Run a service:

```bash
shared-tensor-server --provider examples.basic_service:provider --port 2537
```

Then call it from Python:

```python
import torch

from shared_tensor import SharedTensorClient

with SharedTensorClient(port=2537) as client:
    model = client.call("build_linear_model")
    x = torch.ones(1, 4, device="cuda")
    print(model(x))
```
