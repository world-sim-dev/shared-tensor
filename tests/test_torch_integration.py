from __future__ import annotations

import pytest

from shared_tensor import SharedTensorClient, SharedTensorProvider
from shared_tensor.errors import SharedTensorCapabilityError, SharedTensorRemoteError

torch = pytest.importorskip("torch")


def test_cpu_tensor_round_trip_over_rpc_is_rejected(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(name="double_tensor")
    def double_tensor(tensor):
        return tensor * 2

    server = running_server(provider)
    value = torch.arange(4, dtype=torch.float32)

    with SharedTensorClient(port=server.port) as client:
        with pytest.raises(SharedTensorCapabilityError):
            client.call("double_tensor", value)


def test_cpu_model_round_trip_over_rpc_is_rejected(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(name="build_linear")
    def build_linear():
        return torch.nn.Linear(3, 2)

    server = running_server(provider)

    with SharedTensorClient(port=server.port) as client:
        with pytest.raises(SharedTensorRemoteError):
            client.call("build_linear")


@pytest.mark.gpu
def test_cuda_model_round_trip_same_host(running_server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(name="build_cuda_linear")
    def build_cuda_linear():
        layer = torch.nn.Linear(3, 2).cuda()
        with torch.no_grad():
            layer.weight.fill_(2.0)
            layer.bias.fill_(0.5)
        return layer

    server = running_server(provider)

    with SharedTensorClient(port=server.port) as client:
        model = client.call("build_cuda_linear")

    sample = torch.ones(1, 3, device="cuda")
    output = model(sample)
    assert next(model.parameters()).is_cuda
    assert torch.allclose(output.cpu(), torch.full((1, 2), 6.5))


@pytest.mark.gpu
def test_cuda_tensor_round_trip_same_host(running_server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(name="identity")
    def identity(tensor):
        return tensor

    server = running_server(provider)
    value = torch.arange(4, dtype=torch.float32, device="cuda")

    with SharedTensorClient(port=server.port) as client:
        result = client.call("identity", value)

    assert result.is_cuda
    assert torch.equal(result.cpu(), value.cpu())
