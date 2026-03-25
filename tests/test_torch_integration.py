from __future__ import annotations

import multiprocessing as mp
import socket
import time

import pytest

from shared_tensor import SharedTensorClient, SharedTensorProvider, SharedTensorServer
from shared_tensor.errors import SharedTensorCapabilityError, SharedTensorRemoteError

torch = pytest.importorskip("torch")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for server on port {port}")


def test_cpu_tensor_round_trip_over_rpc_is_rejected(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def double_tensor(tensor):
        return tensor * 2

    server = running_server(provider)
    value = torch.arange(4, dtype=torch.float32)

    with SharedTensorClient(base_port=server.port) as client:
        with pytest.raises(SharedTensorCapabilityError):
            client.call("double_tensor", value)


def test_cpu_model_round_trip_over_rpc_is_rejected(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def build_linear():
        return torch.nn.Linear(3, 2)

    server = running_server(provider)

    with SharedTensorClient(base_port=server.port) as client:
        with pytest.raises(SharedTensorRemoteError):
            client.call("build_linear")


def _gpu_model_round_trip_worker(queue) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def build_cuda_linear():
        layer = torch.nn.Linear(3, 2).cuda()
        with torch.no_grad():
            layer.weight.fill_(2.0)
            layer.bias.fill_(0.5)
        return layer

    port = _find_free_port()
    server = SharedTensorServer(provider, host="127.0.0.1", port=port)
    server.start(blocking=False)
    try:
        _wait_for_server(port)
        with SharedTensorClient(base_port=port) as client:
            model = client.call("build_cuda_linear")
        sample = torch.ones(1, 3, device="cuda")
        output = model(sample)
        queue.put(
            {
                "is_cuda": next(model.parameters()).is_cuda,
                "output": output.cpu().tolist(),
            }
        )
    finally:
        server.stop()


def _gpu_tensor_round_trip_worker(queue) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(cache=False)
    def identity(tensor):
        return tensor

    port = _find_free_port()
    server = SharedTensorServer(provider, host="127.0.0.1", port=port)
    server.start(blocking=False)
    try:
        _wait_for_server(port)
        value = torch.arange(4, dtype=torch.float32, device="cuda")
        with SharedTensorClient(base_port=port) as client:
            result = client.call("identity", value)
        queue.put(
            {
                "is_cuda": result.is_cuda,
                "result": result.cpu().tolist(),
                "expected": value.cpu().tolist(),
            }
        )
    finally:
        server.stop()


def _run_gpu_worker(target):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=target, args=(queue,))
    process.start()
    process.join(timeout=30)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
        raise AssertionError("GPU worker timed out")
    assert process.exitcode == 0
    return queue.get(timeout=5)


@pytest.mark.gpu
def test_cuda_model_round_trip_same_host() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    result = _run_gpu_worker(_gpu_model_round_trip_worker)

    assert result["is_cuda"] is True
    assert result["output"] == [[6.5, 6.5]]


@pytest.mark.gpu
def test_cuda_tensor_round_trip_same_host() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    result = _run_gpu_worker(_gpu_tensor_round_trip_worker)

    assert result["is_cuda"] is True
    assert result["result"] == result["expected"]
