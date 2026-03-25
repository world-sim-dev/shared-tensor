from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time

import pytest

from shared_tensor import SharedTensorClient, SharedTensorProvider, SharedTensorServer
from shared_tensor.errors import SharedTensorCapabilityError, SharedTensorRemoteError

torch = pytest.importorskip("torch")


def _wait_for_socket(socket_path: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(socket_path):
            return
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for server socket {socket_path}")


def test_cpu_tensor_round_trip_over_rpc_is_rejected(running_server, client_for_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def double_tensor(tensor):
        return tensor * 2

    server = running_server(provider)
    value = torch.arange(4, dtype=torch.float32)

    with client_for_server(server) as client:
        with pytest.raises(SharedTensorCapabilityError):
            client.call("double_tensor", value)


def test_cpu_model_round_trip_over_rpc_is_rejected(running_server, client_for_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def build_linear():
        return torch.nn.Linear(3, 2)

    server = running_server(provider)

    with client_for_server(server) as client:
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

    base_dir = tempfile.mkdtemp(prefix="shared-tensor-gpu-model-")
    base_path = os.path.join(base_dir, "runtime")
    provider.base_path = base_path
    server = SharedTensorServer(provider)
    server.start(blocking=False)
    try:
        _wait_for_socket(server.socket_path)
        with SharedTensorClient(base_path=base_path) as client:
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
        try:
            os.rmdir(base_dir)
        except OSError:
            pass


def _gpu_tensor_round_trip_worker(queue) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(cache=False)
    def identity(tensor):
        return tensor

    base_dir = tempfile.mkdtemp(prefix="shared-tensor-gpu-tensor-")
    base_path = os.path.join(base_dir, "runtime")
    provider.base_path = base_path
    server = SharedTensorServer(provider)
    server.start(blocking=False)
    try:
        _wait_for_socket(server.socket_path)
        value = torch.arange(4, dtype=torch.float32, device="cuda")
        with SharedTensorClient(base_path=base_path) as client:
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
        try:
            os.rmdir(base_dir)
        except OSError:
            pass


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
