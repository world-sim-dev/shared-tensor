from __future__ import annotations

import time

import pytest

from shared_tensor import (
    AsyncSharedTensorClient,
    AsyncSharedTensorProvider,
    SharedObjectHandle,
)
from shared_tensor.async_task import TaskStatus
from shared_tensor.errors import SharedTensorTaskError

torch = pytest.importorskip("torch")


def test_async_submit_and_result_flow(running_server) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="server")

    @provider.share(wait=False)
    def slow_noop() -> None:
        time.sleep(0.05)
        return None

    server = running_server(provider)

    with AsyncSharedTensorClient(base_port=server.port, poll_interval=0.01) as client:
        task_id = client.submit("slow_noop")
        assert client.wait_for_task(task_id, timeout=2) is None
        assert client.get_task_status(task_id).status == TaskStatus.COMPLETED


def test_async_cancel_pending_task(running_server) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="server")

    @provider.share(wait=False, cache=False, singleflight=False)
    def slow_job() -> None:
        time.sleep(0.2)
        return None

    server = running_server(provider, max_workers=1)

    with AsyncSharedTensorClient(base_port=server.port, poll_interval=0.01) as client:
        first = client.submit("slow_job")
        second = client.submit("slow_job")
        time.sleep(0.02)
        cancelled = client.cancel_task(second)
        assert cancelled is True
        status = client.get_task_status(second)
        assert status.status == TaskStatus.CANCELLED
        assert client.wait_for_task(first, timeout=2) is None


def test_async_failed_task_surfaces_error(running_server) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="server")

    @provider.share(wait=False)
    def explode() -> None:
        raise ValueError("boom")

    server = running_server(provider)

    with AsyncSharedTensorClient(base_port=server.port, poll_interval=0.01) as client:
        task_id = client.submit("explode")
        with pytest.raises(SharedTensorTaskError):
            client.wait_for_task(task_id, timeout=2)


def test_async_managed_result_returns_handle(running_server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = AsyncSharedTensorProvider(execution_mode="server")

    @provider.share(managed=True, wait=False)
    def build_tensor() -> torch.Tensor:
        return torch.arange(3, dtype=torch.float32, device="cuda")

    server = running_server(provider)

    with AsyncSharedTensorClient(base_port=server.port, poll_interval=0.01) as client:
        task_id = client.submit("build_tensor")
        handle = client.wait_for_task(task_id, timeout=2)
        assert isinstance(handle, SharedObjectHandle)
        assert handle.value.is_cuda
        assert handle.release() is True


def test_async_provider_defaults_to_task_execution() -> None:
    provider = AsyncSharedTensorProvider(execution_mode="local")

    @provider.share
    def build() -> None:
        return None

    metadata = provider.list_endpoints()["build"]
    assert metadata["execution"] == "task"


def test_async_provider_wait_true_uses_sync_call_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="client")
    calls = []

    def fake_call(endpoint, *args, **kwargs):
        calls.append((endpoint, args, kwargs))
        return "ok"

    monkeypatch.setattr(provider, "call", fake_call)

    @provider.share(wait=True)
    def build(value: int) -> None:
        return None

    assert build(3) == "ok"
    assert calls == [("build", (3,), {})]


def test_async_provider_wait_false_uses_submit_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="client")
    calls = []

    def fake_submit(endpoint, *args, **kwargs):
        calls.append((endpoint, args, kwargs))
        return "task-123"

    monkeypatch.setattr(provider, "submit", fake_submit)

    @provider.share(wait=False)
    def build(value: int) -> None:
        return None

    assert build(3) == "task-123"
    assert build.submit_async(4) == "task-123"
    assert calls == [("build", (3,), {}), ("build", (4,), {})]
