from __future__ import annotations

import time

import pytest

from shared_tensor import (
    AsyncSharedTensorClient,
    AsyncSharedTensorProvider,
    SharedObjectHandle,
)
from shared_tensor.async_task import TaskStatus
from shared_tensor.errors import SharedTensorRemoteError, SharedTensorTaskError

torch = pytest.importorskip("torch")


def _client(server, **kwargs) -> AsyncSharedTensorClient:
    return AsyncSharedTensorClient(
        base_path=server.provider.base_path,
        device_index=server.provider.device_index,
        poll_interval=0.01,
        **kwargs,
    )


def test_async_submit_and_result_flow(running_server) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="server")

    @provider.share(wait=False)
    def slow_noop() -> None:
        time.sleep(0.05)
        return None

    server = running_server(provider)

    with _client(server) as client:
        task_id = client.submit("slow_noop")
        assert client.wait_for_task(task_id, timeout=2) is None
        status = client.get_task_status(task_id)
        assert status.status == TaskStatus.COMPLETED
        assert "result_payload" not in status.to_dict()


def test_async_cancel_pending_task(running_server) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="server")

    @provider.share(wait=False, cache=False, singleflight=False)
    def slow_job() -> None:
        time.sleep(0.2)
        return None

    server = running_server(provider, max_workers=1)

    with _client(server) as client:
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

    with _client(server) as client:
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

    with _client(server) as client:
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


def test_async_provider_enabled_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHARED_TENSOR_ENABLED", raising=False)
    monkeypatch.delenv("SHARED_TENSOR_ROLE", raising=False)

    provider = AsyncSharedTensorProvider(enabled=True)

    assert provider.execution_mode == "client"
    assert provider.auto_mode is True


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


def test_async_cached_submit_reuses_server_result(running_server) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="server")

    @provider.share(wait=False, managed=True, cache_format_key="fixed")
    def build_tensor(version: int) -> torch.Tensor:
        return torch.full((1,), float(version), device="cuda")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    server = running_server(provider)

    with _client(server) as client:
        first_task = client.submit("build_tensor", version=1)
        second_task = client.submit("build_tensor", version=1)
        first = client.wait_for_task(first_task, timeout=2)
        second = client.wait_for_task(second_task, timeout=2)
        assert isinstance(first, SharedObjectHandle)
        assert isinstance(second, SharedObjectHandle)
        assert first.object_id == second.object_id
        assert first.release() is True
        assert second.release() is True


def test_async_client_wait_uses_structured_task_error_code() -> None:
    client = AsyncSharedTensorClient(base_path="/tmp/shared-tensor-test", device_index=0)

    def fake_wait_task(task_id, timeout=None):
        raise SharedTensorRemoteError(
            "Remote error [5]: boom",
            code=5,
            error_type="SharedTensorTaskError",
        )

    client._client.wait_task = fake_wait_task
    try:
        with pytest.raises(SharedTensorTaskError, match="boom"):
            client.wait_for_task("task-1", timeout=0.1)
    finally:
        client.close()
