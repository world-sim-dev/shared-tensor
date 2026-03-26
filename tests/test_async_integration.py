from __future__ import annotations

import time

import pytest

from shared_tensor import (
    AsyncSharedTensorClient,
    SharedObjectHandle,
    SharedTensorProvider,
    SharedTensorStaleHandleError,
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
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(execution="task")
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
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(execution="task", cache=False, singleflight=False)
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
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(execution="task")
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

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(execution="task", managed=True)
    def build_tensor() -> torch.Tensor:
        return torch.arange(3, dtype=torch.float32, device="cuda")

    server = running_server(provider)

    with _client(server) as client:
        task_id = client.submit("build_tensor")
        handle = client.wait_for_task(task_id, timeout=2)
        assert isinstance(handle, SharedObjectHandle)
        assert handle.value.is_cuda
        assert handle.release() is True


def test_async_cached_submit_reuses_server_result(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(execution="task", managed=True, cache_format_key="fixed")
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


def test_async_client_forwards_runtime_and_cache_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AsyncSharedTensorClient(base_path="/tmp/shared-tensor-test", device_index=0)
    handle = SharedObjectHandle(
        object_id="obj-1",
        value=object(),
        _releaser=type("Releaser", (), {"release": lambda self: True, "get_object_info": lambda self: None})(),
        server_id="srv-1",
    )

    monkeypatch.setattr(client._client, "ping", lambda: True)
    monkeypatch.setattr(client._client, "get_server_info", lambda: {"server_id": "srv-1"})
    monkeypatch.setattr(client._client, "list_endpoints", lambda: {"build": {"managed": True}})
    monkeypatch.setattr(client._client, "release", lambda object_id: object_id == "obj-1")
    monkeypatch.setattr(client._client, "release_many", lambda object_ids: {object_id: True for object_id in object_ids})
    monkeypatch.setattr(client._client, "get_object_info", lambda object_id: {"object_id": object_id, "server_id": "srv-1"})
    monkeypatch.setattr(client._client, "ensure_handle_live", lambda managed_handle, refresh=True: {"object_id": managed_handle.object_id, "refresh": refresh})
    monkeypatch.setattr(client._client, "invalidate_call_cache", lambda endpoint, *args, **kwargs: endpoint == "build")
    monkeypatch.setattr(client._client, "invalidate_endpoint_cache", lambda endpoint: 2 if endpoint == "build" else 0)

    try:
        assert client.ping() is True
        assert client.get_server_info() == {"server_id": "srv-1"}
        assert client.list_endpoints() == {"build": {"managed": True}}
        assert client.release("obj-1") is True
        assert client.release_many(["obj-1", "obj-2"]) == {"obj-1": True, "obj-2": True}
        assert client.get_object_info("obj-1") == {"object_id": "obj-1", "server_id": "srv-1"}
        assert client.ensure_handle_live(handle, refresh=False) == {"object_id": "obj-1", "refresh": False}
        assert client.invalidate_call_cache("build") is True
        assert client.invalidate_endpoint_cache("build") == 2
    finally:
        client.close()


def test_async_client_preserves_stale_handle_error_details() -> None:
    client = AsyncSharedTensorClient(base_path="/tmp/shared-tensor-test", device_index=0)
    handle = SharedObjectHandle(
        object_id="obj-1",
        value=object(),
        _releaser=type("Releaser", (), {"release": lambda self: True, "get_object_info": lambda self: None})(),
        server_id="srv-1",
    )
    try:
        with pytest.raises(SharedTensorStaleHandleError) as exc_info:
            client.ensure_handle_live(handle)
        assert exc_info.value.reason == "object_missing"
        assert exc_info.value.object_id == "obj-1"
    finally:
        client.close()
