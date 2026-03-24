from __future__ import annotations

import time

import pytest

from shared_tensor import AsyncSharedTensorClient, AsyncSharedTensorProvider
from shared_tensor.async_task import TaskStatus
from shared_tensor.errors import SharedTensorTaskError


def test_async_submit_and_result_flow(running_server) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="server")

    @provider.share(name="slow_noop", wait=False)
    def slow_noop() -> None:
        time.sleep(0.05)
        return None

    server = running_server(provider)

    with AsyncSharedTensorClient(port=server.port, poll_interval=0.01) as client:
        task_id = client.submit("slow_noop")
        assert client.wait_for_task(task_id, timeout=2) is None
        assert client.get_task_status(task_id).status == TaskStatus.COMPLETED


def test_async_cancel_pending_task(running_server) -> None:
    provider = AsyncSharedTensorProvider(execution_mode="server")

    @provider.share(name="slow_job", wait=False)
    def slow_job() -> None:
        time.sleep(0.2)
        return None

    server = running_server(provider, max_workers=1)

    with AsyncSharedTensorClient(port=server.port, poll_interval=0.01) as client:
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

    @provider.share(name="explode", wait=False)
    def explode() -> None:
        raise ValueError("boom")

    server = running_server(provider)

    with AsyncSharedTensorClient(port=server.port, poll_interval=0.01) as client:
        task_id = client.submit("explode")
        with pytest.raises(SharedTensorTaskError):
            client.wait_for_task(task_id, timeout=2)
