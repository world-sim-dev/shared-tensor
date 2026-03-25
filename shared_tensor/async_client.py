"""Task-oriented client facade built on top of :mod:`shared_tensor.client`."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

from shared_tensor.async_task import TaskInfo, TaskStatus
from shared_tensor.client import SharedTensorClient
from shared_tensor.errors import SharedTensorTaskError


class AsyncSharedTensorClient:
    def __init__(
        self,
        base_path: str = "/tmp/shared-tensor",
        verbose_debug: bool = False,
        poll_interval: float = 1.0,
        *,
        device_index: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._client = SharedTensorClient(
            base_path=base_path,
            device_index=device_index,
            timeout=timeout,
            verbose_debug=verbose_debug,
        )

    def submit(self, endpoint: str, *args: Any, **kwargs: Any) -> str:
        return self._client.submit(endpoint, *args, **kwargs)

    def status(self, task_id: str) -> TaskInfo:
        return TaskInfo.from_dict(self._client.get_task_status(task_id))

    def get_task_status(self, task_id: str) -> TaskInfo:
        return self.status(task_id)

    def result(self, task_id: str) -> Any:
        return self._client.get_task_result(task_id)

    def get_task_result(self, task_id: str) -> Any:
        return self.result(task_id)

    def wait(
        self,
        task_id: str,
        timeout: float | None = None,
        callback: Callable[[TaskInfo], None] | None = None,
    ) -> Any:
        started = time.time()
        while True:
            remaining = None if timeout is None else max(timeout - (time.time() - started), 0.0)
            info = TaskInfo.from_dict(self._client.wait_task(task_id, timeout=remaining))
            if callback is not None:
                callback(info)
            if info.status == TaskStatus.COMPLETED:
                return self.result(task_id)
            if info.status == TaskStatus.FAILED:
                raise SharedTensorTaskError(info.error_message or f"Task '{task_id}' failed")
            if info.status == TaskStatus.CANCELLED:
                raise SharedTensorTaskError(f"Task '{task_id}' was cancelled")
            if timeout is not None and time.time() - started > timeout:
                raise SharedTensorTaskError(
                    f"Task '{task_id}' did not complete within {timeout} seconds"
                )

    def wait_for_task(
        self,
        task_id: str,
        timeout: float | None = None,
        callback: Callable[[TaskInfo], None] | None = None,
    ) -> Any:
        return self.wait(task_id, timeout=timeout, callback=callback)

    def cancel(self, task_id: str) -> bool:
        return self._client.cancel_task(task_id)

    def cancel_task(self, task_id: str) -> bool:
        return self.cancel(task_id)

    def list_tasks(self, status: str | None = None) -> dict[str, TaskInfo]:
        result = self._client.list_tasks(status=status)
        return {task_id: TaskInfo.from_dict(data) for task_id, data in result.items()}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AsyncSharedTensorClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
