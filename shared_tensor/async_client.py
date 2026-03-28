"""Task-oriented client facade built on top of :mod:`shared_tensor.client`."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, cast

from shared_tensor.async_task import TaskInfo, TaskStatus
from shared_tensor.client import SharedTensorClient
from shared_tensor.errors import SharedTensorRemoteError, SharedTensorTaskError
from shared_tensor.managed_object import SharedObjectHandle


logger = logging.getLogger(__name__)


class AsyncSharedTensorClient:
    def __init__(
        self,
        base_path: str = "/tmp/shared-tensor",
        verbose_debug: bool = True,
        poll_interval: float = 1.0,
        *,
        device_index: int | None = None,
        timeout: float = 600.0,
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
        if self._client.verbose_debug:
            logger.debug("Async client submitting task", extra={"endpoint": endpoint})
        return self._client.submit(endpoint, *args, **kwargs)

    def status(self, task_id: str) -> TaskInfo:
        return TaskInfo.from_dict(self._client.get_task_status(task_id))

    def get_task_status(self, task_id: str) -> TaskInfo:
        return self.status(task_id)

    def result(self, task_id: str) -> Any:
        return self._client.get_task_result(task_id)

    def get_task_result(self, task_id: str) -> Any:
        return self.result(task_id)

    def ping(self) -> bool:
        return self._client.ping()

    def get_server_info(self) -> dict[str, Any]:
        return self._client.get_server_info()

    def list_endpoints(self) -> dict[str, Any]:
        return self._client.list_endpoints()

    def release(self, object_id: str) -> bool:
        return self._client.release(object_id)

    def release_many(self, object_ids: list[str]) -> dict[str, bool]:
        return self._client.release_many(object_ids)

    def get_object_info(self, object_id: str) -> dict[str, Any] | None:
        return self._client.get_object_info(object_id)

    def ensure_handle_live(
        self,
        handle: SharedObjectHandle[Any],
        *,
        refresh: bool = True,
    ) -> dict[str, Any]:
        return self._client.ensure_handle_live(handle, refresh=refresh)

    def invalidate_call_cache(self, endpoint: str, *args: Any, **kwargs: Any) -> bool:
        return self._client.invalidate_call_cache(endpoint, *args, **kwargs)

    def invalidate_endpoint_cache(self, endpoint: str) -> int:
        return self._client.invalidate_endpoint_cache(endpoint)

    def wait(
        self,
        task_id: str,
        timeout: float | None = None,
        callback: Callable[[TaskInfo], None] | None = None,
    ) -> Any:
        if self._client.verbose_debug:
            logger.debug("Async client waiting for task", extra={"task_id": task_id, "timeout": timeout})
        started = time.time()
        while True:
            remaining = None if timeout is None else max(timeout - (time.time() - started), 0.0)
            try:
                info = TaskInfo.from_dict(self._client.wait_task(task_id, timeout=remaining))
            except SharedTensorRemoteError as exc:
                if exc.code != 5:
                    raise
                if self._client.verbose_debug:
                    logger.warning("Async client observed task failure", extra={"task_id": task_id, "code": exc.code})
                raise SharedTensorTaskError(str(exc)) from exc
            if callback is not None:
                callback(info)
            if self._client.verbose_debug:
                logger.debug("Async client polled task", extra={"task_id": task_id, "status": info.status.value})
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
