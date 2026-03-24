"""Async provider facade for endpoint-oriented shared_tensor usage."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from shared_tensor.async_client import AsyncSharedTensorClient
from shared_tensor.async_task import TaskInfo
from shared_tensor.provider import SharedTensorProvider


class AsyncSharedTensorProvider(SharedTensorProvider):
    def __init__(
        self,
        server_port: int = 2537,
        *,
        server_host: str = "127.0.0.1",
        timeout: float = 30.0,
        execution_mode: str = "client",
        verbose_debug: bool = False,
        poll_interval: float = 1.0,
    ) -> None:
        super().__init__(
            server_port=server_port,
            server_host=server_host,
            timeout=timeout,
            execution_mode=execution_mode,
            verbose_debug=verbose_debug,
        )
        self.poll_interval = poll_interval
        self._async_client: AsyncSharedTensorClient | None = None

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        cache: bool = False,
        async_default_wait: bool = True,
        wait: bool | None = None,
    ) -> Callable[..., Any]:
        endpoint_name = name or func.__name__
        resolved_wait = async_default_wait if wait is None else wait
        registered = super().register(
            func,
            name=endpoint_name,
            cache=cache,
            async_default_wait=resolved_wait,
        )
        if self.execution_mode in {"server", "local"}:
            return registered

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if resolved_wait:
                return self.call(endpoint_name, *args, **kwargs)
            return self.submit(endpoint_name, *args, **kwargs)

        wrapped = cast(Any, wrapper)
        wrapped.submit_async = lambda *args, **kwargs: self.submit(endpoint_name, *args, **kwargs)
        wrapped.execute_async = lambda *args, wait=resolved_wait, timeout=None, callback=None, **kwargs: self.execute(
            endpoint_name,
            *args,
            wait=wait,
            timeout=timeout,
            callback=callback,
            **kwargs,
        )
        return cast(Callable[..., Any], wrapped)

    def share(
        self,
        name: str | None = None,
        cache: bool = False,
        **_: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return self.register(func, name=name, cache=cache)

        return decorator

    def _get_async_client(self) -> AsyncSharedTensorClient:
        if self._async_client is None:
            self._async_client = AsyncSharedTensorClient(
                port=self.server_port,
                host=self.server_host,
                timeout=self.timeout,
                verbose_debug=self.verbose_debug,
                poll_interval=self.poll_interval,
            )
        return self._async_client

    def submit(self, endpoint: str, *args: Any, **kwargs: Any) -> str:
        if self.execution_mode == "local":
            raise RuntimeError("Local mode does not support async task submission")
        return self._get_async_client().submit(endpoint, *args, **kwargs)

    def execute(
        self,
        endpoint: str,
        *args: Any,
        wait: bool = True,
        timeout: float | None = None,
        callback: Callable[[TaskInfo], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        task_id = self.submit(endpoint, *args, **kwargs)
        if not wait:
            return task_id
        return self.wait_for_task(task_id, timeout=timeout, callback=callback)

    def get_task_status(self, task_id: str) -> TaskInfo:
        return self._get_async_client().get_task_status(task_id)

    def get_task_result(self, task_id: str) -> Any:
        return self._get_async_client().get_task_result(task_id)

    def wait_for_task(
        self,
        task_id: str,
        timeout: float | None = None,
        callback: Callable[[TaskInfo], None] | None = None,
    ) -> Any:
        return self._get_async_client().wait_for_task(task_id, timeout=timeout, callback=callback)

    def cancel_task(self, task_id: str) -> bool:
        return self._get_async_client().cancel_task(task_id)

    def list_tasks(self, status: str | None = None) -> dict[str, TaskInfo]:
        return self._get_async_client().list_tasks(status=status)

    def close(self) -> None:
        super().close()
        if self._async_client is not None:
            self._async_client.close()
            self._async_client = None
