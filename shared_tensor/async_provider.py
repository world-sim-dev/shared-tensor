"""Backward-compatible async-flavored provider facade."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from shared_tensor.provider import SharedTensorProvider


class AsyncSharedTensorProvider(SharedTensorProvider):
    def __init__(
        self,
        base_port: int = 2537,
        poll_interval: float = 1.0,
        *,
        enabled: bool | None = None,
        server_host: str = "127.0.0.1",
        device_index: int | None = None,
        timeout: float = 30.0,
        execution_mode: str = "auto",
        verbose_debug: bool = False,
    ) -> None:
        super().__init__(
            base_port=base_port,
            enabled=enabled,
            server_host=server_host,
            device_index=device_index,
            timeout=timeout,
            execution_mode=execution_mode,
            verbose_debug=verbose_debug,
        )
        self.poll_interval = poll_interval

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            from shared_tensor.async_client import AsyncSharedTensorClient

            self._async_client = AsyncSharedTensorClient(
                base_port=self.base_port,
                host=self.server_host,
                device_index=self.device_index,
                timeout=self.timeout,
                verbose_debug=self.verbose_debug,
                poll_interval=self.poll_interval,
            )
        return self._async_client

    def register(
        self,
        func: Callable[..., Any],
        *,
        cache: bool = True,
        cache_format_key: str | None = None,
        managed: bool = False,
        async_default_wait: bool = True,
        execution: str = "task",
        concurrency: str = "parallel",
        singleflight: bool = True,
        wait: bool | None = None,
    ) -> Callable[..., Any]:
        resolved_wait = async_default_wait if wait is None else wait
        registered = super().register(
            func,
            cache=cache,
            cache_format_key=cache_format_key,
            managed=managed,
            async_default_wait=resolved_wait,
            execution=cast(Any, execution),
            concurrency=cast(Any, concurrency),
            singleflight=singleflight,
        )
        if self.execution_mode in {"server", "local"}:
            return registered

        endpoint_name = func.__name__

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
        func: Callable[..., Any] | None = None,
        *,
        cache: bool = True,
        cache_format_key: str | None = None,
        managed: bool = False,
        execution: str = "task",
        concurrency: str = "parallel",
        singleflight: bool = True,
        wait: bool | None = None,
        **_: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
        if func is not None:
            return self.register(
                func,
                cache=cache,
                cache_format_key=cache_format_key,
                managed=managed,
                execution=execution,
                concurrency=concurrency,
                singleflight=singleflight,
                wait=wait,
            )

        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            return self.register(
                inner,
                cache=cache,
                cache_format_key=cache_format_key,
                managed=managed,
                execution=execution,
                concurrency=concurrency,
                singleflight=singleflight,
                wait=wait,
            )

        return decorator
