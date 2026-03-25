"""Deprecated compatibility shim for task-oriented provider usage."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from shared_tensor.provider import SharedTensorProvider


class AsyncSharedTensorProvider(SharedTensorProvider):
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
