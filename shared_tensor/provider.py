"""Endpoint registration for sync shared_tensor clients and servers."""

from __future__ import annotations

import atexit
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal

from shared_tensor.errors import (
    SharedTensorConfigurationError,
    SharedTensorProviderError,
)
from shared_tensor.utils import (
    build_cache_key,
    resolve_runtime_port,
    resolve_server_base_port,
)

EndpointExecution = Literal["direct", "task"]
EndpointConcurrency = Literal["parallel", "serialized"]


@dataclass(slots=True)
class EndpointDefinition:
    name: str
    func: Callable[..., Any]
    cache: bool = True
    cache_format_key: str | None = None
    managed: bool = False
    async_default_wait: bool = True
    execution: EndpointExecution = "direct"
    concurrency: EndpointConcurrency = "parallel"
    singleflight: bool = True


def _resolve_execution_mode(execution_mode: str) -> tuple[str, bool]:
    if execution_mode == "auto":
        env_role = os.getenv("SHARED_TENSOR_ROLE", "").strip().lower()
        if env_role in {"server", "client", "local"}:
            return env_role, True
        return "client", True
    if execution_mode not in {"client", "server", "local"}:
        raise SharedTensorConfigurationError(
            "execution_mode must be one of 'auto', 'client', 'server', or 'local'"
        )
    return execution_mode, False


def _validate_endpoint_options(
    *,
    execution: EndpointExecution,
    concurrency: EndpointConcurrency,
) -> None:
    if execution not in {"direct", "task"}:
        raise SharedTensorConfigurationError(
            "execution must be either 'direct' or 'task'"
        )
    if concurrency not in {"parallel", "serialized"}:
        raise SharedTensorConfigurationError(
            "concurrency must be either 'parallel' or 'serialized'"
        )


class SharedTensorProvider:
    """Explicit endpoint registry with client wrappers for local CUDA IPC services."""

    def __init__(
        self,
        base_port: int = 2537,
        *,
        server_host: str = "127.0.0.1",
        device_index: int | None = None,
        timeout: float = 30.0,
        execution_mode: str = "auto",
        verbose_debug: bool = False,
    ) -> None:
        resolved_mode, auto_mode = _resolve_execution_mode(execution_mode)
        self.server_host = server_host
        self.base_port = resolve_server_base_port(base_port)
        self.device_index = device_index
        self.timeout = timeout
        self.execution_mode = resolved_mode
        self.auto_mode = auto_mode
        self.verbose_debug = verbose_debug
        self._client: Any | None = None
        self._async_client: Any | None = None
        self._server: Any | None = None
        self._cache: dict[str, Any] = {}
        self._endpoints: dict[str, EndpointDefinition] = {}
        self._registered_functions = self._endpoints
        atexit.register(self.close)

    def register(
        self,
        func: Callable[..., Any],
        *,
        cache: bool = True,
        cache_format_key: str | None = None,
        managed: bool = False,
        async_default_wait: bool = True,
        execution: EndpointExecution = "direct",
        concurrency: EndpointConcurrency = "parallel",
        singleflight: bool = True,
    ) -> Callable[..., Any]:
        _validate_endpoint_options(execution=execution, concurrency=concurrency)
        endpoint_name = func.__name__
        if endpoint_name in self._endpoints:
            raise SharedTensorProviderError(f"Endpoint '{endpoint_name}' is already registered")

        resolved_cache_format_key = (
            func.__qualname__ if cache_format_key is None else cache_format_key
        )

        definition = EndpointDefinition(
            name=endpoint_name,
            func=func,
            cache=cache,
            cache_format_key=resolved_cache_format_key,
            managed=managed,
            async_default_wait=async_default_wait,
            execution=execution,
            concurrency=concurrency,
            singleflight=singleflight,
        )
        self._endpoints[endpoint_name] = definition

        if self._should_autostart_server():
            self._restart_autostart_server()

        if self.execution_mode == "server":
            return func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(endpoint_name, *args, **kwargs)

        wrapped = wrapper
        wrapped.submit = lambda *args, **kwargs: self.submit(endpoint_name, *args, **kwargs)
        return wrapped

    def share(
        self,
        func: Callable[..., Any] | None = None,
        *,
        cache: bool = True,
        cache_format_key: str | None = None,
        managed: bool = False,
        execution: EndpointExecution = "direct",
        concurrency: EndpointConcurrency = "parallel",
        singleflight: bool = True,
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
            )

        return decorator

    def call(self, endpoint: str, *args: Any, **kwargs: Any) -> Any:
        if self.execution_mode in {"server", "local"}:
            return self.invoke_local(endpoint, args=args, kwargs=kwargs)
        return self._get_client().call(endpoint, *args, **kwargs)

    def submit(self, endpoint: str, *args: Any, **kwargs: Any) -> str:
        if self.execution_mode in {"server", "local"}:
            raise RuntimeError("Local and server modes do not support task submission")
        return self._get_async_client().submit(endpoint, *args, **kwargs)

    def execute(
        self,
        endpoint: str,
        *args: Any,
        wait: bool = True,
        timeout: float | None = None,
        callback: Callable[[Any], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        task_id = self.submit(endpoint, *args, **kwargs)
        if not wait:
            return task_id
        return self.wait_for_task(task_id, timeout=timeout, callback=callback)

    def get_task_status(self, task_id: str) -> Any:
        return self._get_async_client().get_task_status(task_id)

    def get_task_result(self, task_id: str) -> Any:
        return self._get_async_client().get_task_result(task_id)

    def wait_for_task(
        self,
        task_id: str,
        timeout: float | None = None,
        callback: Callable[[Any], None] | None = None,
    ) -> Any:
        return self._get_async_client().wait_for_task(task_id, timeout=timeout, callback=callback)

    def cancel_task(self, task_id: str) -> bool:
        return self._get_async_client().cancel_task(task_id)

    def list_tasks(self, status: str | None = None) -> dict[str, Any]:
        return self._get_async_client().list_tasks(status=status)

    def invoke_local(
        self,
        endpoint: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        definition = self.get_endpoint(endpoint)
        resolved_kwargs = kwargs or {}
        if not definition.cache:
            return definition.func(*args, **resolved_kwargs)

        cache_key = self._cache_key_for(endpoint, definition, args, resolved_kwargs)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = definition.func(*args, **resolved_kwargs)
        self._cache[cache_key] = result
        return result

    def get_endpoint(self, endpoint: str) -> EndpointDefinition:
        try:
            return self._endpoints[endpoint]
        except KeyError as exc:
            raise SharedTensorProviderError(f"Endpoint '{endpoint}' is not registered") from exc

    def list_endpoints(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "cache": definition.cache,
                "cache_format_key": definition.cache_format_key,
                "managed": definition.managed,
                "async_default_wait": definition.async_default_wait,
                "execution": definition.execution,
                "concurrency": definition.concurrency,
                "singleflight": definition.singleflight,
            }
            for name, definition in self._endpoints.items()
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._async_client is not None:
            self._async_client.close()
            self._async_client = None
        if self._server is not None:
            self._server.stop()
            self._server = None

    def _get_client(self) -> Any:
        if self._client is None:
            from shared_tensor.client import SharedTensorClient

            self._client = SharedTensorClient(
                base_port=self.base_port,
                host=self.server_host,
                device_index=self.device_index,
                timeout=self.timeout,
                verbose_debug=self.verbose_debug,
            )
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            from shared_tensor.async_client import AsyncSharedTensorClient

            self._async_client = AsyncSharedTensorClient(
                base_port=self.base_port,
                host=self.server_host,
                device_index=self.device_index,
                timeout=self.timeout,
                verbose_debug=self.verbose_debug,
            )
        return self._async_client

    def _should_autostart_server(self) -> bool:
        return self.auto_mode and self.execution_mode == "server"

    def _restart_autostart_server(self) -> None:
        from shared_tensor.server import SharedTensorServer

        if self._server is not None:
            self._server.stop()
        self._server = SharedTensorServer(
            self,
            host=self.server_host,
            port=resolve_runtime_port(self.base_port, self.device_index),
            verbose_debug=self.verbose_debug,
        )
        self._server.start(blocking=False)

    def _cache_key_for(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        return build_cache_key(
            endpoint,
            args,
            kwargs,
            func=definition.func,
            cache_format_key=definition.cache_format_key,
        )
