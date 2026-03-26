"""Endpoint registration for sync shared_tensor clients and servers."""

from __future__ import annotations

import atexit
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import Any, Literal

from shared_tensor.errors import (
    SharedTensorConfigurationError,
    SharedTensorProviderError,
)
from shared_tensor.utils import (
    build_cache_key,
    resolve_runtime_socket_path,
    resolve_server_base_path,
)

EndpointExecution = Literal["direct", "task"]
EndpointConcurrency = Literal["parallel", "serialized"]
SHARED_TENSOR_ENABLED_ENV = "SHARED_TENSOR_ENABLED"

logger = logging.getLogger(__name__)


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


def _resolve_execution_mode(
    execution_mode: str,
    *,
    enabled: bool | None = None,
) -> tuple[str, bool]:
    if execution_mode == "auto":
        if not _is_shared_tensor_enabled(enabled):
            return "local", True
        env_role = os.getenv("SHARED_TENSOR_ROLE", "").strip().lower()
        if env_role in {"server", "client", "local"}:
            return env_role, True
        return "client", True
    if execution_mode not in {"client", "server", "local"}:
        raise SharedTensorConfigurationError(
            "execution_mode must be one of 'auto', 'client', 'server', or 'local'"
        )
    return execution_mode, False


def _is_shared_tensor_enabled(enabled: bool | None) -> bool:
    if enabled is not None:
        return enabled
    raw = os.getenv(SHARED_TENSOR_ENABLED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


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
        base_path: str = "/tmp/shared-tensor",
        *,
        enabled: bool | None = None,
        device_index: int | None = None,
        timeout: float = 30.0,
        execution_mode: str = "auto",
        server_startup_timeout: float = 30.0,
        verbose_debug: bool = False,
    ) -> None:
        resolved_mode, auto_mode = _resolve_execution_mode(
            execution_mode,
            enabled=enabled,
        )
        self.base_path = resolve_server_base_path(base_path)
        self.enabled = enabled
        self.device_index = device_index
        self.timeout = timeout
        self.execution_mode = resolved_mode
        self.auto_mode = auto_mode
        self.server_startup_timeout = server_startup_timeout
        self.verbose_debug = verbose_debug
        self._client: Any | None = None
        self._async_client: Any | None = None
        self._server: Any | None = None
        self._cache: dict[str, Any] = {}
        self._endpoints: dict[str, EndpointDefinition] = {}
        self._registered_functions = self._endpoints
        self._lock = RLock()
        self._atexit_registered = False
        self._register_atexit_once()

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
        with self._lock:
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
        if self.verbose_debug:
            logger.debug(
                "Provider registered endpoint",
                extra={
                    "endpoint": endpoint_name,
                    "execution_mode": self.execution_mode,
                    "execution": execution,
                    "managed": managed,
                    "cache": cache,
                },
            )

        if self._should_autostart_server():
            self._ensure_autostart_server()

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
        if self.verbose_debug:
            logger.debug("Provider dispatching call", extra={"endpoint": endpoint, "mode": self.execution_mode})
        if self.execution_mode == "server":
            if self._server is not None and hasattr(self._server, "invoke_local"):
                return self._server.invoke_local(endpoint, args=args, kwargs=kwargs)
            return self.invoke_local(endpoint, args=args, kwargs=kwargs)
        if self.execution_mode == "local":
            return self.invoke_local(endpoint, args=args, kwargs=kwargs)
        return self._get_client().call(endpoint, *args, **kwargs)

    def submit(self, endpoint: str, *args: Any, **kwargs: Any) -> str:
        if self.verbose_debug:
            logger.debug("Provider submitting task", extra={"endpoint": endpoint, "mode": self.execution_mode})
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
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        result = definition.func(*args, **resolved_kwargs)
        with self._lock:
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
        if self.verbose_debug:
            logger.debug("Provider closing resources", extra={"mode": self.execution_mode})
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._async_client is not None:
            self._async_client.close()
            self._async_client = None
        if self._server is not None:
            self._server.stop(wait_for_tasks=True)
            self._server = None

    def get_runtime_info(self) -> dict[str, Any]:
        if self.execution_mode in {"server", "local"}:
            server = self._server
            return {
                "execution_mode": self.execution_mode,
                "auto_mode": self.auto_mode,
                "base_path": self.base_path,
                "device_index": self.device_index,
                "server_socket_path": resolve_runtime_socket_path(self.base_path, self.device_index),
                "server_running": bool(server is not None and getattr(server, "running", True)),
            }
        server_info = self._get_client().get_server_info()
        return {
            "execution_mode": self.execution_mode,
            "auto_mode": self.auto_mode,
            "base_path": self.base_path,
            "device_index": self.device_index,
            "server_socket_path": server_info.get("socket_path"),
            "server_running": bool(server_info.get("running")),
            "server_ready": bool(server_info.get("ready")),
            "server_info": server_info,
        }

    def _get_client(self) -> Any:
        if self._client is None:
            from shared_tensor.client import SharedTensorClient

            self._client = SharedTensorClient(
                base_path=self.base_path,
                device_index=self.device_index,
                timeout=self.timeout,
                verbose_debug=self.verbose_debug,
            )
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            from shared_tensor.async_client import AsyncSharedTensorClient

            self._async_client = AsyncSharedTensorClient(
                base_path=self.base_path,
                device_index=self.device_index,
                timeout=self.timeout,
                verbose_debug=self.verbose_debug,
            )
        return self._async_client

    def _should_autostart_server(self) -> bool:
        return self.auto_mode and self.execution_mode == "server"

    def _ensure_autostart_server(self) -> None:
        from shared_tensor.server import SharedTensorServer

        if self._server is not None:
            return
        if self.verbose_debug:
            logger.debug(
                "Provider starting autostart server",
                extra={
                    "socket_path": resolve_runtime_socket_path(self.base_path, self.device_index),
                },
            )
        self._server = SharedTensorServer(
            self,
            socket_path=resolve_runtime_socket_path(self.base_path, self.device_index),
            startup_timeout=self.server_startup_timeout,
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

    def _register_atexit_once(self) -> None:
        if self._atexit_registered:
            return
        atexit.register(self.close)
        self._atexit_registered = True
