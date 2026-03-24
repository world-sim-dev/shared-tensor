"""Endpoint registration for sync shared_tensor clients and servers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from shared_tensor.errors import (
    SharedTensorConfigurationError,
    SharedTensorProviderError,
)
from shared_tensor.utils import infer_function_path


@dataclass(slots=True)
class EndpointDefinition:
    name: str
    func: Callable[..., Any]
    cache: bool = False
    legacy_path: str | None = None
    async_default_wait: bool = True


class SharedTensorProvider:
    """Explicit endpoint registry with optional client-side decorator wrappers."""

    def __init__(
        self,
        server_port: int = 2537,
        *,
        server_host: str = "127.0.0.1",
        timeout: float = 30.0,
        execution_mode: str = "client",
        verbose_debug: bool = False,
    ) -> None:
        if execution_mode not in {"client", "server", "local"}:
            raise SharedTensorConfigurationError(
                "execution_mode must be one of 'client', 'server', or 'local'"
            )
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.execution_mode = execution_mode
        self.verbose_debug = verbose_debug
        self._client: Any | None = None
        self._endpoints: dict[str, EndpointDefinition] = {}
        self._registered_functions = self._endpoints

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        cache: bool = False,
        async_default_wait: bool = True,
    ) -> Callable[..., Any]:
        endpoint_name = name or func.__name__
        if endpoint_name in self._endpoints:
            raise SharedTensorProviderError(f"Endpoint '{endpoint_name}' is already registered")

        definition = EndpointDefinition(
            name=endpoint_name,
            func=func,
            cache=cache,
            legacy_path=infer_function_path(func),
            async_default_wait=async_default_wait,
        )
        self._endpoints[endpoint_name] = definition

        if self.execution_mode in {"server", "local"}:
            return func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(endpoint_name, *args, **kwargs)

        return wrapper

    def share(
        self,
        name: str | None = None,
        cache: bool = False,
        **_: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return self.register(func, name=name, cache=cache)

        return decorator

    def call(self, endpoint: str, *args: Any, **kwargs: Any) -> Any:
        if self.execution_mode == "local":
            return self.invoke_local(endpoint, args=args, kwargs=kwargs)
        client = self._get_client()
        return client.call(endpoint, *args, **kwargs)

    def invoke_local(
        self,
        endpoint: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        definition = self.get_endpoint(endpoint)
        return definition.func(*args, **(kwargs or {}))

    def get_endpoint(self, endpoint: str) -> EndpointDefinition:
        try:
            return self._endpoints[endpoint]
        except KeyError as exc:
            raise SharedTensorProviderError(f"Endpoint '{endpoint}' is not registered") from exc

    def list_endpoints(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "cache": definition.cache,
                "async_default_wait": definition.async_default_wait,
                "legacy_path": definition.legacy_path,
            }
            for name, definition in self._endpoints.items()
        }

    def _get_client(self) -> Any:
        if self._client is None:
            from shared_tensor.client import SharedTensorClient

            self._client = SharedTensorClient(
                port=self.server_port,
                host=self.server_host,
                timeout=self.timeout,
                verbose_debug=self.verbose_debug,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
