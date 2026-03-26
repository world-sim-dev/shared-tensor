"""Exception hierarchy for shared_tensor."""

from __future__ import annotations

from typing import Any

__all__ = [
    "SharedTensorError",
    "SharedTensorConfigurationError",
    "SharedTensorProtocolError",
    "SharedTensorSerializationError",
    "SharedTensorCapabilityError",
    "SharedTensorRemoteError",
    "SharedTensorTaskError",
    "SharedTensorClientError",
    "SharedTensorServerError",
    "SharedTensorProviderError",
    "SharedTensorStaleHandleError",
]


class SharedTensorError(Exception):
    """Base exception for the package."""


class SharedTensorConfigurationError(SharedTensorError):
    """Raised when local configuration is invalid."""


class SharedTensorProtocolError(SharedTensorError):
    """Raised for malformed RPC payloads or unsupported protocol behavior."""


class SharedTensorSerializationError(SharedTensorError):
    """Raised when a payload cannot be encoded or decoded safely."""


class SharedTensorCapabilityError(SharedTensorError):
    """Raised when the runtime cannot satisfy a requested feature."""


class SharedTensorRemoteError(SharedTensorError):
    """Raised when the remote endpoint reports an application error."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        data: Any = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.data = data
        self.error_type = error_type


class SharedTensorTaskError(SharedTensorError):
    """Raised for async task lifecycle failures."""


class SharedTensorClientError(SharedTensorError):
    """Raised when the client cannot talk to the server."""


class SharedTensorServerError(SharedTensorError):
    """Raised for local server-side failures."""


class SharedTensorProviderError(SharedTensorError):
    """Raised for provider registration or invocation problems."""


class SharedTensorStaleHandleError(SharedTensorError):
    """Raised when a managed handle can no longer be trusted."""

    def __init__(
        self,
        message: str,
        *,
        object_id: str | None = None,
        server_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.object_id = object_id
        self.server_id = server_id
        self.reason = reason
