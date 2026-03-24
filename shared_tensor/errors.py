"""Exception hierarchy for shared_tensor."""

from __future__ import annotations

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


class SharedTensorTaskError(SharedTensorError):
    """Raised for async task lifecycle failures."""


class SharedTensorClientError(SharedTensorError):
    """Raised when the client cannot talk to the server."""


class SharedTensorServerError(SharedTensorError):
    """Raised for local server-side failures."""


class SharedTensorProviderError(SharedTensorError):
    """Raised for provider registration or invocation problems."""
