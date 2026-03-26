"""In-process runtime registry for thread-backed local servers."""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING

from shared_tensor.errors import SharedTensorConfigurationError

if TYPE_CHECKING:
    from shared_tensor.server import SharedTensorServer


_LOCK = RLock()
_SERVERS: dict[str, "SharedTensorServer"] = {}


def register_local_server(socket_path: str, server: "SharedTensorServer") -> None:
    with _LOCK:
        current = _SERVERS.get(socket_path)
        if current is not None and current is not server:
            raise SharedTensorConfigurationError(
                f"Local runtime socket '{socket_path}' is already registered by another server"
            )
        _SERVERS[socket_path] = server


def unregister_local_server(socket_path: str, server: "SharedTensorServer") -> None:
    with _LOCK:
        current = _SERVERS.get(socket_path)
        if current is server:
            _SERVERS.pop(socket_path, None)


def get_local_server(socket_path: str) -> "SharedTensorServer | None":
    with _LOCK:
        return _SERVERS.get(socket_path)
