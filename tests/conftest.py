from __future__ import annotations

import os
import socket
import tempfile
import time
from collections.abc import Iterator

import pytest

from shared_tensor import SharedTensorClient, SharedTensorServer


@pytest.fixture
def running_server() -> Iterator:
    servers: list[tuple[SharedTensorServer, str]] = []

    def factory(provider, **kwargs):
        base_dir = tempfile.mkdtemp(prefix="shared-tensor-")
        base_path = os.path.join(base_dir, "runtime")
        provider.base_path = base_path
        server = SharedTensorServer(provider, **kwargs)
        server.start(blocking=False)
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.2)
                    sock.connect(server.socket_path)
                break
            except OSError:
                state = server.server_thread
                if state is not None and state.error is not None:
                    raise RuntimeError(
                        f"shared_tensor server thread exited before becoming ready: {state.error}"
                    ) from state.error
                time.sleep(0.05)
        else:
            raise TimeoutError(f"Timed out waiting for server socket {server.socket_path}")
        servers.append((server, base_dir))
        return server

    yield factory

    for server, base_dir in reversed(servers):
        server.stop()
        try:
            os.rmdir(base_dir)
        except OSError:
            pass


@pytest.fixture
def client_for_server():
    clients: list[SharedTensorClient] = []

    def factory(server: SharedTensorServer, **kwargs) -> SharedTensorClient:
        client = SharedTensorClient(
            base_path=server.provider.base_path,
            device_index=server.provider.device_index,
            **kwargs,
        )
        clients.append(client)
        return client

    yield factory

    for client in reversed(clients):
        client.close()
