from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
from collections.abc import Iterator

import pytest

from shared_tensor import SharedTensorClient, SharedTensorServer


@pytest.fixture
def running_server() -> Iterator:
    servers: list[tuple[SharedTensorServer, str, threading.Thread]] = []

    def factory(provider, **kwargs):
        base_dir = tempfile.mkdtemp(prefix="shared-tensor-")
        base_path = os.path.join(base_dir, "runtime")
        provider.base_path = base_path
        server = SharedTensorServer(provider, **kwargs)
        thread = threading.Thread(target=server.start, kwargs={"blocking": True}, daemon=True)
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.2)
                    sock.connect(server.socket_path)
                break
            except OSError:
                time.sleep(0.01)
        else:
            raise TimeoutError(f"Timed out waiting for server socket {server.socket_path}")
        servers.append((server, base_dir, thread))
        return server

    yield factory

    for server, base_dir, thread in reversed(servers):
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
