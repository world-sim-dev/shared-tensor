from __future__ import annotations

import socket
import time

import pytest

from shared_tensor import SharedTensorServer


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def running_server():
    servers = []

    def factory(provider, **kwargs):
        port = kwargs.pop("port", find_free_port())
        server = SharedTensorServer(provider, host="127.0.0.1", port=port, **kwargs)
        server.start(blocking=False)
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.01)
        servers.append(server)
        return server

    yield factory

    for server in reversed(servers):
        server.stop()
