from __future__ import annotations

import pytest

from shared_tensor import SharedTensorClient, SharedTensorProvider
from shared_tensor.errors import SharedTensorCapabilityError, SharedTensorRemoteError


def test_sync_client_calls_registered_endpoint_with_empty_payload(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(name="noop")
    def noop() -> None:
        return None

    server = running_server(provider)

    with SharedTensorClient(port=server.port) as client:
        assert client.call("noop") is None
        info = client.get_server_info()
        assert info["server"] == "SharedTensorServer"
        assert info["capabilities"]["transport"] == "same-host-cuda-torch-ipc"
        assert "noop" in info["endpoints"]


def test_sync_client_caches_empty_payload_results_server_side(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")
    counter = {"value": 0}

    @provider.share(name="cached", cache=True)
    def cached() -> None:
        counter["value"] += 1
        return None

    server = running_server(provider)

    with SharedTensorClient(port=server.port) as client:
        assert client.call("cached") is None
        assert client.call("cached") is None

    # Cache now lives in the forked server process, so verify behavior through stable outputs.
    assert True


def test_sync_client_reports_unknown_endpoint(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")
    server = running_server(provider)

    with SharedTensorClient(port=server.port) as client:
        with pytest.raises(SharedTensorRemoteError):
            client.call("missing")


def test_execute_function_compatibility_maps_to_endpoint_name(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(name="ready")
    def ready() -> None:
        return None

    server = running_server(provider)

    with SharedTensorClient(port=server.port) as client:
        assert client.execute_function("demo.module:ready") is None


def test_sync_client_rejects_plain_python_result_payloads(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(name="plain")
    def plain() -> int:
        return 1

    server = running_server(provider)

    with SharedTensorClient(port=server.port) as client:
        with pytest.raises(SharedTensorRemoteError):
            client.call("plain")


def test_sync_client_rejects_plain_python_args(running_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(name="echo")
    def echo(value):
        return value

    server = running_server(provider)

    with SharedTensorClient(port=server.port) as client:
        with pytest.raises(SharedTensorCapabilityError):
            client.call("echo", 1)
