from __future__ import annotations

import pytest

from shared_tensor import SharedTensorProvider
from shared_tensor.errors import (
    SharedTensorConfigurationError,
    SharedTensorProviderError,
)


def test_provider_registers_function_name_endpoints() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    def add(a: int, b: int) -> int:
        return a + b

    provider.register(add)
    assert provider.invoke_local("add", args=(2, 3)) == 5
    metadata = provider.list_endpoints()["add"]
    assert metadata["cache"] is True
    assert metadata["cache_format_key"] == (
        "test_provider_registers_function_name_endpoints.<locals>.add"
    )
    assert metadata["execution"] == "direct"
    assert metadata["concurrency"] == "parallel"
    assert metadata["singleflight"] is True


def test_provider_rejects_duplicate_function_names() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    def dup() -> None:
        return None

    provider.register(dup)
    with pytest.raises(SharedTensorProviderError):
        provider.register(dup)


def test_provider_rejects_unknown_execution_mode() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    with pytest.raises(SharedTensorConfigurationError):
        provider.register(lambda: None, execution="invalid")  # type: ignore[arg-type]


def test_provider_rejects_unknown_concurrency_mode() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    with pytest.raises(SharedTensorConfigurationError):
        provider.register(lambda: None, concurrency="invalid")  # type: ignore[arg-type]


def test_share_decorator_local_mode_uses_provider_wrapper() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    @provider.share
    def trim(value: str) -> str:
        return value.strip()

    assert trim("  x ") == "x"
    assert provider.invoke_local("trim", args=("  x ",)) == "x"


def test_server_mode_returns_original_function() -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def inc(value: int) -> int:
        return value + 1

    assert inc(3) == 4


def test_client_mode_wrapper_calls_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SharedTensorProvider(execution_mode="client")

    calls = []

    class FakeClient:
        def call(self, endpoint, *args, **kwargs):
            calls.append((endpoint, args, kwargs))
            return "ok"

        def close(self):
            return None

    provider._client = FakeClient()

    @provider.share
    def wrapped(value: int) -> int:
        return value + 1

    assert wrapped(3) == "ok"
    assert calls == [("wrapped", (3,), {})]


def test_client_mode_wrapper_exposes_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SharedTensorProvider(execution_mode="client")

    calls = []

    class FakeAsyncClient:
        def submit(self, endpoint, *args, **kwargs):
            calls.append((endpoint, args, kwargs))
            return "task-1"

        def close(self):
            return None

    provider._async_client = FakeAsyncClient()

    @provider.share(execution="task")
    def load_model(size: int) -> None:
        return None

    assert load_model.submit(4) == "task-1"
    assert calls == [("load_model", (4,), {})]


def test_auto_mode_defaults_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHARED_TENSOR_ROLE", raising=False)
    provider = SharedTensorProvider()
    assert provider.execution_mode == "client"
    assert provider.auto_mode is True


def test_env_server_role_autostarts_and_restarts_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARED_TENSOR_ROLE", "server")
    monkeypatch.setenv("SHARED_TENSOR_BASE_PORT", "34567")

    events: list[tuple[str, int]] = []

    class FakeServer:
        def __init__(self, provider, *, host, port, verbose_debug=False):
            assert provider.execution_mode == "server"
            assert host == "127.0.0.1"
            assert port == 34567
            self.port = port

        def start(self, blocking=False):
            assert blocking is False
            events.append(("start", self.port))

        def stop(self):
            events.append(("stop", self.port))

    monkeypatch.setattr("shared_tensor.server.SharedTensorServer", FakeServer)

    provider = SharedTensorProvider()

    @provider.share
    def first():
        return "first"

    @provider.share
    def second():
        return "second"

    assert provider.execution_mode == "server"
    assert first() == "first"
    assert second() == "second"
    assert events == [("start", 34567), ("stop", 34567), ("start", 34567)]

    provider.close()
    assert events == [("start", 34567), ("stop", 34567), ("start", 34567), ("stop", 34567)]


def test_provider_defers_port_resolution_until_client_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SharedTensorProvider(execution_mode="client", base_port=2500)

    calls: list[tuple[int, int | None]] = []

    class FakeClient:
        def __init__(self, *, base_port, host, device_index, timeout, verbose_debug):
            del host, timeout, verbose_debug
            calls.append((base_port, device_index))

        def close(self):
            return None

        def call(self, endpoint, *args, **kwargs):
            return (endpoint, args, kwargs)

    monkeypatch.setattr("shared_tensor.client.SharedTensorClient", FakeClient)

    @provider.share
    def load_model() -> None:
        return None

    assert calls == []
    assert load_model() == ("load_model", (), {})
    assert calls == [(2500, None)]


def test_provider_accepts_explicit_device_index_for_port_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARED_TENSOR_ROLE", "server")

    observed_ports: list[int] = []

    class FakeServer:
        def __init__(self, provider, *, host, port, verbose_debug=False):
            del provider, host, verbose_debug
            observed_ports.append(port)

        def start(self, blocking=False):
            assert blocking is False

        def stop(self):
            return None

    monkeypatch.setattr("shared_tensor.server.SharedTensorServer", FakeServer)

    provider = SharedTensorProvider(base_port=3000, device_index=3)

    @provider.share
    def build() -> None:
        return None

    assert observed_ports == [3003]


def test_provider_local_cache_reuses_result_by_default_for_same_call() -> None:
    provider = SharedTensorProvider(execution_mode="local")
    calls = {"value": 0}

    @provider.share
    def cached(value: int) -> int:
        calls["value"] += 1
        return value * 2

    assert cached(3) == 6
    assert cached(3) == 6
    assert calls["value"] == 1


def test_provider_local_cache_can_be_disabled() -> None:
    provider = SharedTensorProvider(execution_mode="local")
    calls = {"value": 0}

    @provider.share(cache=False)
    def uncached(value: int) -> int:
        calls["value"] += 1
        return value * 2

    assert uncached(3) == 6
    assert uncached(4) == 8
    assert calls["value"] == 2


def test_provider_cache_format_key_uses_bound_arguments() -> None:
    provider = SharedTensorProvider(execution_mode="local")
    calls = {"value": 0}

    @provider.share(cache_format_key="{factor}")
    def scaled(tensor, factor: int = 1):
        calls["value"] += 1
        return factor

    assert provider.invoke_local("scaled", args=("left",), kwargs={"factor": 2}) == 2
    assert provider.invoke_local("scaled", args=("right",), kwargs={"factor": 2}) == 2
    assert calls["value"] == 1


def test_provider_cache_format_key_rejects_unknown_fields() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    @provider.share(cache_format_key="{missing}")
    def broken(value: int) -> int:
        return value

    with pytest.raises(SharedTensorConfigurationError):
        provider.invoke_local("broken", args=(1,))


def test_provider_lists_managed_endpoint_metadata() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    @provider.share(managed=True, execution="task", concurrency="serialized")
    def load_model(value: int) -> int:
        return value

    metadata = provider.list_endpoints()["load_model"]
    assert metadata["managed"] is True
    assert metadata["cache"] is True
    assert metadata["cache_format_key"] == "test_provider_lists_managed_endpoint_metadata.<locals>.load_model"
    assert metadata["execution"] == "task"
    assert metadata["concurrency"] == "serialized"
    assert metadata["singleflight"] is True
