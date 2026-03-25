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


def test_client_mode_wrapper_calls_endpoint() -> None:
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


def test_client_mode_wrapper_exposes_submit() -> None:
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


def test_provider_submit_rejects_local_and_server_modes() -> None:
    local_provider = SharedTensorProvider(execution_mode="local")
    server_provider = SharedTensorProvider(execution_mode="server")

    with pytest.raises(RuntimeError, match="do not support task submission"):
        local_provider.submit("missing")
    with pytest.raises(RuntimeError, match="do not support task submission"):
        server_provider.submit("missing")


def test_provider_execute_wait_false_returns_task_id(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SharedTensorProvider(execution_mode="client")
    monkeypatch.setattr(provider, "submit", lambda endpoint, *args, **kwargs: "task-1")

    assert provider.execute("load", wait=False) == "task-1"


def test_provider_execute_wait_true_uses_wait_for_task(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SharedTensorProvider(execution_mode="client")
    monkeypatch.setattr(provider, "submit", lambda endpoint, *args, **kwargs: "task-1")
    monkeypatch.setattr(
        provider,
        "wait_for_task",
        lambda task_id, timeout=None, callback=None: (task_id, timeout, callback),
    )

    assert provider.execute("load", wait=True, timeout=2) == ("task-1", 2, None)


def test_provider_task_helpers_forward_to_async_client() -> None:
    provider = SharedTensorProvider(execution_mode="client")

    class FakeAsyncClient:
        def get_task_status(self, task_id):
            return {"status": task_id}

        def get_task_result(self, task_id):
            return {"result": task_id}

        def wait_for_task(self, task_id, timeout=None, callback=None):
            return (task_id, timeout, callback)

        def cancel_task(self, task_id):
            return task_id == "t1"

        def list_tasks(self, status=None):
            return {"status": status}

        def close(self):
            return None

    provider._async_client = FakeAsyncClient()

    assert provider.get_task_status("t1") == {"status": "t1"}
    assert provider.get_task_result("t1") == {"result": "t1"}
    assert provider.wait_for_task("t1", timeout=1) == ("t1", 1, None)
    assert provider.cancel_task("t1") is True
    assert provider.list_tasks(status="done") == {"status": "done"}


def test_provider_close_closes_all_resources() -> None:
    provider = SharedTensorProvider(execution_mode="client")
    events: list[str] = []

    class FakeClient:
        def close(self):
            events.append("client")

    class FakeAsyncClient:
        def close(self):
            events.append("async")

    class FakeServer:
        def stop(self):
            events.append("server")

    provider._client = FakeClient()
    provider._async_client = FakeAsyncClient()
    provider._server = FakeServer()

    provider.close()

    assert events == ["client", "async", "server"]
    assert provider._client is None
    assert provider._async_client is None
    assert provider._server is None


def test_provider_get_endpoint_rejects_missing_endpoint() -> None:
    provider = SharedTensorProvider(execution_mode="local")

    with pytest.raises(SharedTensorProviderError, match="not registered"):
        provider.get_endpoint("missing")


def test_auto_mode_defaults_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARED_TENSOR_ENABLED", "1")
    monkeypatch.delenv("SHARED_TENSOR_ROLE", raising=False)
    provider = SharedTensorProvider()
    assert provider.execution_mode == "client"
    assert provider.auto_mode is True


def test_auto_mode_defaults_to_local_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHARED_TENSOR_ENABLED", raising=False)
    monkeypatch.delenv("SHARED_TENSOR_ROLE", raising=False)
    provider = SharedTensorProvider()
    assert provider.execution_mode == "local"
    assert provider.auto_mode is True


def test_auto_mode_provider_enabled_true_overrides_disabled_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SHARED_TENSOR_ENABLED", raising=False)
    monkeypatch.delenv("SHARED_TENSOR_ROLE", raising=False)
    provider = SharedTensorProvider(enabled=True)
    assert provider.execution_mode == "client"
    assert provider.auto_mode is True


def test_auto_mode_provider_enabled_false_overrides_enabled_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHARED_TENSOR_ENABLED", "1")
    monkeypatch.setenv("SHARED_TENSOR_ROLE", "server")
    provider = SharedTensorProvider(enabled=False)
    assert provider.execution_mode == "local"
    assert provider.auto_mode is True


def test_env_server_role_autostarts_and_restarts_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARED_TENSOR_ENABLED", "1")
    monkeypatch.setenv("SHARED_TENSOR_ROLE", "server")
    monkeypatch.setenv("SHARED_TENSOR_BASE_PATH", "/tmp/shared-tensor-provider")

    events: list[tuple[str, str]] = []

    class FakeServer:
        def __init__(self, provider, *, socket_path, verbose_debug=False):
            assert provider.execution_mode == "server"
            assert socket_path == "/tmp/shared-tensor-provider-0.sock"
            self.socket_path = socket_path

        def start(self, blocking=False):
            assert blocking is False
            events.append(("start", self.socket_path))

        def stop(self):
            events.append(("stop", self.socket_path))

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
    assert events == [
        ("start", "/tmp/shared-tensor-provider-0.sock"),
        ("stop", "/tmp/shared-tensor-provider-0.sock"),
        ("start", "/tmp/shared-tensor-provider-0.sock"),
    ]

    provider.close()
    assert events[-1] == ("stop", "/tmp/shared-tensor-provider-0.sock")


def test_provider_defers_client_construction_until_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SharedTensorProvider(execution_mode="client", base_path="/tmp/shared-tensor-client")

    calls: list[tuple[str, int | None]] = []

    class FakeClient:
        def __init__(self, *, base_path, device_index, timeout, verbose_debug):
            del timeout, verbose_debug
            calls.append((base_path, device_index))

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
    assert calls == [("/tmp/shared-tensor-client", None)]


def test_provider_accepts_explicit_device_index_for_socket_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHARED_TENSOR_ENABLED", "1")
    monkeypatch.setenv("SHARED_TENSOR_ROLE", "server")
    monkeypatch.setenv("SHARED_TENSOR_BASE_PATH", "/tmp/shared-tensor-device")

    observed_paths: list[str] = []

    class FakeServer:
        def __init__(self, provider, *, socket_path, verbose_debug=False):
            del provider, verbose_debug
            observed_paths.append(socket_path)

        def start(self, blocking=False):
            assert blocking is False

        def stop(self):
            return None

    monkeypatch.setattr("shared_tensor.server.SharedTensorServer", FakeServer)

    provider = SharedTensorProvider(device_index=3)

    @provider.share
    def build() -> None:
        return None

    assert observed_paths == ["/tmp/shared-tensor-device-3.sock"]


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
