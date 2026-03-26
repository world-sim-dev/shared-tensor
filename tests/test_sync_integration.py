from __future__ import annotations

import time
from threading import Thread

import pytest

from shared_tensor import SharedObjectHandle, SharedTensorProvider
from shared_tensor.errors import SharedTensorRemoteError
from shared_tensor.utils import resolve_runtime_socket_path

torch = pytest.importorskip("torch")


def test_sync_client_calls_registered_endpoint_with_empty_payload(
    running_server, client_for_server
) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def noop() -> None:
        return None

    server = running_server(provider)

    with client_for_server(server) as client:
        assert client.call("noop") is None
        info = client.get_server_info()
        assert info["server"] == "SharedTensorServer"
        assert info["socket_path"] == server.socket_path
        assert info["capabilities"]["transport"] == "same-host-cuda-torch-ipc"
        assert "noop" in info["endpoints"]


def test_sync_client_caches_empty_payload_results_server_side(running_server, client_for_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def cached() -> None:
        return None

    server = running_server(provider)

    with client_for_server(server) as client:
        assert client.call("cached") is None
        assert client.call("cached") is None


def test_sync_task_endpoint_waits_for_completion(running_server, client_for_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(execution="task")
    def delayed_none() -> None:
        time.sleep(0.05)
        return None

    server = running_server(provider, max_workers=1)

    started = time.time()
    with client_for_server(server) as client:
        assert client.call("delayed_none") is None
    assert time.time() - started >= 0.04


def test_sync_client_uses_cache_format_key_for_server_cache(running_server, client_for_server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(cache_format_key="{version}")
    def versioned(tensor: torch.Tensor, version: int) -> torch.Tensor:
        return torch.full((1,), float(version), device="cuda")

    server = running_server(provider)

    with client_for_server(server) as client:
        first = client.call("versioned", torch.ones(1, device="cuda"), version=3)
        second = client.call("versioned", torch.zeros(1, device="cuda"), version=3)

    assert torch.equal(first.cpu(), second.cpu())


def test_task_endpoint_singleflight_deduplicates_concurrent_calls(running_server, client_for_server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(execution="task", managed=True, cache_format_key="fixed")
    def load_once() -> torch.Tensor:
        time.sleep(0.15)
        return torch.arange(4, dtype=torch.float32, device="cuda")

    server = running_server(provider, max_workers=2)
    results: list[SharedObjectHandle] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with client_for_server(server) as client:
                handle = client.call("load_once")
                assert isinstance(handle, SharedObjectHandle)
                results.append(handle)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    first = Thread(target=worker)
    second = Thread(target=worker)
    first.start()
    second.start()
    first.join(timeout=3)
    second.join(timeout=3)

    assert errors == []
    assert len(results) == 2
    assert results[0].object_id == results[1].object_id

    with client_for_server(server) as client:
        info = client.get_object_info(results[0].object_id)
        assert info is not None
        assert info["refcount"] == 2

    assert results[0].release() is True
    assert results[1].release() is True


def test_task_endpoint_serialized_concurrency_runs_one_at_a_time(
    running_server, client_for_server
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(execution="task", concurrency="serialized", cache=False, singleflight=False)
    def serialized_job(tensor: torch.Tensor) -> torch.Tensor:
        time.sleep(0.15)
        return tensor

    server = running_server(provider, max_workers=2)
    errors: list[BaseException] = []

    def worker(value: float) -> None:
        try:
            with client_for_server(server) as client:
                tensor = torch.full((1,), value, device="cuda")
                result = client.call("serialized_job", tensor)
                assert float(result.cpu()[0]) == value
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    started = time.time()
    first = Thread(target=worker, args=(1.0,))
    second = Thread(target=worker, args=(2.0,))
    first.start()
    second.start()
    first.join(timeout=3)
    second.join(timeout=3)
    elapsed = time.time() - started

    assert errors == []
    assert elapsed >= 0.28


def test_managed_endpoint_returns_handle_and_supports_release(running_server, client_for_server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(managed=True, cache=False)
    def managed_tensor() -> torch.Tensor:
        return torch.arange(4, dtype=torch.float32, device="cuda")

    server = running_server(provider)

    with client_for_server(server) as client:
        handle = client.call("managed_tensor")
        assert isinstance(handle, SharedObjectHandle)
        assert handle.value.is_cuda
        info = client.get_object_info(handle.object_id)
        assert info is not None
        assert info["endpoint"] == "managed_tensor"
        assert info["refcount"] == 1
        assert handle.release() is True
        assert handle.released is True
        assert client.get_object_info(handle.object_id) is None
        assert handle.release() is False


def test_managed_cached_endpoint_reuses_object_id_and_refcount(running_server, client_for_server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(managed=True, cache_format_key="{hidden_size}")
    def cached_model(hidden_size: int) -> torch.nn.Module:
        return torch.nn.Linear(hidden_size, 2, device="cuda")

    server = running_server(provider)

    with client_for_server(server) as client:
        first = client.call("cached_model", hidden_size=4)
        second = client.call("cached_model", hidden_size=4)
        assert isinstance(first, SharedObjectHandle)
        assert isinstance(second, SharedObjectHandle)
        assert first.object_id == second.object_id
        info = client.get_object_info(first.object_id)
        assert info is not None
        assert info["refcount"] == 2
        assert first.release() is True
        info = client.get_object_info(second.object_id)
        assert info is not None
        assert info["refcount"] == 1
        assert second.release() is True
        assert client.get_object_info(second.object_id) is None


def test_managed_endpoint_without_cache_returns_distinct_handles(running_server, client_for_server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(managed=True, cache=False)
    def uncached_tensor() -> torch.Tensor:
        return torch.ones(1, device="cuda")

    server = running_server(provider)

    with client_for_server(server) as client:
        first = client.call("uncached_tensor")
        second = client.call("uncached_tensor")
        assert isinstance(first, SharedObjectHandle)
        assert isinstance(second, SharedObjectHandle)
        assert first.object_id != second.object_id
        released = client.release_many([first.object_id, second.object_id])
        assert released == {first.object_id: True, second.object_id: True}
        assert client.get_object_info(first.object_id) is None
        assert client.get_object_info(second.object_id) is None


def test_server_mode_managed_endpoint_stays_local() -> None:
    provider = SharedTensorProvider(execution_mode="server")

    class Token:
        pass

    token = Token()

    @provider.share(managed=True)
    def local_token() -> Token:
        return token

    assert local_token() is token


def test_auto_server_mode_calls_shared_function_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARED_TENSOR_ENABLED", "1")
    monkeypatch.setenv("SHARED_TENSOR_ROLE", "server")
    monkeypatch.setenv("SHARED_TENSOR_BASE_PATH", "/tmp/shared-tensor-auto")

    events: list[str] = []

    class FakeServer:
        def __init__(
            self,
            provider,
            *,
            socket_path,
            startup_timeout=30.0,
            verbose_debug=False,
        ):
            del provider, startup_timeout, verbose_debug
            events.append(f"init:{socket_path}")

        def start(self, blocking=False):
            events.append(f"start:{blocking}")

        def stop(self):
            events.append("stop")

    monkeypatch.setattr("shared_tensor.server.SharedTensorServer", FakeServer)

    provider = SharedTensorProvider()

    @provider.share
    def local_only() -> str:
        return "ok"

    assert local_only() == "ok"
    assert events == ["init:/tmp/shared-tensor-auto-0.sock", "start:False"]

    provider.close()
    assert events == ["init:/tmp/shared-tensor-auto-0.sock", "start:False", "stop"]


def test_client_uses_base_path_plus_device_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shared_tensor.utils.resolve_device_index", lambda device_index=None: 2)
    client = SharedTensorProvider(execution_mode="client", base_path="/tmp/shared-tensor-client-path")
    try:
        resolved = resolve_runtime_socket_path(client.base_path)
        assert resolved == "/tmp/shared-tensor-client-path-2.sock"
    finally:
        client.close()


def test_sync_client_reports_unknown_endpoint(running_server, client_for_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")
    server = running_server(provider)

    with client_for_server(server) as client:
        with pytest.raises(SharedTensorRemoteError):
            client.call("missing")


def test_sync_client_rejects_plain_python_result_payloads(running_server, client_for_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def plain() -> int:
        return 1

    server = running_server(provider)

    with client_for_server(server) as client:
        with pytest.raises(SharedTensorRemoteError):
            client.call("plain")


def test_sync_client_rejects_plain_python_result_payloads_with_scalar_arg(
    running_server, client_for_server
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def echo(tensor: torch.Tensor, value: int):
        del tensor
        return value

    server = running_server(provider)

    with client_for_server(server) as client:
        with pytest.raises(SharedTensorRemoteError):
            client.call("echo", torch.ones(1, device="cuda"), value=1)


def test_auto_server_mode_local_and_remote_calls_share_same_cache(
    running_server, client_for_server
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")
    calls = {"value": 0}

    @provider.share(cache_format_key="{version}")
    def build_tensor(tensor: torch.Tensor, version: int) -> torch.Tensor:
        calls["value"] += 1
        return torch.full((1,), float(version), device="cuda")

    server = running_server(provider)

    local_first = provider.call("build_tensor", torch.ones(1, device="cuda"), version=7)

    with client_for_server(server) as client:
        remote_second = client.call("build_tensor", torch.zeros(1, device="cuda"), version=7)

    assert calls["value"] == 1
    assert torch.equal(local_first.cpu(), remote_second.cpu())


def test_auto_server_mode_managed_local_and_remote_calls_share_same_object_registry(
    running_server, client_for_server
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    provider = SharedTensorProvider(execution_mode="server")
    calls = {"value": 0}

    @provider.share(managed=True, cache_format_key="{hidden_size}")
    def build_model(hidden_size: int) -> torch.nn.Module:
        calls["value"] += 1
        return torch.nn.Linear(hidden_size, 2, device="cuda")

    server = running_server(provider)

    local_model = provider.call("build_model", hidden_size=4)

    with client_for_server(server) as client:
        handle = client.call("build_model", hidden_size=4)
        assert isinstance(handle, SharedObjectHandle)
        info = client.get_object_info(handle.object_id)
        assert info is not None
        assert info["refcount"] == 2
        sample = torch.ones(1, 4, device="cuda")
        with handle as managed:
            remote_output = managed.value(sample)

    local_output = local_model(torch.ones(1, 4, device="cuda"))

    assert calls["value"] == 1
    assert torch.equal(local_output.cpu(), remote_output.cpu())


def test_thread_backed_local_client_surfaces_remote_style_errors(running_server, client_for_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def plain() -> int:
        return 1

    server = running_server(provider)

    with client_for_server(server) as client:
        with pytest.raises(SharedTensorRemoteError) as exc_info:
            client.call("plain")

    assert exc_info.value.code == 4
    assert exc_info.value.error_type == "SharedTensorCapabilityError"


def test_thread_backed_local_client_task_call_waits_for_completion(running_server, client_for_server) -> None:
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share(execution="task")
    def delayed_none() -> None:
        time.sleep(0.05)
        return None

    server = running_server(provider, max_workers=1)

    started = time.time()
    with client_for_server(server) as client:
        assert client.call("delayed_none") is None

    assert time.time() - started >= 0.04
