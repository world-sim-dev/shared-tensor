from __future__ import annotations

import pickle
import socket
import threading

import pytest

from shared_tensor.async_task import TaskStatus
from shared_tensor.errors import (
    SharedTensorCapabilityError,
    SharedTensorConfigurationError,
    SharedTensorProtocolError,
    SharedTensorProviderError,
    SharedTensorSerializationError,
    SharedTensorTaskError,
)
from shared_tensor.provider import SharedTensorProvider
from shared_tensor.server import SharedTensorServer
from shared_tensor.utils import CONTROL_ENCODING, serialize_empty_payload


def test_server_dispatch_handles_ping_and_unknown_method() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))

    result = server._dispatch("ping", {})

    assert result["pong"] is True
    with pytest.raises(SharedTensorProtocolError, match="Unknown RPC method"):
        server._dispatch("missing", {})


def test_server_dispatch_list_tasks_uses_status_filter() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))

    class FakeTaskInfo:
        def to_dict(self):
            return {"status": "completed"}

    class FakeTaskManager:
        def list(self, status=None):
            assert status == TaskStatus.COMPLETED
            return {"task-1": FakeTaskInfo()}

    server._task_manager = FakeTaskManager()

    assert server._dispatch("list_tasks", {"status": "completed"}) == {
        "task-1": {"status": "completed"}
    }


def test_server_decode_call_params_validates_control_encoding() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))
    _, args_payload = serialize_empty_payload(())
    _, kwargs_payload = serialize_empty_payload({})

    assert server._decode_call_params(
        {
            "endpoint": "noop",
            "encoding": CONTROL_ENCODING,
            "args_bytes": args_payload,
            "kwargs_bytes": kwargs_payload,
        }
    ) == ("noop", (), {})


def test_server_decode_call_params_rejects_missing_fields() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))

    with pytest.raises(SharedTensorProtocolError, match="'endpoint'"):
        server._decode_call_params({})
    with pytest.raises(SharedTensorProtocolError, match="'encoding'"):
        server._decode_call_params({"endpoint": "noop"})


def test_server_decode_call_params_rejects_non_empty_control_payload() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))

    with pytest.raises(SharedTensorProtocolError, match="reserved for empty args/kwargs"):
        args_payload = pickle.dumps((1,), protocol=pickle.HIGHEST_PROTOCOL)
        kwargs_payload = pickle.dumps({}, protocol=pickle.HIGHEST_PROTOCOL)
        server._decode_call_params(
            {
                "endpoint": "noop",
                "encoding": CONTROL_ENCODING,
                "args_bytes": args_payload,
                "kwargs_bytes": kwargs_payload,
            }
        )


def test_server_release_helpers_validate_inputs() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))

    with pytest.raises(SharedTensorProtocolError, match="'object_ids'"):
        server._handle_release_objects({"object_ids": [""]})
    with pytest.raises(SharedTensorProtocolError, match="'task_id'"):
        server._require_task_id({})
    with pytest.raises(SharedTensorProtocolError, match="'object_id'"):
        server._require_object_id({})


def test_server_handle_object_info_and_release() -> None:
    provider = SharedTensorProvider(execution_mode="server")
    server = SharedTensorServer(provider)
    entry = server._managed_objects.register(endpoint="load", value=object(), cache_key="k")

    assert server._handle_get_object_info({"object_id": entry.object_id})["object"]["endpoint"] == "load"
    assert server._handle_release_object({"object_id": entry.object_id}) == {
        "object_id": entry.object_id,
        "released": True,
        "destroyed": True,
        "refcount": 0,
    }


def test_server_get_server_info_contains_metadata() -> None:
    provider = SharedTensorProvider(execution_mode="server", base_path="/tmp/test-shared-tensor")

    @provider.share
    def load() -> None:
        return None

    server = SharedTensorServer(provider)
    info = server._get_server_info()

    assert info["server"] == "SharedTensorServer"
    assert info["socket_path"].endswith("-0.sock")
    assert info["running"] is False
    assert info["ready"] is False
    assert info["process_start_method"] is None
    assert info["device_index"] == 0
    assert "load" in info["endpoints"]
    assert info["capabilities"]["transport"] == "same-host-cuda-torch-ipc"


def test_server_error_code_mapping() -> None:
    assert SharedTensorServer._error_code_for(SharedTensorProtocolError("x"))
    assert SharedTensorServer._error_code_for(SharedTensorProviderError("x"))
    assert SharedTensorServer._error_code_for(SharedTensorSerializationError("x"))
    assert SharedTensorServer._error_code_for(SharedTensorCapabilityError("x"))
    assert SharedTensorServer._error_code_for(SharedTensorTaskError("x"))
    assert SharedTensorServer._error_code_for(SharedTensorConfigurationError("x"))
    assert SharedTensorServer._error_code_for(RuntimeError("x"))


def test_server_stop_is_idempotent() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))
    server.stop()
    server.stop()


def test_server_handle_connection_returns_error_frame_for_invalid_request() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))
    left, right = socket.socketpair()
    try:
        left.sendall(b"\x00\x00\x00\x05hello")
        left.shutdown(socket.SHUT_WR)
        server._handle_connection(right)
        payload = left.recv(4096)
    finally:
        left.close()
        right.close()

    assert payload


def test_server_serialize_error_includes_type() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))

    payload = server._serialize_error(SharedTensorTaskError("boom"))

    assert payload == {
        "code": 5,
        "message": "boom",
        "type": "SharedTensorTaskError",
        "data": None,
    }


def test_server_nonblocking_start_uses_background_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SharedTensorProvider(execution_mode="server", base_path="/tmp/shared-tensor-thread")
    server = SharedTensorServer(provider, startup_timeout=1.0)

    calls: list[threading.Event | None] = []

    def fake_serve_forever(*, started_event=None):
        calls.append(started_event)
        server.running = True
        if started_event is not None:
            started_event.set()

    monkeypatch.setattr(server, "_serve_forever", fake_serve_forever)

    server.start(blocking=False)

    assert server.server_thread is not None
    assert server._resolved_process_start_method == "thread"
    assert len(calls) == 1
    assert calls[0] is not None


def test_server_nonblocking_start_rejects_process_start_method() -> None:
    server = SharedTensorServer(
        SharedTensorProvider(execution_mode="server"),
        process_start_method="spawn",
    )

    with pytest.raises(SharedTensorConfigurationError, match="thread-backed non-blocking servers"):
        server.start(blocking=False)


def test_server_invoke_local_reuses_direct_cache() -> None:
    provider = SharedTensorProvider(execution_mode="server")
    server = SharedTensorServer(provider)
    calls = {"value": 0}

    class Token:
        pass

    token = Token()

    @provider.share(cache_format_key="{value}")
    def build(value: int):
        calls["value"] += 1
        return token

    assert server.invoke_local("build", args=(3,)) is token
    assert server.invoke_local("build", args=(3,)) is token
    assert calls["value"] == 1


def test_server_invoke_local_reuses_managed_registry() -> None:
    provider = SharedTensorProvider(execution_mode="server")
    server = SharedTensorServer(provider)
    calls = {"value": 0}

    token = object()

    @provider.share(managed=True, cache_format_key="{key}")
    def build(key: int):
        calls["value"] += 1
        return token

    first = server.invoke_local("build", kwargs={"key": 1})
    second = server.invoke_local("build", kwargs={"key": 1})

    assert first is token
    assert second is token
    assert calls["value"] == 1
    assert len(server._managed_objects._entries) == 1
