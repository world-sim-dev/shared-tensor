from __future__ import annotations

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
    with pytest.raises(SharedTensorProtocolError, match="Unknown JSON-RPC method"):
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

    assert server._dispatch("list_tasks", {"status": "completed"}) == {"task-1": {"status": "completed"}}


def test_server_decode_call_params_validates_control_encoding() -> None:
    server = SharedTensorServer(SharedTensorProvider(execution_mode="server"))
    _, args_payload = serialize_empty_payload(())
    _, kwargs_payload = serialize_empty_payload({})

    assert server._decode_call_params(
        {
            "endpoint": "noop",
            "encoding": CONTROL_ENCODING,
            "args_hex": args_payload.hex(),
            "kwargs_hex": kwargs_payload.hex(),
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
        server._decode_call_params(
            {
                "endpoint": "noop",
                "encoding": CONTROL_ENCODING,
                "args_hex": "80059505000000000000004b0185942e",
                "kwargs_hex": "80057d942e",
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
    provider = SharedTensorProvider(execution_mode="server")

    @provider.share
    def load() -> None:
        return None

    server = SharedTensorServer(provider, port=9999)
    info = server._get_server_info()

    assert info["server"] == "SharedTensorServer"
    assert info["port"] == 9999
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
