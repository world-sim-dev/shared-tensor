from __future__ import annotations

import pytest

from shared_tensor import SharedObjectHandle, SharedTensorClient
from shared_tensor.errors import (
    SharedTensorClientError,
    SharedTensorProtocolError,
    SharedTensorRemoteError,
)
from shared_tensor.utils import CONTROL_ENCODING, serialize_empty_payload

pytest.importorskip("torch")


def test_client_decode_rpc_payload_returns_none_for_empty_result() -> None:
    client = SharedTensorClient(device_index=0)
    try:
        assert client._decode_rpc_payload({"encoding": None, "payload_bytes": None}) is None
    finally:
        client.close()


def test_client_decode_rpc_payload_rejects_missing_payload_bytes() -> None:
    client = SharedTensorClient(device_index=0)
    try:
        with pytest.raises(SharedTensorProtocolError, match="missing 'payload_bytes'"):
            client._decode_rpc_payload({"encoding": CONTROL_ENCODING})
    finally:
        client.close()


def test_client_decode_rpc_payload_wraps_managed_result() -> None:
    client = SharedTensorClient(device_index=0)
    try:
        _, payload = serialize_empty_payload(())
        result = client._decode_rpc_payload(
            {
                "encoding": CONTROL_ENCODING,
                "payload_bytes": payload,
                "object_id": "obj-123",
            }
        )
        assert isinstance(result, SharedObjectHandle)
        assert result.object_id == "obj-123"
        assert result.value == ()
    finally:
        client.close()


def test_client_send_request_converts_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(base_path="/tmp/shared-tensor-test", device_index=0, timeout=0.5)
    try:
        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def settimeout(self, timeout):
                return None

            def connect(self, path):
                raise TimeoutError()

        monkeypatch.setattr("socket.socket", lambda *args, **kwargs: FakeSocket())
        with pytest.raises(SharedTensorClientError, match="Timed out"):
            client._request("ping")
    finally:
        client.close()


def test_client_send_request_converts_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(base_path="/tmp/shared-tensor-test", device_index=0)
    try:
        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def settimeout(self, timeout):
                return None

            def connect(self, path):
                raise OSError("boom")

        monkeypatch.setattr("socket.socket", lambda *args, **kwargs: FakeSocket())
        with pytest.raises(SharedTensorClientError, match="Failed to contact"):
            client._request("ping")
    finally:
        client.close()


def test_client_send_request_surfaces_remote_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        monkeypatch.setattr(
            client,
            "_send_request",
            lambda request: (_ for _ in ()).throw(
                SharedTensorRemoteError("Remote error [3]: boom (extra)")
            ),
        )
        with pytest.raises(SharedTensorRemoteError, match="boom") as exc_info:
            client._request("ping")
        assert "extra" in str(exc_info.value)
    finally:
        client.close()


def test_client_send_request_preserves_structured_remote_error_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def settimeout(self, timeout):
            return None

        def connect(self, path):
            return None

    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: FakeSocket())
    monkeypatch.setattr(
        "shared_tensor.client.send_message",
        lambda sock, payload: None,
    )
    monkeypatch.setattr(
        "shared_tensor.client.recv_message",
        lambda sock: {
            "ok": False,
            "error": {"code": 5, "message": "boom", "type": "SharedTensorTaskError", "data": None},
        },
    )

    try:
        with pytest.raises(SharedTensorRemoteError) as exc_info:
            client._request("ping")
        assert exc_info.value.code == 5
        assert exc_info.value.error_type == "SharedTensorTaskError"
    finally:
        client.close()


def test_client_ping_returns_false_on_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        monkeypatch.setattr(
            client,
            "_request",
            lambda *args, **kwargs: (_ for _ in ()).throw(SharedTensorClientError("down")),
        )
        assert client.ping() is False
    finally:
        client.close()


def test_client_ping_returns_false_on_remote_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        monkeypatch.setattr(
            client,
            "_request",
            lambda *args, **kwargs: (_ for _ in ()).throw(SharedTensorRemoteError("bad")),
        )
        assert client.ping() is False
    finally:
        client.close()


def test_client_release_related_helpers_forward_results(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        responses = {
            "release_object": {"released": True},
            "release_objects": {"released": {"a": True, "b": False}},
            "get_object_info": {"object": {"object_id": "a"}},
            "get_server_info": {"server": "SharedTensorServer"},
            "list_endpoints": {"load_model": {"managed": True}},
        }
        monkeypatch.setattr(client, "_request", lambda method, params=None: responses[method])
        assert client.release("a") is True
        assert client.release_many(["a", "b"]) == {"a": True, "b": False}
        assert client.get_object_info("a") == {"object_id": "a"}
        assert client.get_server_info() == {"server": "SharedTensorServer"}
        assert client.list_endpoints() == {"load_model": {"managed": True}}
    finally:
        client.close()
