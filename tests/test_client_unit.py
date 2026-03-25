from __future__ import annotations

import pytest
import requests

from shared_tensor import SharedObjectHandle, SharedTensorClient
from shared_tensor.errors import (
    SharedTensorClientError,
    SharedTensorProtocolError,
    SharedTensorRemoteError,
)
from shared_tensor.utils import CONTROL_ENCODING, serialize_empty_payload

torch = pytest.importorskip("torch")


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, text: str = '{"jsonrpc": "2.0", "id": "1", "result": {}}') -> None:
        self.status_code = status_code
        self.text = text


def test_client_decode_rpc_payload_returns_none_for_empty_result() -> None:
    client = SharedTensorClient(device_index=0)
    try:
        assert client._decode_rpc_payload({"encoding": None, "payload_hex": None}) is None
    finally:
        client.close()


def test_client_decode_rpc_payload_rejects_missing_payload_hex() -> None:
    client = SharedTensorClient(device_index=0)
    try:
        with pytest.raises(SharedTensorProtocolError, match="missing 'payload_hex'"):
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
                "payload_hex": payload.hex(),
                "object_id": "obj-123",
            }
        )

        assert isinstance(result, SharedObjectHandle)
        assert result.object_id == "obj-123"
        assert result.value == ()
    finally:
        client.close()


def test_client_send_request_converts_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        monkeypatch.setattr(
            client.session,
            "post",
            lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.Timeout()),
        )

        with pytest.raises(SharedTensorClientError, match="Timed out"):
            client._request("ping")
    finally:
        client.close()


def test_client_send_request_converts_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        monkeypatch.setattr(
            client.session,
            "post",
            lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.RequestException("boom")),
        )

        with pytest.raises(SharedTensorClientError, match="Failed to contact"):
            client._request("ping")
    finally:
        client.close()


def test_client_send_request_rejects_non_200_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        monkeypatch.setattr(
            client.session,
            "post",
            lambda *args, **kwargs: _FakeResponse(status_code=503, text="down"),
        )

        with pytest.raises(SharedTensorClientError, match="HTTP 503"):
            client._request("ping")
    finally:
        client.close()


def test_client_send_request_surfaces_remote_error_with_data(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        monkeypatch.setattr(
            client.session,
            "post",
            lambda *args, **kwargs: _FakeResponse(
                text='{"jsonrpc": "2.0", "id": "1", "error": {"code": -32005, "message": "boom", "data": "extra"}}'
            ),
        )

        with pytest.raises(SharedTensorRemoteError, match="boom") as exc_info:
            client._request("ping")

        assert "extra" in str(exc_info.value)
    finally:
        client.close()


def test_client_ping_returns_false_on_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        monkeypatch.setattr(client, "_request", lambda *args, **kwargs: (_ for _ in ()).throw(SharedTensorClientError("down")))
        assert client.ping() is False
    finally:
        client.close()


def test_client_ping_returns_false_on_remote_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SharedTensorClient(device_index=0)
    try:
        monkeypatch.setattr(client, "_request", lambda *args, **kwargs: (_ for _ in ()).throw(SharedTensorRemoteError("bad")))
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
