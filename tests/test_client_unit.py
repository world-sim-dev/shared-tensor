from __future__ import annotations

import pytest

from shared_tensor import SharedObjectHandle, SharedTensorClient
from shared_tensor.errors import SharedTensorProtocolError
from shared_tensor.utils import CONTROL_ENCODING, serialize_empty_payload

torch = pytest.importorskip("torch")


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
