from __future__ import annotations

import socket

import pytest

from shared_tensor.errors import SharedTensorClientError, SharedTensorProtocolError
from shared_tensor.transport import decode_message, encode_message, recv_message, send_message


def test_transport_message_round_trip() -> None:
    payload = {"method": "ping", "params": {"value": 1}}
    encoded = encode_message(payload)
    assert decode_message(encoded[4:]) == payload


def test_transport_send_and_receive_over_socketpair() -> None:
    left, right = socket.socketpair()
    try:
        send_message(left, {"ok": True, "result": {"value": 1}})
        assert recv_message(right) == {"ok": True, "result": {"value": 1}}
    finally:
        left.close()
        right.close()


def test_decode_message_rejects_invalid_pickle() -> None:
    with pytest.raises(SharedTensorProtocolError, match="Failed to decode transport message"):
        decode_message(b"not-a-pickle")


def test_recv_message_rejects_zero_length_frame() -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(bytes.fromhex("00000000"))
        with pytest.raises(SharedTensorProtocolError, match="must be positive"):
            recv_message(right)
    finally:
        left.close()
        right.close()


def test_recv_message_rejects_closed_connection_mid_frame() -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(bytes.fromhex("000000056162"))
        left.close()
        with pytest.raises(SharedTensorClientError, match="Connection closed"):
            recv_message(right)
    finally:
        right.close()
