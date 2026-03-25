"""Unix domain socket framed transport for shared_tensor control messages."""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

from shared_tensor.errors import SharedTensorClientError, SharedTensorProtocolError

FRAME_HEADER = struct.Struct("!I")


def encode_message(payload: Any) -> bytes:
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    return FRAME_HEADER.pack(len(encoded)) + encoded


def decode_message(data: bytes) -> Any:
    try:
        return pickle.loads(data)
    except Exception as exc:  # pragma: no cover - defensive
        raise SharedTensorProtocolError(f"Failed to decode transport message: {exc}") from exc


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise SharedTensorClientError("Connection closed while receiving transport frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Any:
    header = recv_exact(sock, FRAME_HEADER.size)
    (size,) = FRAME_HEADER.unpack(header)
    if size <= 0:
        raise SharedTensorProtocolError("Transport frame size must be positive")
    return decode_message(recv_exact(sock, size))


def send_message(sock: socket.socket, payload: Any) -> None:
    sock.sendall(encode_message(payload))
