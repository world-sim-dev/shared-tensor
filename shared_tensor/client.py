"""Synchronous client for the shared_tensor local UDS transport."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, cast

from shared_tensor.errors import (
    SharedTensorClientError,
    SharedTensorProtocolError,
    SharedTensorRemoteError,
)
from shared_tensor.managed_object import ReleaseHandle, SharedObjectHandle
from shared_tensor.transport import recv_message, send_message
from shared_tensor.utils import (
    deserialize_payload,
    resolve_runtime_socket_path,
    serialize_call_payloads,
)


@dataclass(slots=True)
class _ClientReleaser(ReleaseHandle):
    client: SharedTensorClient
    object_id: str

    def release(self) -> bool:
        return self.client.release(self.object_id)


class SharedTensorClient:
    """UDS client for endpoint-oriented local RPC execution."""

    def __init__(
        self,
        base_path: str = "/tmp/shared-tensor",
        *,
        device_index: int | None = None,
        timeout: float = 30.0,
        verbose_debug: bool = False,
    ) -> None:
        self.base_path = base_path
        self.device_index = device_index
        self.socket_path = resolve_runtime_socket_path(self.base_path, self.device_index)
        self.timeout = timeout
        self.verbose_debug = verbose_debug

    def _send_request(self, request: dict[str, Any]) -> Any:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(self.socket_path)
                send_message(sock, request)
                response = recv_message(sock)
        except TimeoutError as exc:
            raise SharedTensorClientError(
                f"Timed out after {self.timeout} seconds while contacting {self.socket_path}"
            ) from exc
        except OSError as exc:
            raise SharedTensorClientError(f"Failed to contact {self.socket_path}: {exc}") from exc

        if not isinstance(response, dict):
            raise SharedTensorProtocolError("Transport response must be a dict")
        if response.get("ok") is not True:
            error = response.get("error") or {}
            message = error.get("message", "Unknown remote error")
            code = error.get("code")
            data = error.get("data")
            raise SharedTensorRemoteError(
                f"Remote error [{code}]: {message}" + (f" ({data})" if data else "")
            )
        return response.get("result")

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return self._send_request({"method": method, "params": params or {}})

    def call(self, endpoint: str, *args: Any, **kwargs: Any) -> Any:
        encoding, args_payload, kwargs_payload = serialize_call_payloads(tuple(args), dict(kwargs))
        result = self._request(
            "call",
            {
                "endpoint": endpoint,
                "args_bytes": args_payload,
                "kwargs_bytes": kwargs_payload,
                "encoding": encoding,
            },
        )
        return self._decode_rpc_payload(result)

    def submit(self, endpoint: str, *args: Any, **kwargs: Any) -> str:
        encoding, args_payload, kwargs_payload = serialize_call_payloads(tuple(args), dict(kwargs))
        result = self._request(
            "submit",
            {
                "endpoint": endpoint,
                "args_bytes": args_payload,
                "kwargs_bytes": kwargs_payload,
                "encoding": encoding,
            },
        )
        return cast(str, result["task_id"])

    def release(self, object_id: str) -> bool:
        result = self._request("release_object", {"object_id": object_id})
        return bool(result["released"])

    def release_many(self, object_ids: list[str]) -> dict[str, bool]:
        result = self._request("release_objects", {"object_ids": object_ids})
        return {object_id: bool(released) for object_id, released in result["released"].items()}

    def get_object_info(self, object_id: str) -> dict[str, Any] | None:
        result = self._request("get_object_info", {"object_id": object_id})
        return cast(dict[str, Any] | None, result.get("object"))

    def ping(self) -> bool:
        try:
            self._request("ping")
        except (SharedTensorClientError, SharedTensorRemoteError):
            return False
        return True

    def get_server_info(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("get_server_info"))

    def list_endpoints(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("list_endpoints"))

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("get_task", {"task_id": task_id}))

    def get_task_result(self, task_id: str) -> Any:
        return self._decode_rpc_payload(self._request("get_task_result", {"task_id": task_id}))

    def wait_task(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        params = {"task_id": task_id}
        if timeout is not None:
            params["timeout"] = timeout
        return cast(dict[str, Any], self._request("wait_task", params))

    def cancel_task(self, task_id: str) -> bool:
        return bool(self._request("cancel_task", {"task_id": task_id})["cancelled"])

    def list_tasks(self, status: str | None = None) -> dict[str, Any]:
        params = {"status": status} if status else None
        return cast(dict[str, Any], self._request("list_tasks", params))

    def close(self) -> None:
        return None

    def __enter__(self) -> SharedTensorClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def _decode_rpc_payload(self, result: dict[str, Any]) -> Any:
        encoding = result.get("encoding")
        payload_bytes = result.get("payload_bytes")
        if encoding is None:
            return None
        if payload_bytes is None:
            raise SharedTensorProtocolError("RPC response is missing 'payload_bytes'")
        value = deserialize_payload(encoding, payload_bytes)
        object_id = result.get("object_id")
        if object_id is None:
            return value
        return SharedObjectHandle(
            object_id=cast(str, object_id),
            value=value,
            _releaser=_ClientReleaser(client=self, object_id=cast(str, object_id)),
        )
