"""Synchronous client for the shared_tensor local UDS transport."""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import Any, cast

from shared_tensor.async_task import TaskStatus
from shared_tensor.errors import (
    SharedTensorCapabilityError,
    SharedTensorClientError,
    SharedTensorConfigurationError,
    SharedTensorError,
    SharedTensorProviderError,
    SharedTensorProtocolError,
    SharedTensorRemoteError,
    SharedTensorSerializationError,
    SharedTensorStaleHandleError,
    SharedTensorTaskError,
)
from shared_tensor.managed_object import ReleaseHandle, SharedObjectHandle
from shared_tensor.runtime import get_local_server
from shared_tensor.transport import recv_message, send_message
from shared_tensor.utils import (
    deserialize_payload,
    resolve_runtime_socket_path,
    serialize_call_payloads,
    validate_payload_for_transport,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ClientReleaser(ReleaseHandle):
    client: SharedTensorClient
    object_id: str

    def release(self) -> bool:
        return self.client.release(self.object_id)

    def get_object_info(self) -> dict[str, Any] | None:
        return self.client.get_object_info(self.object_id)


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

    def _local_server(self):
        return get_local_server(self.socket_path)

    @staticmethod
    def _remote_error_from_local(exc: SharedTensorError) -> SharedTensorRemoteError:
        if isinstance(exc, SharedTensorProtocolError):
            code = 1
        elif isinstance(exc, SharedTensorProviderError):
            code = 2
        elif isinstance(exc, SharedTensorSerializationError):
            code = 3
        elif isinstance(exc, SharedTensorCapabilityError):
            code = 4
        elif isinstance(exc, SharedTensorTaskError):
            code = 5
        elif isinstance(exc, SharedTensorConfigurationError):
            code = 6
        elif isinstance(exc, SharedTensorStaleHandleError):
            code = 8
        else:
            code = 7
        return SharedTensorRemoteError(
            f"Remote error [{code}]: {exc}",
            code=code,
            data=None,
            error_type=type(exc).__name__,
        )

    def _run_local(self, operation):
        try:
            return operation()
        except SharedTensorError as exc:
            raise self._remote_error_from_local(exc) from exc

    def _decode_local_result(self, result: Any) -> Any:
        if result is None:
            return None
        value = result.value
        if value is None:
            return None
        validate_payload_for_transport(value, allow_dict_keys=isinstance(value, dict))
        object_id = result.object_id
        if object_id is None:
            return value
        return SharedObjectHandle(
            object_id=cast(str, object_id),
            value=value,
            _releaser=_ClientReleaser(client=self, object_id=cast(str, object_id)),
            server_id=self._infer_server_id(),
        )

    def _send_request(self, request: dict[str, Any]) -> Any:
        method = request.get("method", "<unknown>")
        if self.verbose_debug:
            logger.debug("Client sending request", extra={"method": method, "socket_path": self.socket_path})
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(self.socket_path)
                send_message(sock, request)
                response = recv_message(sock)
        except TimeoutError as exc:
            if self.verbose_debug:
                logger.warning("Client request timed out", extra={"method": method, "socket_path": self.socket_path})
            raise SharedTensorClientError(
                f"Timed out after {self.timeout} seconds while contacting {self.socket_path}"
            ) from exc
        except OSError as exc:
            if self.verbose_debug:
                logger.warning(
                    "Client request failed to connect",
                    extra={"method": method, "socket_path": self.socket_path, "error": str(exc)},
                )
            raise SharedTensorClientError(f"Failed to contact {self.socket_path}: {exc}") from exc

        if not isinstance(response, dict):
            raise SharedTensorProtocolError("Transport response must be a dict")
        if self.verbose_debug:
            logger.debug("Client received response", extra={"method": method, "ok": response.get("ok")})
        if response.get("ok") is not True:
            error = response.get("error") or {}
            message = error.get("message", "Unknown remote error")
            code = error.get("code")
            data = error.get("data")
            error_type = error.get("type")
            formatted = f"Remote error [{code}]: {message}" + (f" ({data})" if data else "")
            if self.verbose_debug:
                logger.warning(
                    "Client received remote error",
                    extra={"method": method, "code": code, "error_type": error_type},
                )
            raise SharedTensorRemoteError(
                formatted,
                code=code,
                data=data,
                error_type=error_type,
            )
        return response.get("result")

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return self._send_request({"method": method, "params": params or {}})

    def _infer_server_id(self) -> str | None:
        local_server = self._local_server()
        if local_server is not None:
            return cast(str | None, getattr(local_server, "server_id", None))
        try:
            return cast(str | None, self.get_server_info().get("server_id"))
        except (SharedTensorClientError, SharedTensorRemoteError, SharedTensorProtocolError):
            return None

    def call(self, endpoint: str, *args: Any, **kwargs: Any) -> Any:
        if self.verbose_debug:
            logger.debug("Client calling endpoint", extra={"endpoint": endpoint})
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: self._decode_local_result(
                    local_server.call_local_client(endpoint, args=tuple(args), kwargs=dict(kwargs))
                )
            )
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
        if self.verbose_debug:
            logger.debug("Client submitting task", extra={"endpoint": endpoint})
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: cast(
                    str,
                    local_server._submit_endpoint_task(
                        endpoint,
                        local_server.provider.get_endpoint(endpoint),
                        tuple(args),
                        dict(kwargs),
                    ).task_id,
                )
            )
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
        if self.verbose_debug:
            logger.debug("Client releasing managed object", extra={"object_id": object_id})
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: bool(local_server._handle_release_object({"object_id": object_id})["released"])
            )
        result = self._request("release_object", {"object_id": object_id})
        return bool(result["released"])

    def release_many(self, object_ids: list[str]) -> dict[str, bool]:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: {
                    object_id: bool(released)
                    for object_id, released in local_server._handle_release_objects({"object_ids": object_ids})[
                        "released"
                    ].items()
                }
            )
        result = self._request("release_objects", {"object_ids": object_ids})
        return {object_id: bool(released) for object_id, released in result["released"].items()}

    def get_object_info(self, object_id: str) -> dict[str, Any] | None:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: cast(
                    dict[str, Any] | None,
                    local_server._handle_get_object_info({"object_id": object_id}).get("object"),
                )
            )
        result = self._request("get_object_info", {"object_id": object_id})
        return cast(dict[str, Any] | None, result.get("object"))

    def ensure_handle_live(self, handle: SharedObjectHandle[Any], *, refresh: bool = True) -> dict[str, Any]:
        info = handle.get_object_info(refresh=refresh)
        if info is None:
            raise SharedTensorStaleHandleError(
                f"Managed object '{handle.object_id}' is no longer registered on the producer",
                object_id=handle.object_id,
                server_id=handle.server_id,
                reason="object_missing",
            )
        observed_server_id = cast(str | None, info.get("server_id"))
        if handle.server_id is not None and observed_server_id is not None and observed_server_id != handle.server_id:
            raise SharedTensorStaleHandleError(
                f"Managed object '{handle.object_id}' belongs to server '{handle.server_id}' but producer now reports '{observed_server_id}'",
                object_id=handle.object_id,
                server_id=handle.server_id,
                reason="server_mismatch",
            )
        return info

    def ping(self) -> bool:
        if self._local_server() is not None:
            return True
        try:
            self._request("ping")
        except (SharedTensorClientError, SharedTensorRemoteError):
            return False
        return True

    def get_server_info(self) -> dict[str, Any]:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(lambda: cast(dict[str, Any], local_server._get_server_info()))
        return cast(dict[str, Any], self._request("get_server_info"))

    def invalidate_call_cache(self, endpoint: str, *args: Any, **kwargs: Any) -> bool:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: bool(local_server.invalidate_call_cache(endpoint, args=tuple(args), kwargs=dict(kwargs)))
            )
        encoding, args_payload, kwargs_payload = serialize_call_payloads(tuple(args), dict(kwargs))
        result = self._request(
            "invalidate_call_cache",
            {
                "endpoint": endpoint,
                "args_bytes": args_payload,
                "kwargs_bytes": kwargs_payload,
                "encoding": encoding,
            },
        )
        return bool(result["invalidated"])

    def invalidate_endpoint_cache(self, endpoint: str) -> int:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(lambda: int(local_server.invalidate_endpoint_cache(endpoint)))
        result = self._request("invalidate_endpoint_cache", {"endpoint": endpoint})
        return int(result["invalidated"])

    def list_endpoints(self) -> dict[str, Any]:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(lambda: cast(dict[str, Any], local_server.provider.list_endpoints()))
        return cast(dict[str, Any], self._request("list_endpoints"))

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: cast(dict[str, Any], local_server._task_manager_instance().get(task_id).to_dict())
            )
        return cast(dict[str, Any], self._request("get_task", {"task_id": task_id}))

    def get_task_result(self, task_id: str) -> Any:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: self._decode_local_result(local_server.get_task_result_local(task_id))
            )
        return self._decode_rpc_payload(self._request("get_task_result", {"task_id": task_id}))

    def wait_task(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        if self.verbose_debug:
            logger.debug("Client waiting for task", extra={"task_id": task_id, "timeout": timeout})
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: cast(dict[str, Any], local_server.wait_task_local(task_id, timeout=timeout))
            )
        params = {"task_id": task_id}
        if timeout is not None:
            params["timeout"] = timeout
        return cast(dict[str, Any], self._request("wait_task", params))

    def cancel_task(self, task_id: str) -> bool:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(lambda: bool(local_server._task_manager_instance().cancel(task_id)))
        return bool(self._request("cancel_task", {"task_id": task_id})["cancelled"])

    def list_tasks(self, status: str | None = None) -> dict[str, Any]:
        local_server = self._local_server()
        if local_server is not None:
            return self._run_local(
                lambda: cast(
                    dict[str, Any],
                    {
                        listed_task_id: info.to_dict()
                        for listed_task_id, info in local_server._task_manager_instance()
                        .list(status=None if status is None else TaskStatus(status))
                        .items()
                    },
                )
            )
        params = {"status": status} if status else None
        return cast(dict[str, Any], self._request("list_tasks", params))

    def close(self) -> None:
        if self.verbose_debug:
            logger.debug("Client closed", extra={"socket_path": self.socket_path})
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
            server_id=cast(str | None, result.get("server_id")),
        )
