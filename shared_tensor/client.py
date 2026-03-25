"""Synchronous client for the shared_tensor local RPC server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import requests

from shared_tensor.errors import (
    SharedTensorClientError,
    SharedTensorProtocolError,
    SharedTensorRemoteError,
)
from shared_tensor.jsonrpc import JsonRpcRequest, parse_response
from shared_tensor.managed_object import ReleaseHandle, SharedObjectHandle
from shared_tensor.utils import (
    deserialize_payload,
    resolve_runtime_port,
    serialize_call_payloads,
)


@dataclass(slots=True)
class _ClientReleaser(ReleaseHandle):
    client: SharedTensorClient
    object_id: str

    def release(self) -> bool:
        return self.client.release(self.object_id)


class SharedTensorClient:
    """JSON-RPC client for endpoint-oriented local RPC execution."""

    def __init__(
        self,
        base_port: int = 2537,
        *,
        host: str = "127.0.0.1",
        device_index: int | None = None,
        timeout: float = 30.0,
        verbose_debug: bool = False,
    ) -> None:
        self.host = host
        self.base_port = base_port
        self.device_index = device_index
        self.port = resolve_runtime_port(self.base_port, self.device_index)
        self.timeout = timeout
        self.verbose_debug = verbose_debug
        self.server_url = f"http://{host}:{self.port}"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "shared-tensor/modernized-client",
            }
        )

    def _send_request(self, request: JsonRpcRequest) -> Any:
        try:
            response = self.session.post(
                f"{self.server_url}/jsonrpc",
                data=request.to_json(),
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise SharedTensorClientError(
                f"Timed out after {self.timeout} seconds while contacting {self.server_url}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise SharedTensorClientError(f"Failed to contact {self.server_url}: {exc}") from exc

        if response.status_code != 200:
            raise SharedTensorClientError(
                f"Server returned HTTP {response.status_code}: {response.text}"
            )

        parsed = parse_response(response.text)
        if parsed.error is not None:
            message = parsed.error.get("message", "Unknown remote error")
            code = parsed.error.get("code")
            data = parsed.error.get("data")
            raise SharedTensorRemoteError(f"Remote error [{code}]: {message}" + (f" ({data})" if data else ""))
        return parsed.result

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        return self._send_request(JsonRpcRequest(method=method, params=params))

    def call(self, endpoint: str, *args: Any, **kwargs: Any) -> Any:
        encoding, args_payload, kwargs_payload = serialize_call_payloads(tuple(args), dict(kwargs))
        result = self._request(
            "call",
            {
                "endpoint": endpoint,
                "args_hex": args_payload.hex(),
                "kwargs_hex": kwargs_payload.hex(),
                "encoding": encoding,
            },
        )
        return self._decode_rpc_payload(result)

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
        except SharedTensorClientError:
            return False
        except SharedTensorRemoteError:
            return False
        return True

    def get_server_info(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("get_server_info"))

    def list_endpoints(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._request("list_endpoints"))

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> SharedTensorClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def _decode_rpc_payload(self, result: dict[str, Any]) -> Any:
        encoding = result.get("encoding")
        payload_hex = result.get("payload_hex")
        if encoding is None:
            return None
        if payload_hex is None:
            raise SharedTensorProtocolError("RPC response is missing 'payload_hex'")
        value = deserialize_payload(encoding, payload_hex)
        object_id = result.get("object_id")
        if object_id is None:
            return value
        return SharedObjectHandle(
            object_id=cast(str, object_id),
            value=value,
            _releaser=_ClientReleaser(client=self, object_id=cast(str, object_id)),
        )
