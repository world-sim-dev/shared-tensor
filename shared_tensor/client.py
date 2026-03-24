"""Synchronous client for the shared_tensor local RPC server."""

from __future__ import annotations

from typing import Any, cast

import requests

from shared_tensor.errors import (
    SharedTensorClientError,
    SharedTensorProtocolError,
    SharedTensorRemoteError,
)
from shared_tensor.jsonrpc import JsonRpcRequest, parse_response
from shared_tensor.utils import (
    deserialize_payload,
    resolve_legacy_endpoint_name,
    serialize_call_payloads,
)


class SharedTensorClient:
    """JSON-RPC client for endpoint-oriented local RPC execution."""

    def __init__(
        self,
        port: int = 2537,
        *,
        host: str = "127.0.0.1",
        timeout: float = 30.0,
        verbose_debug: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.verbose_debug = verbose_debug
        self.server_url = f"http://{host}:{port}"
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

    def execute_function(
        self,
        function_path: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        del options
        endpoint = resolve_legacy_endpoint_name(function_path)
        return self.call(endpoint, *(args or ()), **(kwargs or {}))

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

    def list_functions(self) -> dict[str, Any]:
        return self.list_endpoints()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> SharedTensorClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    @staticmethod
    def _decode_rpc_payload(result: dict[str, Any]) -> Any:
        encoding = result.get("encoding")
        payload_hex = result.get("payload_hex")
        if encoding is None:
            return None
        if payload_hex is None:
            raise SharedTensorProtocolError("RPC response is missing 'payload_hex'")
        return deserialize_payload(encoding, payload_hex)


def execute_remote_function(
    function_path: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    server_port: int = 2537,
    host: str = "127.0.0.1",
    timeout: float = 30.0,
    verbose_debug: bool = False,
) -> Any:
    with SharedTensorClient(
        port=server_port,
        host=host,
        timeout=timeout,
        verbose_debug=verbose_debug,
    ) as client:
        return client.execute_function(function_path, args=args, kwargs=kwargs)
