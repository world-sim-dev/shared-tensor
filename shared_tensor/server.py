"""HTTP JSON-RPC server for explicitly registered endpoints."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

from shared_tensor.async_task import TaskManager, TaskStatus
from shared_tensor.errors import (
    SharedTensorCapabilityError,
    SharedTensorConfigurationError,
    SharedTensorProtocolError,
    SharedTensorProviderError,
    SharedTensorSerializationError,
    SharedTensorTaskError,
)
from shared_tensor.jsonrpc import (
    JsonRpcErrorCodes,
    create_error_response,
    create_success_response,
    parse_request,
)
from shared_tensor.provider import SharedTensorProvider
from shared_tensor.utils import (
    CONTROL_ENCODING,
    build_cache_key,
    capability_snapshot,
    deserialize_payload,
    load_object,
    serialize_payload,
    validate_payload_for_transport,
)

logger = logging.getLogger(__name__)


class SharedTensorRequestHandler(BaseHTTPRequestHandler):
    server_version = "shared-tensor"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_error(404, "Not Found")
            return
        payload = self.server.shared_tensor_server.health_response()  # type: ignore[attr-defined]
        body = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/jsonrpc":
            self.send_error(404, "Not Found")
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("application/json"):
            self.send_error(400, "Content-Type must be application/json")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length header")
            return
        server = self.server.shared_tensor_server  # type: ignore[attr-defined]
        if content_length > server.max_request_bytes:
            self.send_error(413, "Request body too large")
            return
        raw = self.rfile.read(content_length).decode("utf-8")
        response = server.process_jsonrpc_request(raw)
        body = response.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class SharedTensorServer:
    def __init__(
        self,
        provider: SharedTensorProvider | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 2537,
        max_request_bytes: int = 64 * 1024 * 1024,
        max_workers: int = 4,
        result_ttl: float = 3600.0,
        verbose_debug: bool = False,
    ) -> None:
        self.provider = provider or SharedTensorProvider(execution_mode="server")
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.verbose_debug = verbose_debug
        self.server: ThreadedHTTPServer | None = None
        self.server_thread: threading.Thread | None = None
        self.running = False
        self.started_at: float | None = None
        self.stats = {
            "requests_processed": 0,
            "errors_encountered": 0,
        }
        self._task_manager = TaskManager(max_workers=max_workers, result_ttl=result_ttl)
        self._cache: dict[str, dict[str, str | None]] = {}

    def health_response(self) -> str:
        from json import dumps

        return dumps({"status": "healthy", "timestamp": time.time()})

    def process_jsonrpc_request(self, raw: str) -> str:
        request_id: str | int | None = None
        try:
            request = parse_request(raw)
            request_id = request.id
            self.stats["requests_processed"] += 1
            result = self._dispatch(request.method, request.params or {})
            return create_success_response(request_id, result).to_json()
        except Exception as exc:
            self.stats["errors_encountered"] += 1
            logger.exception("Failed to process request")
            code = self._error_code_for(exc)
            return create_error_response(request_id, code, str(exc)).to_json()

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "ping":
            return {"pong": True, "timestamp": time.time()}
        if method == "get_server_info":
            return self._get_server_info()
        if method == "list_endpoints":
            return self.provider.list_endpoints()
        if method == "list_functions":
            return self.provider.list_endpoints()
        if method == "call":
            return self._handle_call(params)
        if method == "submit":
            return self._handle_submit(params)
        if method == "get_task":
            return self._task_manager.get(self._require_task_id(params)).to_dict()
        if method == "get_task_result":
            return self._encode_result(self._task_manager.result(self._require_task_id(params)))
        if method == "cancel_task":
            task_id = self._require_task_id(params)
            return {"task_id": task_id, "cancelled": self._task_manager.cancel(task_id)}
        if method == "list_tasks":
            status = params.get("status")
            status_enum = TaskStatus(status) if status else None
            return {
                task_id: info.to_dict()
                for task_id, info in self._task_manager.list(status=status_enum).items()
            }
        raise SharedTensorProtocolError(f"Unknown JSON-RPC method '{method}'")

    def _handle_call(self, params: dict[str, Any]) -> dict[str, str | None]:
        endpoint, args, kwargs = self._decode_call_params(params)
        definition = self.provider.get_endpoint(endpoint)
        cache_key = None
        if definition.cache:
            cache_key = build_cache_key(endpoint, args, kwargs)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        result = definition.func(*args, **kwargs)
        encoded = self._encode_result(result)
        if cache_key is not None:
            self._cache[cache_key] = encoded
        return encoded

    def _handle_submit(self, params: dict[str, Any]) -> dict[str, Any]:
        endpoint, args, kwargs = self._decode_call_params(params)
        definition = self.provider.get_endpoint(endpoint)
        info = self._task_manager.submit(endpoint, definition.func, args, kwargs)
        return info.to_dict()

    def _decode_call_params(self, params: dict[str, Any]) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        endpoint = params.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise SharedTensorProtocolError("Missing required parameter 'endpoint'")
        encoding = params.get("encoding")
        if not isinstance(encoding, str) or not encoding:
            raise SharedTensorProtocolError("Missing required parameter 'encoding'")
        args = deserialize_payload(encoding, params.get("args_hex", ""))
        kwargs = deserialize_payload(encoding, params.get("kwargs_hex", ""))
        if not isinstance(args, tuple):
            raise SharedTensorProtocolError("Decoded 'args_hex' must produce a tuple")
        if not isinstance(kwargs, dict):
            raise SharedTensorProtocolError("Decoded 'kwargs_hex' must produce a dict")
        if encoding == CONTROL_ENCODING:
            if args or kwargs:
                raise SharedTensorProtocolError(
                    "Control encoding is reserved for empty args/kwargs only"
                )
            return endpoint, args, kwargs
        validate_payload_for_transport(args)
        validate_payload_for_transport(kwargs, allow_dict_keys=True)
        return endpoint, args, kwargs

    def _encode_result(self, value: Any) -> dict[str, str | None]:
        if value is None:
            return {"encoding": None, "payload_hex": None}
        encoding, payload = serialize_payload(value)
        return {"encoding": encoding, "payload_hex": payload.hex()}

    @staticmethod
    def _require_task_id(params: dict[str, Any]) -> str:
        task_id = params.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise SharedTensorProtocolError("Missing required parameter 'task_id'")
        return task_id

    def _get_server_info(self) -> dict[str, Any]:
        uptime = 0.0 if self.started_at is None else time.time() - self.started_at
        return {
            "server": "SharedTensorServer",
            "version": "0.2.0",
            "host": self.host,
            "port": self.port,
            "uptime": uptime,
            "stats": dict(self.stats),
            "capabilities": capability_snapshot(),
            "endpoints": list(self.provider.list_endpoints().keys()),
        }

    def start(self, blocking: bool = True) -> None:
        if self.running:
            raise SharedTensorConfigurationError("Server is already running")
        self.server = ThreadedHTTPServer((self.host, self.port), SharedTensorRequestHandler)
        self.server.shared_tensor_server = self  # type: ignore[attr-defined]
        self.running = True
        self.started_at = time.time()
        if blocking:
            self.server.serve_forever()
            return
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread is not None and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
        self._task_manager.shutdown(wait=False)
        self.running = False

    def __enter__(self) -> SharedTensorServer:
        self.start(blocking=False)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()

    @staticmethod
    def _error_code_for(exc: Exception) -> int:
        if isinstance(exc, SharedTensorProtocolError):
            return JsonRpcErrorCodes.INVALID_PARAMS
        if isinstance(exc, SharedTensorProviderError):
            return JsonRpcErrorCodes.ENDPOINT_NOT_FOUND
        if isinstance(exc, SharedTensorSerializationError):
            return JsonRpcErrorCodes.SERIALIZATION_ERROR
        if isinstance(exc, SharedTensorCapabilityError):
            return JsonRpcErrorCodes.CAPABILITY_ERROR
        if isinstance(exc, SharedTensorTaskError):
            return JsonRpcErrorCodes.TASK_ERROR
        if isinstance(exc, SharedTensorConfigurationError):
            return JsonRpcErrorCodes.INTERNAL_ERROR
        return JsonRpcErrorCodes.REMOTE_ERROR


def load_server_target(target: str, *, host: str, port: int) -> SharedTensorServer:
    obj = load_object(target)
    if isinstance(obj, SharedTensorServer):
        return obj
    if callable(obj):
        obj = obj()
    if isinstance(obj, SharedTensorServer):
        return obj
    if isinstance(obj, SharedTensorProvider):
        return SharedTensorServer(obj, host=host, port=port)
    raise SharedTensorConfigurationError(
        "Target must resolve to a SharedTensorProvider, SharedTensorServer, or a factory returning one"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a shared_tensor endpoint server")
    parser.add_argument("--provider", required=True, help="module:attribute for a provider/server or factory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2537)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    server = load_server_target(args.provider, host=args.host, port=args.port)
    try:
        server.start(blocking=True)
    except KeyboardInterrupt:
        logger.info("Shutting down shared_tensor server")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
