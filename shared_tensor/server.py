"""Unix domain socket server for explicitly registered endpoints."""

from __future__ import annotations

import cloudpickle
import logging
import multiprocessing as mp
import os
import socket
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
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
from shared_tensor.managed_object import ManagedObjectRegistry
from shared_tensor.provider import EndpointDefinition, SharedTensorProvider
from shared_tensor.transport import recv_message, send_message
from shared_tensor.utils import (
    CONTROL_ENCODING,
    build_cache_key,
    capability_snapshot,
    deserialize_payload,
    resolve_runtime_socket_path,
    serialize_payload,
    unlink_socket_path,
    validate_call_payload_for_transport,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _InFlightCall:
    future: Future[dict[str, Any]]


class SharedTensorServer:
    def __init__(
        self,
        provider: SharedTensorProvider | None = None,
        *,
        socket_path: str | None = None,
        max_request_bytes: int = 64 * 1024 * 1024,
        max_workers: int = 4,
        result_ttl: float = 3600.0,
        verbose_debug: bool = False,
    ) -> None:
        self.provider = provider or SharedTensorProvider(execution_mode="server")
        self.socket_path = socket_path or resolve_runtime_socket_path(
            self.provider.base_path,
            self.provider.device_index,
        )
        self.max_request_bytes = max_request_bytes
        self.verbose_debug = verbose_debug
        self.max_workers = max_workers
        self.result_ttl = result_ttl
        self.listener: socket.socket | None = None
        self.server_process: Any | None = None
        self.running = False
        self.started_at: float | None = None
        self.stats = {
            "requests_processed": 0,
            "errors_encountered": 0,
        }
        self._task_manager: TaskManager | None = None
        self._cache: dict[str, dict[str, Any]] = {}
        self._managed_objects = ManagedObjectRegistry()
        self._inflight: dict[str, _InFlightCall] = {}
        self._endpoint_locks: dict[str, threading.Lock] = {}
        self._coordination_lock = threading.RLock()

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            method = request.get("method")
            if not isinstance(method, str) or not method:
                raise SharedTensorProtocolError("Missing required field 'method'")
            params = request.get("params") or {}
            if not isinstance(params, dict):
                raise SharedTensorProtocolError("'params' must be an object when provided")
            self.stats["requests_processed"] += 1
            return {"ok": True, "result": self._dispatch(method, params)}
        except Exception as exc:
            self.stats["errors_encountered"] += 1
            logger.exception("Failed to process request")
            return {
                "ok": False,
                "error": {
                    "code": self._error_code_for(exc),
                    "message": str(exc),
                },
            }

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "ping":
            return {"pong": True, "timestamp": time.time()}
        if method == "get_server_info":
            return self._get_server_info()
        if method == "list_endpoints":
            return self.provider.list_endpoints()
        if method == "call":
            return self._handle_call(params)
        if method == "submit":
            return self._handle_submit(params)
        if method == "get_task":
            return self._task_manager_instance().get(self._require_task_id(params)).to_dict()
        if method == "get_task_result":
            return self._task_manager_instance().result_payload(self._require_task_id(params))
        if method == "wait_task":
            return self._wait_task(params)
        if method == "cancel_task":
            task_id = self._require_task_id(params)
            return {"task_id": task_id, "cancelled": self._task_manager_instance().cancel(task_id)}
        if method == "list_tasks":
            status = params.get("status")
            status_enum = TaskStatus(status) if status else None
            return {
                task_id: info.to_dict()
                for task_id, info in self._task_manager_instance().list(status=status_enum).items()
            }
        if method == "release_object":
            return self._handle_release_object(params)
        if method == "release_objects":
            return self._handle_release_objects(params)
        if method == "get_object_info":
            return self._handle_get_object_info(params)
        raise SharedTensorProtocolError(f"Unknown RPC method '{method}'")

    def _handle_call(self, params: dict[str, Any]) -> dict[str, Any]:
        endpoint, args, kwargs = self._decode_call_params(params)
        definition = self.provider.get_endpoint(endpoint)
        if definition.execution == "task":
            task_info = self._submit_endpoint_task(endpoint, definition, args, kwargs)
            return self._task_manager_instance().wait_result_payload(task_info.task_id)
        return self._execute_endpoint_call(endpoint, definition, args, kwargs)

    def _handle_submit(self, params: dict[str, Any]) -> dict[str, Any]:
        endpoint, args, kwargs = self._decode_call_params(params)
        definition = self.provider.get_endpoint(endpoint)
        return self._submit_endpoint_task(endpoint, definition, args, kwargs).to_dict()

    def _wait_task(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = self._require_task_id(params)
        timeout = params.get("timeout")
        if timeout is not None and not isinstance(timeout, (int, float)):
            raise SharedTensorProtocolError("'timeout' must be a number when provided")
        try:
            self._task_manager_instance().wait_result_payload(task_id, timeout=timeout)
        except SharedTensorTaskError:
            info = self._task_manager_instance().get(task_id)
            if info.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                return info.to_dict()
            raise
        return self._task_manager_instance().get(task_id).to_dict()

    def _submit_endpoint_task(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return self._task_manager_instance().submit(
            endpoint,
            self._execute_endpoint_call,
            (endpoint, definition, args, kwargs),
            {},
            result_encoder=lambda payload: payload,
        )

    def _execute_endpoint_call(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        cache_key = self._cache_key(endpoint, definition, args, kwargs)
        if cache_key is not None:
            cached = self._lookup_cached_result(definition, cache_key)
            if cached is not None:
                return cached

        inflight_key = cache_key if cache_key is not None and definition.singleflight else None
        if inflight_key is not None:
            future, owner = self._acquire_inflight(inflight_key)
            if not owner:
                if definition.managed:
                    payload = future.result()
                    object_id = payload.get("object_id")
                    if object_id is not None:
                        self._managed_objects.add_ref(object_id)
                    return payload
                return future.result()
        else:
            future = None

        try:
            encoded = self._run_endpoint_under_policy(endpoint, definition, args, kwargs, cache_key)
        except Exception as exc:
            if future is not None:
                future.set_exception(exc)
                self._release_inflight(inflight_key, future)
            raise

        if future is not None:
            future.set_result(encoded)
            self._release_inflight(inflight_key, future)
        return encoded

    def _run_endpoint_under_policy(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        cache_key: str | None,
    ) -> dict[str, Any]:
        if definition.concurrency == "serialized":
            lock = self._endpoint_lock(endpoint)
            with lock:
                cached = self._lookup_cached_result(definition, cache_key)
                if cached is not None:
                    return cached
                return self._materialize_endpoint_result(endpoint, definition, args, kwargs, cache_key)
        return self._materialize_endpoint_result(endpoint, definition, args, kwargs, cache_key)

    def _materialize_endpoint_result(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        cache_key: str | None,
    ) -> dict[str, Any]:
        if definition.managed:
            return self._materialize_managed_result(endpoint, definition, args, kwargs, cache_key)
        value = definition.func(*args, **kwargs)
        result = self._encode_result(value)
        if cache_key is not None:
            self._cache[cache_key] = result
        return result

    def _materialize_managed_result(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        cache_key: str | None,
    ) -> dict[str, Any]:
        if cache_key is not None:
            cached = self._managed_objects.get_cached(cache_key)
            if cached is not None:
                self._managed_objects.add_ref(cached.object_id)
                return self._encode_result(cached.value, object_id=cached.object_id)

        result = definition.func(*args, **kwargs)
        entry = self._managed_objects.register(endpoint=endpoint, value=result, cache_key=cache_key)
        return self._encode_result(entry.value, object_id=entry.object_id)

    def _lookup_cached_result(
        self,
        definition: EndpointDefinition,
        cache_key: str | None,
    ) -> dict[str, Any] | None:
        if cache_key is None:
            return None
        if definition.managed:
            cached = self._managed_objects.get_cached(cache_key)
            if cached is None:
                return None
            self._managed_objects.add_ref(cached.object_id)
            return self._encode_result(cached.value, object_id=cached.object_id)
        return self._cache.get(cache_key)

    def _cache_key(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str | None:
        if not definition.cache:
            return None
        return build_cache_key(
            endpoint,
            args,
            kwargs,
            func=definition.func,
            cache_format_key=definition.cache_format_key,
        )

    def _acquire_inflight(self, inflight_key: str) -> tuple[Future[dict[str, Any]], bool]:
        with self._coordination_lock:
            inflight = self._inflight.get(inflight_key)
            if inflight is not None:
                return inflight.future, False
            future: Future[dict[str, Any]] = Future()
            self._inflight[inflight_key] = _InFlightCall(future=future)
            return future, True

    def _release_inflight(self, inflight_key: str | None, future: Future[dict[str, Any]]) -> None:
        if inflight_key is None:
            return
        with self._coordination_lock:
            current = self._inflight.get(inflight_key)
            if current is not None and current.future is future:
                self._inflight.pop(inflight_key, None)

    def _endpoint_lock(self, endpoint: str) -> threading.Lock:
        with self._coordination_lock:
            lock = self._endpoint_locks.get(endpoint)
            if lock is None:
                lock = threading.Lock()
                self._endpoint_locks[endpoint] = lock
            return lock

    def _handle_release_object(self, params: dict[str, Any]) -> dict[str, Any]:
        object_id = self._require_object_id(params)
        result = self._managed_objects.release(object_id)
        return {
            "object_id": object_id,
            "released": result.released,
            "destroyed": result.destroyed,
            "refcount": result.refcount,
        }

    def _handle_release_objects(self, params: dict[str, Any]) -> dict[str, Any]:
        object_ids = params.get("object_ids")
        if not isinstance(object_ids, list) or not all(
            isinstance(item, str) and item for item in object_ids
        ):
            raise SharedTensorProtocolError("Missing required parameter 'object_ids'")
        released: dict[str, bool] = {}
        for object_id in object_ids:
            released[object_id] = self._managed_objects.release(object_id).released
        return {"released": released}

    def _handle_get_object_info(self, params: dict[str, Any]) -> dict[str, Any]:
        object_id = self._require_object_id(params)
        return {"object": self._managed_objects.info(object_id)}

    def _decode_call_params(self, params: dict[str, Any]) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
        endpoint = params.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise SharedTensorProtocolError("Missing required parameter 'endpoint'")
        encoding = params.get("encoding")
        if not isinstance(encoding, str) or not encoding:
            raise SharedTensorProtocolError("Missing required parameter 'encoding'")
        args = deserialize_payload(encoding, params.get("args_bytes", b""))
        kwargs = deserialize_payload(encoding, params.get("kwargs_bytes", b""))
        if not isinstance(args, tuple):
            raise SharedTensorProtocolError("Decoded 'args_bytes' must produce a tuple")
        if not isinstance(kwargs, dict):
            raise SharedTensorProtocolError("Decoded 'kwargs_bytes' must produce a dict")
        if encoding == CONTROL_ENCODING:
            if args or kwargs:
                raise SharedTensorProtocolError(
                    "Control encoding is reserved for empty args/kwargs only"
                )
            return endpoint, args, kwargs
        validate_call_payload_for_transport(args)
        validate_call_payload_for_transport(kwargs, allow_dict_keys=True)
        return endpoint, args, kwargs

    def _encode_result(self, value: Any, *, object_id: str | None = None) -> dict[str, Any]:
        if value is None:
            return {"encoding": None, "payload_bytes": None, "object_id": object_id}
        encoding, payload = serialize_payload(value)
        return {"encoding": encoding, "payload_bytes": payload, "object_id": object_id}

    def _task_manager_instance(self) -> TaskManager:
        if self._task_manager is None:
            self._task_manager = TaskManager(
                max_workers=self.max_workers,
                result_ttl=self.result_ttl,
            )
        return self._task_manager

    @staticmethod
    def _require_task_id(params: dict[str, Any]) -> str:
        task_id = params.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise SharedTensorProtocolError("Missing required parameter 'task_id'")
        return task_id

    @staticmethod
    def _require_object_id(params: dict[str, Any]) -> str:
        object_id = params.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            raise SharedTensorProtocolError("Missing required parameter 'object_id'")
        return object_id

    def _get_server_info(self) -> dict[str, Any]:
        uptime = 0.0 if self.started_at is None else time.time() - self.started_at
        return {
            "server": "SharedTensorServer",
            "version": "0.2.4",
            "socket_path": self.socket_path,
            "uptime": uptime,
            "stats": dict(self.stats),
            "capabilities": capability_snapshot(),
            "endpoints": list(self.provider.list_endpoints().keys()),
        }

    def start(self, blocking: bool = True) -> None:
        if self.running:
            raise SharedTensorConfigurationError("Server is already running")
        if blocking:
            self._serve_forever()
            return
        if os.name != "posix":
            raise SharedTensorConfigurationError(
                "Non-blocking shared_tensor servers require POSIX multiprocessing support"
            )
        payload = cloudpickle.dumps(self.provider)
        process = mp.get_context("spawn").Process(
            target=self._serve_forever_from_payload,
            args=(
                payload,
                self.socket_path,
                self.max_request_bytes,
                self.max_workers,
                self.result_ttl,
                self.verbose_debug,
            ),
            name=f"shared-tensor-daemon:{self.socket_path}",
        )
        process.start()
        self.server_process = process
        self.running = True
        self.started_at = time.time()

    @staticmethod
    def _serve_forever_from_payload(
        payload: bytes,
        socket_path: str,
        max_request_bytes: int,
        max_workers: int,
        result_ttl: float,
        verbose_debug: bool,
    ) -> None:
        SharedTensorServer._configure_cuda_runtime()
        provider = cloudpickle.loads(payload)
        server = SharedTensorServer(
            provider,
            socket_path=socket_path,
            max_request_bytes=max_request_bytes,
            max_workers=max_workers,
            result_ttl=result_ttl,
            verbose_debug=verbose_debug,
        )
        server._serve_forever()

    def _serve_forever(self) -> None:
        self._configure_cuda_runtime()
        unlink_socket_path(self.socket_path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.socket_path)
        listener.listen()
        self.listener = listener
        self.running = True
        self.started_at = time.time()
        try:
            while self.running:
                try:
                    conn, _ = listener.accept()
                except OSError:
                    if self.running:
                        raise
                    break
                thread = threading.Thread(target=self._handle_connection, args=(conn,), daemon=True)
                thread.start()
        finally:
            self._shutdown_local_resources()

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                request = recv_message(conn)
                if not isinstance(request, dict):
                    raise SharedTensorProtocolError("Transport request must be a dict")
                response = self.process_request(request)
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": {
                        "code": self._error_code_for(exc),
                        "message": str(exc),
                    },
                }
            send_message(conn, response)

    @staticmethod
    def _configure_cuda_runtime() -> None:
        try:
            import torch
        except ImportError:
            return
        if not torch.cuda.is_available():
            return
        local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))
        if 0 <= local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)

    def stop(self) -> None:
        if not self.running:
            unlink_socket_path(self.socket_path)
            return
        self.running = False
        if self.server_process is not None:
            self.server_process.terminate()
            self.server_process.join(timeout=5)
            if self.server_process.is_alive():
                self.server_process.kill()
                self.server_process.join(timeout=5)
            self.server_process = None
            unlink_socket_path(self.socket_path)
            return
        if self.listener is not None:
            self.listener.close()
        self._shutdown_local_resources()

    def _shutdown_local_resources(self) -> None:
        if self.listener is not None:
            self.listener.close()
            self.listener = None
        if self._task_manager is not None:
            self._task_manager.shutdown(wait=False)
            self._task_manager = None
        self._managed_objects.clear()
        self._cache.clear()
        self._inflight.clear()
        self._endpoint_locks.clear()
        unlink_socket_path(self.socket_path)

    def __enter__(self) -> SharedTensorServer:
        self.start(blocking=False)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()

    @staticmethod
    def _error_code_for(exc: Exception) -> int:
        if isinstance(exc, SharedTensorProtocolError):
            return 1
        if isinstance(exc, SharedTensorProviderError):
            return 2
        if isinstance(exc, SharedTensorSerializationError):
            return 3
        if isinstance(exc, SharedTensorCapabilityError):
            return 4
        if isinstance(exc, SharedTensorTaskError):
            return 5
        if isinstance(exc, SharedTensorConfigurationError):
            return 6
        return 7
