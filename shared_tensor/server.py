"""Unix domain socket server for explicitly registered endpoints."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from shared_tensor.async_task import TaskManager, TaskStatus
from shared_tensor.errors import (
    SharedTensorCapabilityError,
    SharedTensorConfigurationError,
    SharedTensorProtocolError,
    SharedTensorProviderError,
    SharedTensorSerializationError,
    SharedTensorStaleHandleError,
    SharedTensorTaskError,
)
from shared_tensor.managed_object import ManagedObjectRegistry
from shared_tensor.provider import EndpointDefinition, SharedTensorProvider
from shared_tensor.runtime import register_local_server, unregister_local_server
from shared_tensor.transport import recv_message, send_message
from shared_tensor.utils import (
    CONTROL_ENCODING,
    build_cache_key,
    capability_snapshot,
    deserialize_payload,
    resolve_device_index,
    resolve_runtime_socket_path,
    serialize_payload,
    unlink_socket_path,
    validate_call_payload_for_transport,
)

logger = logging.getLogger(__name__)


class _ConnectionExecutor:
    def __init__(self, *, max_workers: int) -> None:
        self._semaphore = threading.BoundedSemaphore(max_workers)

    def submit(self, func, *args, **kwargs) -> threading.Thread:
        self._semaphore.acquire()

        def runner() -> None:
            try:
                func(*args, **kwargs)
            finally:
                self._semaphore.release()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return thread


def _server_version() -> str:
    try:
        from shared_tensor import __version__
    except ImportError:
        return "unknown"
    return __version__


@dataclass(slots=True)
class _InFlightCall:
    future: Future


@dataclass(slots=True)
class _ServerThreadState:
    thread: threading.Thread
    ready: threading.Event = field(default_factory=threading.Event)
    stopped: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass(slots=True)
class _EndpointResult:
    value: Any
    object_id: str | None = None


class SharedTensorServer:
    def __init__(
        self,
        provider: SharedTensorProvider | None = None,
        *,
        socket_path: str | None = None,
        max_request_bytes: int = 64 * 1024 * 1024,
        max_workers: int = 4,
        result_ttl: float = 3600.0,
        process_start_method: str | None = None,
        startup_timeout: float = 30.0,
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
        self.process_start_method = process_start_method
        self.startup_timeout = startup_timeout
        self.listener: socket.socket | None = None
        self.server_process: Any | None = None
        self.server_id = uuid.uuid4().hex
        self.server_thread: _ServerThreadState | None = None
        self._resolved_process_start_method: str | None = None
        self.running = False
        self.started_at: float | None = None
        self.stats = {
            "requests_processed": 0,
            "errors_encountered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "task_submissions": 0,
            "cache_invalidations": 0,
        }
        self._task_manager: TaskManager | None = None
        self._cache: dict[str, str] = {}
        self._local_cache: dict[str, Any] = {}
        self._managed_objects = ManagedObjectRegistry()
        self._inflight: dict[str, _InFlightCall] = {}
        self._endpoint_locks: dict[str, threading.Lock] = {}
        self._coordination_lock = threading.RLock()
        self._connection_executor = _ConnectionExecutor(max_workers=max_workers)
        self._accepting_requests = True
        if getattr(self.provider, "_server", None) is None:
            self.provider._server = self

    def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.verbose_debug:
            logger.debug("Server processing request", extra={"method": request.get("method")})
        try:
            if not self._accepting_requests:
                raise SharedTensorConfigurationError("Server is stopping and not accepting new requests")
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
            return {"ok": False, "error": self._serialize_error(exc)}

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
        if method == "invalidate_call_cache":
            return self._handle_invalidate_call_cache(params)
        if method == "invalidate_endpoint_cache":
            return self._handle_invalidate_endpoint_cache(params)
        raise SharedTensorProtocolError(f"Unknown RPC method '{method}'")

    def _handle_call(self, params: dict[str, Any]) -> dict[str, Any]:
        endpoint, args, kwargs = self._decode_call_params(params)
        if self.verbose_debug:
            logger.debug("Server handling call", extra={"endpoint": endpoint})
        definition = self.provider.get_endpoint(endpoint)
        if definition.execution == "task":
            task_info = self._submit_endpoint_task(endpoint, definition, args, kwargs)
            return self._task_manager_instance().wait_result_payload(task_info.task_id)
        return self._execute_endpoint_call(endpoint, definition, args, kwargs)

    def _handle_submit(self, params: dict[str, Any]) -> dict[str, Any]:
        endpoint, args, kwargs = self._decode_call_params(params)
        if self.verbose_debug:
            logger.debug("Server handling submit", extra={"endpoint": endpoint})
        definition = self.provider.get_endpoint(endpoint)
        return self._submit_endpoint_task(endpoint, definition, args, kwargs).to_dict()

    def _wait_task(self, params: dict[str, Any]) -> dict[str, Any]:
        task_id = self._require_task_id(params)
        if self.verbose_debug:
            logger.debug("Server waiting for task", extra={"task_id": task_id})
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
        self.stats["task_submissions"] += 1
        return self._task_manager_instance().submit(
            endpoint,
            self._execute_endpoint_result,
            (endpoint, definition, args, kwargs),
            {},
            result_encoder=self._encode_endpoint_result,
        )

    def _execute_endpoint_result(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _EndpointResult:
        cache_key = self._cache_key(endpoint, definition, args, kwargs)
        if cache_key is not None:
            cached = self._lookup_cached_result_value(definition, cache_key)
            if cached is not None:
                self.stats["cache_hits"] += 1
                if self.verbose_debug:
                    logger.debug("Server cache hit", extra={"endpoint": endpoint, "cache_key": cache_key})
                return cached
            self.stats["cache_misses"] += 1

        inflight_key = cache_key if cache_key is not None and definition.singleflight else None
        if inflight_key is not None:
            future, owner = self._acquire_inflight(inflight_key)
            if self.verbose_debug and owner:
                logger.debug("Server created singleflight entry", extra={"endpoint": endpoint, "cache_key": inflight_key})
            if not owner:
                result = future.result()
                if definition.managed and result.object_id is not None:
                    self._managed_objects.add_ref(result.object_id)
                return result
        else:
            future = None

        try:
            result = self._run_endpoint_under_policy(endpoint, definition, args, kwargs, cache_key)
        except Exception as exc:
            if future is not None:
                future.set_exception(exc)
                self._release_inflight(inflight_key, future)
            raise

        if future is not None:
            future.set_result(result)
            self._release_inflight(inflight_key, future)
        return result

    def _execute_endpoint_call(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return self._encode_endpoint_result(
            self._execute_endpoint_result(endpoint, definition, args, kwargs)
        )

    def _run_endpoint_under_policy(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        cache_key: str | None,
    ) -> _EndpointResult:
        if definition.concurrency == "serialized":
            lock = self._endpoint_lock(endpoint)
            with lock:
                cached = self._lookup_cached_result_value(definition, cache_key)
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
    ) -> _EndpointResult:
        if definition.managed:
            return self._materialize_managed_result(endpoint, definition, args, kwargs, cache_key)
        value = definition.func(*args, **kwargs)
        if self.verbose_debug:
            logger.debug("Server executed direct endpoint", extra={"endpoint": endpoint})
        if cache_key is not None:
            with self._coordination_lock:
                self._local_cache[cache_key] = value
                self._cache[cache_key] = endpoint
        return _EndpointResult(value=value)

    def _materialize_managed_result(
        self,
        endpoint: str,
        definition: EndpointDefinition,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        cache_key: str | None,
    ) -> _EndpointResult:
        if cache_key is not None:
            cached = self._managed_objects.get_cached(cache_key)
            if cached is not None:
                self._managed_objects.add_ref(cached.object_id)
                return _EndpointResult(value=cached.value, object_id=cached.object_id)

        result = definition.func(*args, **kwargs)
        if self.verbose_debug:
            logger.debug("Server created managed object", extra={"endpoint": endpoint, "cache_key": cache_key})
        entry = self._managed_objects.register(endpoint=endpoint, value=result, cache_key=cache_key)
        if cache_key is not None:
            with self._coordination_lock:
                self._cache[cache_key] = endpoint
        return _EndpointResult(value=entry.value, object_id=entry.object_id)

    def _lookup_cached_result_value(
        self,
        definition: EndpointDefinition,
        cache_key: str | None,
    ) -> _EndpointResult | None:
        if cache_key is None:
            return None
        if definition.managed:
            cached = self._managed_objects.get_cached(cache_key)
            if cached is None:
                return None
            self._managed_objects.add_ref(cached.object_id)
            return _EndpointResult(value=cached.value, object_id=cached.object_id)
        with self._coordination_lock:
            if cache_key not in self._local_cache:
                return None
            local_value = self._local_cache[cache_key]
        return _EndpointResult(value=local_value)

    def call_local_client(
        self,
        endpoint: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> _EndpointResult | None:
        definition = self.provider.get_endpoint(endpoint)
        resolved_kwargs = kwargs or {}
        if definition.execution == "task":
            task_info = self._submit_endpoint_task(endpoint, definition, args, resolved_kwargs)
            return self.wait_task_result_local(task_info.task_id)
        return self._execute_endpoint_result(endpoint, definition, args, resolved_kwargs)

    def get_task_result_local(self, task_id: str) -> _EndpointResult | None:
        result = self._task_manager_instance().result_local(task_id)
        if result is None:
            return None
        return result

    def wait_task_result_local(self, task_id: str, timeout: float | None = None) -> _EndpointResult | None:
        result = self._task_manager_instance().wait_result_local(task_id, timeout=timeout)
        if result is None:
            return None
        return result

    def wait_task_local(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        try:
            self._task_manager_instance().wait_result_local(task_id, timeout=timeout)
        except SharedTensorTaskError:
            info = self._task_manager_instance().get(task_id)
            if info.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
                return info.to_dict()
            raise
        return self._task_manager_instance().get(task_id).to_dict()

    def encode_local_result(self, result: _EndpointResult | None) -> dict[str, Any]:
        if result is None:
            return {"encoding": None, "payload_bytes": None, "object_id": None}
        return self._encode_endpoint_result(result)

    def invoke_local(
        self,
        endpoint: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        definition = self.provider.get_endpoint(endpoint)
        resolved_kwargs = kwargs or {}
        cache_key = self._cache_key(endpoint, definition, args, resolved_kwargs)
        if definition.managed:
            if cache_key is not None:
                cached = self._managed_objects.get_cached(cache_key)
                if cached is not None:
                    self.stats["cache_hits"] += 1
                    return cached.value
                self.stats["cache_misses"] += 1
            value = definition.func(*args, **resolved_kwargs)
            if cache_key is not None:
                existing = self._managed_objects.get_cached(cache_key)
                if existing is not None:
                    self.stats["cache_hits"] += 1
                    return existing.value
                self._managed_objects.register(endpoint=endpoint, value=value, cache_key=cache_key)
                with self._coordination_lock:
                    self._cache[cache_key] = endpoint
            return value
        if cache_key is not None:
            with self._coordination_lock:
                if cache_key in self._local_cache:
                    self.stats["cache_hits"] += 1
                    return self._local_cache[cache_key]
            self.stats["cache_misses"] += 1
        value = definition.func(*args, **resolved_kwargs)
        if cache_key is not None:
            with self._coordination_lock:
                self._local_cache[cache_key] = value
                self._cache[cache_key] = endpoint
        return value

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

    def _acquire_inflight(self, inflight_key: str) -> tuple[Future, bool]:
        with self._coordination_lock:
            inflight = self._inflight.get(inflight_key)
            if inflight is not None:
                return inflight.future, False
            future = Future()
            self._inflight[inflight_key] = _InFlightCall(future=future)
            return future, True

    def _release_inflight(self, inflight_key: str | None, future: Future) -> None:
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
        if self.verbose_debug:
            logger.debug("Server releasing managed object", extra={"object_id": object_id})
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
        info = self._managed_objects.info(object_id)
        if info is None:
            return {"object": None}
        return {"object": {**info, "server_id": self.server_id}}

    def _handle_invalidate_call_cache(self, params: dict[str, Any]) -> dict[str, Any]:
        endpoint, args, kwargs = self._decode_call_params(params)
        removed = self.invalidate_call_cache(endpoint, args=args, kwargs=kwargs)
        return {"invalidated": removed}

    def _handle_invalidate_endpoint_cache(self, params: dict[str, Any]) -> dict[str, Any]:
        endpoint = params.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise SharedTensorProtocolError("Missing required parameter 'endpoint'")
        return {"invalidated": self.invalidate_endpoint_cache(endpoint)}

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

    def _encode_result(
        self,
        value: Any,
        *,
        object_id: str | None = None,
        server_id: str | None = None,
    ) -> dict[str, Any]:
        if value is None:
            return {
                "encoding": None,
                "payload_bytes": None,
                "object_id": object_id,
                "server_id": server_id,
            }
        encoding, payload = serialize_payload(value)
        return {
            "encoding": encoding,
            "payload_bytes": payload,
            "object_id": object_id,
            "server_id": server_id,
        }

    def _encode_endpoint_result(self, result: _EndpointResult) -> dict[str, Any]:
        return self._encode_result(
            result.value,
            object_id=result.object_id,
            server_id=self.server_id if result.object_id is not None else None,
        )

    def _task_manager_instance(self) -> TaskManager:
        if self._task_manager is None:
            self._task_manager = TaskManager(
                max_workers=self.max_workers,
                result_ttl=self.result_ttl,
            )
        return self._task_manager

    def invalidate_call_cache(
        self,
        endpoint: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> bool:
        definition = self.provider.get_endpoint(endpoint)
        resolved_kwargs = kwargs or {}
        cache_key = self._cache_key(endpoint, definition, args, resolved_kwargs)
        if cache_key is None:
            return False
        invalidated_managed = False
        if definition.managed:
            invalidated_managed = self._managed_objects.invalidate_cache_key(cache_key)
        with self._coordination_lock:
            removed = self._local_cache.pop(cache_key, None)
            self._cache.pop(cache_key, None)
        invalidated = invalidated_managed or removed is not None
        if invalidated:
            self.stats["cache_invalidations"] += 1
        return invalidated

    def invalidate_endpoint_cache(self, endpoint: str) -> int:
        self.provider.get_endpoint(endpoint)
        removed = 0
        with self._coordination_lock:
            keys = [cache_key for cache_key, cache_endpoint in self._cache.items() if cache_endpoint == endpoint]
            for cache_key in keys:
                self._cache.pop(cache_key, None)
                if cache_key in self._local_cache:
                    self._local_cache.pop(cache_key, None)
                    removed += 1
        removed += self._managed_objects.invalidate_endpoint(endpoint)
        if removed:
            self.stats["cache_invalidations"] += removed
        return removed

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
            "version": _server_version(),
            "server_id": self.server_id,
            "socket_path": self.socket_path,
            "uptime": uptime,
            "running": self.running,
            "ready": self.running and self.listener is not None and self._accepting_requests,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "device_index": resolve_device_index(self.provider.device_index),
            "process_start_method": self._resolved_process_start_method,
            "stats": {
                **dict(self.stats),
                "cache_entries": len(self._local_cache),
                "inflight_calls": len(self._inflight),
                **self._managed_objects.stats(),
                "task_count": 0 if self._task_manager is None else len(self._task_manager.list()),
            },
            "capabilities": capability_snapshot(),
            "endpoints": list(self.provider.list_endpoints().keys()),
        }

    def _serialize_error(self, exc: Exception) -> dict[str, Any]:
        return {
            "code": self._error_code_for(exc),
            "message": str(exc),
            "type": type(exc).__name__,
            "data": None,
        }

    def start(self, blocking: bool = True) -> None:
        if self.verbose_debug:
            logger.info("Server starting", extra={"socket_path": self.socket_path, "blocking": blocking})
        if self.running or self.server_thread is not None:
            raise SharedTensorConfigurationError("Server is already running")
        self._accepting_requests = True
        if blocking:
            self._resolved_process_start_method = None
            self._serve_forever()
            return
        if self.process_start_method is not None:
            raise SharedTensorConfigurationError(
                "process_start_method is not supported for thread-backed non-blocking servers"
            )
        thread = threading.Thread(
            target=self._serve_forever_in_thread,
            name=f"shared-tensor-server:{self.socket_path}",
            daemon=True,
        )
        state = _ServerThreadState(thread=thread)
        self.server_thread = state
        self._resolved_process_start_method = "thread"
        thread.start()
        if not state.ready.wait(timeout=self.startup_timeout):
            self.stop(wait_for_tasks=False)
            raise TimeoutError(f"Timed out waiting for server socket {self.socket_path}")
        if state.error is not None:
            error = state.error
            self.stop(wait_for_tasks=False)
            raise SharedTensorConfigurationError(
                f"Failed to start background server thread for {self.socket_path}: {error}"
            ) from error

    def _serve_forever_in_thread(self) -> None:
        state = self.server_thread
        if state is None:
            return
        try:
            self._serve_forever(started_event=state.ready)
        except BaseException as exc:  # noqa: BLE001
            state.error = exc
            state.ready.set()
            raise
        finally:
            state.stopped.set()

    def _serve_forever(self, *, started_event: threading.Event | None = None) -> None:
        self._configure_cuda_runtime()
        unlink_socket_path(self.socket_path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(self.socket_path)
            listener.listen()
            if self.verbose_debug:
                logger.info("Server listening", extra={"socket_path": self.socket_path})
            self.listener = listener
            self.running = True
            self.started_at = time.time()
            register_local_server(self.socket_path, self)
            if started_event is not None:
                started_event.set()
            while self.running:
                try:
                    conn, _ = listener.accept()
                except OSError:
                    if self.running:
                        raise
                    break
                self._connection_executor.submit(self._handle_connection, conn)
        finally:
            if started_event is not None and not started_event.is_set():
                started_event.set()
            self._shutdown_local_resources(wait_for_tasks=True)

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                request = recv_message(conn)
                if not isinstance(request, dict):
                    raise SharedTensorProtocolError("Transport request must be a dict")
                response = self.process_request(request)
            except Exception as exc:
                response = {"ok": False, "error": self._serialize_error(exc)}
            try:
                send_message(conn, response)
                if self.verbose_debug:
                    logger.debug("Server sent response", extra={"ok": response.get("ok")})
            except OSError:
                logger.debug("Client disconnected before response could be sent", exc_info=True)

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

    def stop(self, *, wait_for_tasks: bool = True) -> None:
        if self.verbose_debug:
            logger.info("Server stopping", extra={"socket_path": self.socket_path})
        self._accepting_requests = False
        self.running = False
        if self.listener is not None:
            self.listener.close()
        state = self.server_thread
        if state is not None and state.thread.is_alive() and threading.current_thread() is not state.thread:
            state.stopped.wait(timeout=5)
            state.thread.join(timeout=5)
        self.server_thread = None
        self.server_process = None
        if self.listener is None:
            self._shutdown_local_resources(wait_for_tasks=wait_for_tasks)

    def _shutdown_local_resources(self, *, wait_for_tasks: bool) -> None:
        self._accepting_requests = False
        self.running = False
        if self.listener is not None:
            self.listener.close()
            self.listener = None
        if self._task_manager is not None:
            self._task_manager.shutdown(wait=wait_for_tasks, cancel_futures=not wait_for_tasks)
            self._task_manager = None
        self._managed_objects.clear()
        with self._coordination_lock:
            self._cache.clear()
            self._local_cache.clear()
            self._inflight.clear()
            self._endpoint_locks.clear()
        unregister_local_server(self.socket_path, self)
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
        if isinstance(exc, SharedTensorStaleHandleError):
            return 8
        return 7
