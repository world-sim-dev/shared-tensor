"""Async task primitives used by the RPC server."""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

from shared_tensor.errors import SharedTensorTaskError
from shared_tensor.utils import deserialize_payload, serialize_payload


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class TaskInfo:
    task_id: str
    endpoint: str
    status: TaskStatus
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "endpoint": self.endpoint,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskInfo:
        payload = dict(data)
        payload["status"] = TaskStatus(payload["status"])
        payload.setdefault("metadata", {})
        return cls(**payload)


@dataclass(slots=True)
class _TaskEntry:
    info: TaskInfo
    future: Future[Any]
    result_encoding: str | None = None
    result_payload: bytes | None = None
    local_result: Any = None


class TaskManager:
    """In-process task executor with bounded retention and explicit task states."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        result_ttl: float = 3600.0,
        cleanup_interval: float = 60.0,
        max_tasks: int = 1000,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._result_ttl = result_ttl
        self._cleanup_interval = cleanup_interval
        self._max_tasks = max_tasks
        self._last_cleanup = 0.0
        self._lock = RLock()
        self._tasks: dict[str, _TaskEntry] = {}
        self._accepting_submissions = True

    def submit(
        self,
        endpoint: str,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result_encoder: Callable[[Any], dict[str, Any]] | None = None,
    ) -> TaskInfo:
        self._maybe_cleanup()
        with self._lock:
            if not self._accepting_submissions:
                raise SharedTensorTaskError("Task manager is shutting down and is not accepting new tasks")
            self._drop_oldest_finished_tasks_if_needed()
            if len(self._tasks) >= self._max_tasks:
                raise SharedTensorTaskError("Task capacity exceeded")

            info = TaskInfo(
                task_id=str(uuid.uuid4()),
                endpoint=endpoint,
                status=TaskStatus.PENDING,
                created_at=time.time(),
            )
            future = self._executor.submit(
                self._run_task,
                info.task_id,
                func,
                args,
                kwargs,
                result_encoder,
            )
            self._tasks[info.task_id] = _TaskEntry(info=info, future=future)
            return copy.deepcopy(info)

    def _run_task(
        self,
        task_id: str,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result_encoder: Callable[[Any], dict[str, Any]] | None,
    ) -> None:
        self._transition(task_id, status=TaskStatus.RUNNING, started_at=time.time())
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - hit in integration tests
            self._transition(
                task_id,
                status=TaskStatus.FAILED,
                completed_at=time.time(),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return

        self._store_local_result(task_id, result)

        if result is None:
            self._store_payload(task_id, encoding=None, payload=None, object_id=None)
            self._transition(
                task_id,
                status=TaskStatus.COMPLETED,
                completed_at=time.time(),
            )
            return

        try:
            payload = (
                result_encoder(result)
                if result_encoder is not None
                else self._default_result_encoder(result)
            )
        except Exception as exc:  # pragma: no cover - hit in integration tests
            self._transition(
                task_id,
                status=TaskStatus.FAILED,
                completed_at=time.time(),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return

        self._store_payload(
            task_id,
            encoding=payload["encoding"],
            payload=payload["payload_bytes"],
            object_id=payload.get("object_id"),
        )
        self._transition(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=time.time(),
        )

    @staticmethod
    def _default_result_encoder(value: Any) -> dict[str, Any]:
        encoding, payload = serialize_payload(value)
        return {
            "encoding": encoding,
            "payload_bytes": payload,
            "object_id": None,
        }

    def _transition(self, task_id: str, **updates: Any) -> None:
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            for key, value in updates.items():
                setattr(entry.info, key, value)

    def _store_local_result(self, task_id: str, value: Any) -> None:
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            entry.local_result = value

    def _store_payload(
        self,
        task_id: str,
        *,
        encoding: str | None,
        payload: bytes | None,
        object_id: str | None,
    ) -> None:
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            entry.result_encoding = encoding
            entry.result_payload = payload
            metadata = dict(entry.info.metadata)
            metadata["object_id"] = object_id
            entry.info.metadata = metadata

    def get(self, task_id: str) -> TaskInfo:
        self._maybe_cleanup()
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise SharedTensorTaskError(f"Task '{task_id}' was not found")
            return copy.deepcopy(entry.info)

    def result(self, task_id: str) -> Any:
        payload = self.result_payload(task_id)
        encoding = payload["encoding"]
        payload_bytes = payload["payload_bytes"]
        if encoding is None or payload_bytes is None:
            return None
        return deserialize_payload(encoding, payload_bytes)

    def result_local(self, task_id: str) -> Any:
        self._maybe_cleanup()
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise SharedTensorTaskError(f"Task '{task_id}' was not found")
            info = copy.deepcopy(entry.info)
            value = entry.local_result
        if info.status == TaskStatus.CANCELLED:
            raise SharedTensorTaskError(f"Task '{task_id}' was cancelled")
        if info.status == TaskStatus.FAILED:
            raise SharedTensorTaskError(info.error_message or f"Task '{task_id}' failed")
        if info.status != TaskStatus.COMPLETED:
            raise SharedTensorTaskError(
                f"Task '{task_id}' is not complete; current status is '{info.status.value}'"
            )
        return value

    def wait_result_payload(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> dict[str, str | bytes | None]:
        self._maybe_cleanup()
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise SharedTensorTaskError(f"Task '{task_id}' was not found")
            future = entry.future
        try:
            future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise SharedTensorTaskError(
                f"Task '{task_id}' did not complete within {timeout} seconds"
            ) from exc
        return self.result_payload(task_id)

    def result_payload(self, task_id: str) -> dict[str, str | bytes | None]:
        self._maybe_cleanup()
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise SharedTensorTaskError(f"Task '{task_id}' was not found")
            info = copy.deepcopy(entry.info)
            encoding = entry.result_encoding
            payload = entry.result_payload
        if info.status == TaskStatus.CANCELLED:
            raise SharedTensorTaskError(f"Task '{task_id}' was cancelled")
        if info.status == TaskStatus.FAILED:
            raise SharedTensorTaskError(info.error_message or f"Task '{task_id}' failed")
        if info.status != TaskStatus.COMPLETED:
            raise SharedTensorTaskError(
                f"Task '{task_id}' is not complete; current status is '{info.status.value}'"
            )
        return {
            "encoding": encoding,
            "payload_bytes": payload,
            "object_id": info.metadata.get("object_id"),
        }

    def wait_result_local(self, task_id: str, timeout: float | None = None) -> Any:
        self._maybe_cleanup()
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise SharedTensorTaskError(f"Task '{task_id}' was not found")
            future = entry.future
        try:
            future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise SharedTensorTaskError(
                f"Task '{task_id}' did not complete within {timeout} seconds"
            ) from exc
        return self.result_local(task_id)

    def cancel(self, task_id: str) -> bool:
        self._maybe_cleanup()
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                raise SharedTensorTaskError(f"Task '{task_id}' was not found")
            if entry.info.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                return False
            cancelled = entry.future.cancel()
            if cancelled:
                entry.info.status = TaskStatus.CANCELLED
                entry.info.completed_at = time.time()
            return cancelled

    def list(self, status: TaskStatus | None = None) -> dict[str, TaskInfo]:
        self._maybe_cleanup()
        with self._lock:
            items = {
                task_id: copy.deepcopy(entry.info)
                for task_id, entry in self._tasks.items()
                if status is None or entry.info.status == status
            }
        return items

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = True) -> None:
        with self._lock:
            self._accepting_submissions = False
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        with self._lock:
            expired = [
                task_id
                for task_id, entry in self._tasks.items()
                if entry.info.completed_at is not None
                and now - entry.info.completed_at >= self._result_ttl
            ]
            for task_id in expired:
                self._tasks.pop(task_id, None)

    def _drop_oldest_finished_tasks_if_needed(self) -> None:
        if len(self._tasks) < self._max_tasks:
            return
        finished = [
            (task_id, entry.info.completed_at or entry.info.created_at)
            for task_id, entry in self._tasks.items()
            if entry.info.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
        ]
        finished.sort(key=lambda item: item[1])
        while len(self._tasks) >= self._max_tasks and finished:
            task_id, _ = finished.pop(0)
            self._tasks.pop(task_id, None)
