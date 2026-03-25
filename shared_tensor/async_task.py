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
    result_encoding: str | None = None
    result_hex: str | None = None
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
            "result_encoding": self.result_encoding,
            "result_hex": self.result_hex,
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

        if result is None:
            self._transition(
                task_id,
                status=TaskStatus.COMPLETED,
                completed_at=time.time(),
                result_encoding=None,
                result_hex=None,
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

        self._transition(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=time.time(),
            result_encoding=payload["encoding"],
            result_hex=payload["payload_hex"],
            metadata={"object_id": payload.get("object_id")},
        )

    @staticmethod
    def _default_result_encoder(value: Any) -> dict[str, Any]:
        encoding, payload = serialize_payload(value)
        return {
            "encoding": encoding,
            "payload_hex": payload.hex(),
            "object_id": None,
        }

    def _transition(self, task_id: str, **updates: Any) -> None:
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            for key, value in updates.items():
                setattr(entry.info, key, value)

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
        payload_hex = payload["payload_hex"]
        if encoding is None or payload_hex is None:
            return None
        return deserialize_payload(encoding, payload_hex)

    def wait_result_payload(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> dict[str, str | None]:
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

    def result_payload(self, task_id: str) -> dict[str, str | None]:
        info = self.get(task_id)
        if info.status == TaskStatus.CANCELLED:
            raise SharedTensorTaskError(f"Task '{task_id}' was cancelled")
        if info.status == TaskStatus.FAILED:
            raise SharedTensorTaskError(info.error_message or f"Task '{task_id}' failed")
        if info.status != TaskStatus.COMPLETED:
            raise SharedTensorTaskError(
                f"Task '{task_id}' is not complete; current status is '{info.status.value}'"
            )
        return {
            "encoding": info.result_encoding,
            "payload_hex": info.result_hex,
            "object_id": info.metadata.get("object_id"),
        }

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

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

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
