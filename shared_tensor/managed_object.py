"""Managed remote object handles and registry state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from threading import RLock
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class ManagedObjectEntry:
    object_id: str
    value: Any
    endpoint: str
    cache_key: str | None
    refcount: int = 1


@dataclass(slots=True)
class ManagedReleaseResult:
    released: bool
    destroyed: bool
    refcount: int
    cache_key: str | None


class ManagedObjectRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ManagedObjectEntry] = {}
        self._cache_index: dict[str, str] = {}
        self._lock = RLock()

    def get_cached(self, cache_key: str) -> ManagedObjectEntry | None:
        with self._lock:
            object_id = self._cache_index.get(cache_key)
            if object_id is None:
                return None
            entry = self._entries.get(object_id)
            if entry is None:
                self._cache_index.pop(cache_key, None)
                return None
            return entry

    def register(self, *, endpoint: str, value: Any, cache_key: str | None) -> ManagedObjectEntry:
        with self._lock:
            entry = ManagedObjectEntry(
                object_id=uuid.uuid4().hex,
                value=value,
                endpoint=endpoint,
                cache_key=cache_key,
            )
            self._entries[entry.object_id] = entry
            if cache_key is not None:
                self._cache_index[cache_key] = entry.object_id
            return entry

    def get(self, object_id: str) -> ManagedObjectEntry | None:
        with self._lock:
            return self._entries.get(object_id)

    def add_ref(self, object_id: str) -> ManagedObjectEntry | None:
        with self._lock:
            entry = self._entries.get(object_id)
            if entry is None:
                return None
            entry.refcount += 1
            return entry

    def release(self, object_id: str) -> ManagedReleaseResult:
        with self._lock:
            entry = self._entries.get(object_id)
            if entry is None:
                return ManagedReleaseResult(released=False, destroyed=False, refcount=0, cache_key=None)

            entry.refcount -= 1
            destroyed = entry.refcount <= 0
            cache_key = entry.cache_key
            refcount = max(entry.refcount, 0)
            if destroyed:
                self._entries.pop(object_id, None)
                if cache_key is not None and self._cache_index.get(cache_key) == object_id:
                    self._cache_index.pop(cache_key, None)
            return ManagedReleaseResult(
                released=True,
                destroyed=destroyed,
                refcount=refcount,
                cache_key=cache_key,
            )

    def info(self, object_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(object_id)
            if entry is None:
                return None
            return {
                "object_id": entry.object_id,
                "endpoint": entry.endpoint,
                "cache_key": entry.cache_key,
                "refcount": entry.refcount,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._cache_index.clear()


class ReleaseHandle:
    def release(self) -> bool:  # pragma: no cover - protocol surface only
        raise NotImplementedError


@dataclass(slots=True)
class SharedObjectHandle(Generic[T]):
    object_id: str
    value: T
    _releaser: ReleaseHandle
    released: bool = False

    def release(self) -> bool:
        if self.released:
            return False
        released = self._releaser.release()
        if released:
            self.released = True
        return released

    def __enter__(self) -> SharedObjectHandle[T]:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.release()
