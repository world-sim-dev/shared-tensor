from __future__ import annotations

import threading

from shared_tensor.managed_object import ManagedObjectRegistry, SharedObjectHandle


class _FakeReleaser:
    def __init__(self, info: dict | None = None) -> None:
        self.calls = 0
        self.info = info
        self.info_calls = 0

    def release(self) -> bool:
        self.calls += 1
        return True

    def get_object_info(self) -> dict | None:
        self.info_calls += 1
        return self.info


def test_managed_registry_tracks_cache_and_refcounts() -> None:
    registry = ManagedObjectRegistry()

    entry = registry.register(endpoint="load_model", value=object(), cache_key="model:4")

    assert registry.get(entry.object_id) is entry
    assert registry.get_cached("model:4") is entry
    assert registry.add_ref(entry.object_id) is entry
    info = registry.info(entry.object_id)
    assert info is not None
    assert info["object_id"] == entry.object_id
    assert info["endpoint"] == "load_model"
    assert info["cache_key"] == "model:4"
    assert info["refcount"] == 2
    assert info["created_at"] <= info["last_accessed_at"]

    first = registry.release(entry.object_id)
    assert first.released is True
    assert first.destroyed is False
    assert first.refcount == 1
    assert registry.get_cached("model:4") is entry

    second = registry.release(entry.object_id)
    assert second.released is True
    assert second.destroyed is True
    assert second.refcount == 0
    assert registry.get(entry.object_id) is None
    assert registry.get_cached("model:4") is None


def test_managed_registry_release_missing_object_is_noop() -> None:
    registry = ManagedObjectRegistry()

    result = registry.release("missing")

    assert result.released is False
    assert result.destroyed is False
    assert result.refcount == 0
    assert result.cache_key is None


def test_managed_registry_can_invalidate_cache_entries_without_destroying_objects() -> None:
    registry = ManagedObjectRegistry()
    entry = registry.register(endpoint="load_model", value=object(), cache_key="model:4")

    assert registry.invalidate_cache_key("model:4") is True
    assert registry.invalidate_cache_key("model:4") is False
    assert registry.get(entry.object_id) is entry
    assert registry.get_cached("model:4") is None
    assert registry.release(entry.object_id).destroyed is True


def test_managed_registry_can_invalidate_endpoint_cache_index() -> None:
    registry = ManagedObjectRegistry()
    first = registry.register(endpoint="load_model", value=object(), cache_key="model:4")
    second = registry.register(endpoint="load_model", value=object(), cache_key="model:8")
    third = registry.register(endpoint="other", value=object(), cache_key="other:1")

    assert registry.invalidate_endpoint("load_model") == 2
    assert registry.get(first.object_id) is first
    assert registry.get(second.object_id) is second
    assert registry.get(third.object_id) is third
    assert registry.get_cached("model:4") is None
    assert registry.get_cached("model:8") is None
    assert registry.get_cached("other:1") is third
    assert registry.stats() == {"objects": 3, "cached_objects": 1}


def test_managed_registry_is_thread_safe_for_refcount_updates() -> None:
    registry = ManagedObjectRegistry()
    entry = registry.register(endpoint="load_model", value=object(), cache_key="model:4")

    def worker() -> None:
        for _ in range(200):
            registry.add_ref(entry.object_id)
            registry.release(entry.object_id)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert registry.info(entry.object_id)["refcount"] == 1
    assert registry.release(entry.object_id).destroyed is True


def test_shared_object_handle_releases_once_and_context_manager_marks_released() -> None:
    releaser = _FakeReleaser()
    handle = SharedObjectHandle(object_id="obj-1", value="payload", _releaser=releaser)

    assert handle.release() is True
    assert handle.released is True
    assert handle.release() is False
    assert releaser.calls == 1

    other_releaser = _FakeReleaser()
    with SharedObjectHandle(object_id="obj-2", value="payload", _releaser=other_releaser) as ctx:
        assert ctx.value == "payload"
        assert ctx.released is False

    assert other_releaser.calls == 1


def test_shared_object_handle_caches_metadata_and_detects_staleness() -> None:
    releaser = _FakeReleaser(info={"object_id": "obj-1", "server_id": "srv-1"})
    handle = SharedObjectHandle(
        object_id="obj-1",
        value="payload",
        _releaser=releaser,
        server_id="srv-1",
    )

    assert handle.get_object_info() == {"object_id": "obj-1", "server_id": "srv-1"}
    assert handle.get_object_info() == {"object_id": "obj-1", "server_id": "srv-1"}
    assert releaser.info_calls == 1
    assert handle.get_object_info(refresh=True) == {"object_id": "obj-1", "server_id": "srv-1"}
    assert releaser.info_calls == 2
    assert handle.is_stale() is False

    stale_releaser = _FakeReleaser(info=None)
    stale_handle = SharedObjectHandle(object_id="obj-2", value="payload", _releaser=stale_releaser)
    assert stale_handle.is_stale() is True
