from __future__ import annotations

import threading

from shared_tensor.managed_object import ManagedObjectRegistry, SharedObjectHandle


class _FakeReleaser:
    def __init__(self) -> None:
        self.calls = 0

    def release(self) -> bool:
        self.calls += 1
        return True


def test_managed_registry_tracks_cache_and_refcounts() -> None:
    registry = ManagedObjectRegistry()

    entry = registry.register(endpoint="load_model", value=object(), cache_key="model:4")

    assert registry.get(entry.object_id) is entry
    assert registry.get_cached("model:4") is entry
    assert registry.add_ref(entry.object_id) is entry
    assert registry.info(entry.object_id) == {
        "object_id": entry.object_id,
        "endpoint": "load_model",
        "cache_key": "model:4",
        "refcount": 2,
    }

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
