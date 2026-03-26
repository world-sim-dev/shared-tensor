from __future__ import annotations

import time

import pytest

from shared_tensor.async_task import TaskManager, TaskStatus
from shared_tensor.errors import SharedTensorTaskError


def test_task_info_round_trip() -> None:
    from shared_tensor.async_task import TaskInfo

    info = TaskInfo(task_id="t1", endpoint="load", status=TaskStatus.RUNNING, created_at=1.0)

    assert TaskInfo.from_dict(info.to_dict()) == info


def test_task_manager_wait_result_payload_for_none_result() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=0.0)
    try:
        info = manager.submit("noop", lambda: None, (), {})

        payload = manager.wait_result_payload(info.task_id, timeout=1)

        assert payload == {"encoding": None, "payload_bytes": None, "object_id": None}
        assert manager.get(info.task_id).status == TaskStatus.COMPLETED
    finally:
        manager.shutdown(wait=True)


def test_task_manager_wait_result_local_returns_none_for_none_result() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=0.0)
    try:
        info = manager.submit("noop", lambda: None, (), {})

        assert manager.wait_result_local(info.task_id, timeout=1) is None
        assert manager.result_local(info.task_id) is None
    finally:
        manager.shutdown(wait=True)


def test_task_manager_retains_local_result_for_same_process_consumers() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=0.0)
    token = object()
    try:
        info = manager.submit("token", lambda: token, (), {}, result_encoder=lambda value: {
            "encoding": None,
            "payload_bytes": None,
            "object_id": None,
        })

        assert manager.wait_result_local(info.task_id, timeout=1) is token
        assert manager.result_local(info.task_id) is token
    finally:
        manager.shutdown(wait=True)


def test_task_manager_timeout_surfaces_clear_error() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=0.0)
    try:
        info = manager.submit("slow", lambda: time.sleep(0.2), (), {})

        with pytest.raises(SharedTensorTaskError, match="did not complete within 0.01 seconds"):
            manager.wait_result_payload(info.task_id, timeout=0.01)
    finally:
        manager.shutdown(wait=True)


def test_task_manager_cleans_up_expired_results() -> None:
    manager = TaskManager(max_workers=1, result_ttl=0.01, cleanup_interval=0.0)
    try:
        info = manager.submit("noop", lambda: None, (), {})
        manager.wait_result_payload(info.task_id, timeout=1)
        time.sleep(0.03)

        with pytest.raises(SharedTensorTaskError, match="was not found"):
            manager.get(info.task_id)
    finally:
        manager.shutdown(wait=True)


def test_task_manager_list_filters_by_status() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=0.0)
    try:
        done = manager.submit("done", lambda: None, (), {})
        manager.wait_result_payload(done.task_id, timeout=1)

        failed = manager.submit("failed", lambda: (_ for _ in ()).throw(ValueError("boom")), (), {})
        with pytest.raises(SharedTensorTaskError, match="boom"):
            manager.wait_result_payload(failed.task_id, timeout=1)

        completed = manager.list(status=TaskStatus.COMPLETED)
        failed_only = manager.list(status=TaskStatus.FAILED)

        assert set(completed) == {done.task_id}
        assert set(failed_only) == {failed.task_id}
    finally:
        manager.shutdown(wait=True)


def test_task_manager_result_decodes_completed_payload() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=0.0)
    try:
        info = manager.submit("value", lambda: None, (), {})
        manager.wait_result_payload(info.task_id, timeout=1)
        assert manager.result(info.task_id) is None
    finally:
        manager.shutdown(wait=True)


def test_task_manager_result_payload_rejects_incomplete_task() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=999.0)
    try:
        info = manager.submit("slow", lambda: time.sleep(0.2), (), {})
        with pytest.raises(SharedTensorTaskError, match="is not complete"):
            manager.result_payload(info.task_id)
    finally:
        manager.shutdown(wait=True)


def test_task_manager_cancel_rejects_missing_task() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=0.0)
    try:
        with pytest.raises(SharedTensorTaskError, match="was not found"):
            manager.cancel("missing")
    finally:
        manager.shutdown(wait=True)


def test_task_manager_cancel_returns_false_for_completed_task() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=0.0)
    try:
        info = manager.submit("noop", lambda: None, (), {})
        manager.wait_result_payload(info.task_id, timeout=1)
        assert manager.cancel(info.task_id) is False
    finally:
        manager.shutdown(wait=True)


def test_task_manager_respects_capacity_limit() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=999.0, max_tasks=1)
    try:
        manager.submit("slow", lambda: time.sleep(0.2), (), {})
        with pytest.raises(SharedTensorTaskError, match="Task capacity exceeded"):
            manager.submit("second", lambda: None, (), {})
    finally:
        manager.shutdown(wait=True)


def test_task_manager_drops_oldest_finished_task_when_full() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=999.0, max_tasks=1)
    try:
        first = manager.submit("first", lambda: None, (), {})
        manager.wait_result_payload(first.task_id, timeout=1)

        second = manager.submit("second", lambda: None, (), {})
        manager.wait_result_payload(second.task_id, timeout=1)

        with pytest.raises(SharedTensorTaskError, match="was not found"):
            manager.get(first.task_id)
        assert manager.get(second.task_id).endpoint == "second"
    finally:
        manager.shutdown(wait=True)
