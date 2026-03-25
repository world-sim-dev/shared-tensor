from __future__ import annotations

import time

import pytest

from shared_tensor.async_task import TaskManager, TaskStatus
from shared_tensor.errors import SharedTensorTaskError


def test_task_manager_wait_result_payload_for_none_result() -> None:
    manager = TaskManager(max_workers=1, cleanup_interval=0.0)
    try:
        info = manager.submit("noop", lambda: None, (), {})

        payload = manager.wait_result_payload(info.task_id, timeout=1)

        assert payload == {"encoding": None, "payload_hex": None, "object_id": None}
        assert manager.get(info.task_id).status == TaskStatus.COMPLETED
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
