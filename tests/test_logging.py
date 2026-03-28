from __future__ import annotations

import importlib
import io
import logging

import pytest


def test_shared_tensor_default_logger_honors_env_level(monkeypatch: pytest.MonkeyPatch) -> None:
    import shared_tensor

    logger = logging.getLogger("shared_tensor")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    monkeypatch.setenv("SHARED_TENSOR_LOG_LEVEL", "INFO")
    shared_tensor = importlib.reload(shared_tensor)

    logger = logging.getLogger("shared_tensor")
    assert logger.level == logging.INFO
    assert logger.handlers


def test_safe_stream_handler_ignores_closed_stream() -> None:
    import shared_tensor

    stream = io.StringIO()
    handler = shared_tensor._SafeStreamHandler(stream)
    stream.close()

    handler.emit(
        logging.LogRecord(
            name="shared_tensor.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="ignored",
            args=(),
            exc_info=None,
        )
    )
