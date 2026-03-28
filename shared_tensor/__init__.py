"""shared_tensor: same-host same-GPU PyTorch CUDA IPC over local UDS RPC."""

import logging
import os


class _SafeStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        stream = self.stream
        if stream is None or getattr(stream, "closed", False):
            return
        try:
            super().emit(record)
        except ValueError:
            return


def _configure_default_logging() -> None:
    logger = logging.getLogger("shared_tensor")
    if logger.handlers:
        return
    level_name = os.getenv("SHARED_TENSOR_LOG_LEVEL", "INFO").strip().upper() or "INFO"
    level = getattr(logging, level_name, logging.INFO)
    handler = _SafeStreamHandler()
    handler.setFormatter(logging.Formatter("[shared_tensor] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

_configure_default_logging()

from shared_tensor.async_client import AsyncSharedTensorClient
from shared_tensor.async_task import TaskInfo, TaskStatus
from shared_tensor.client import SharedTensorClient
from shared_tensor.errors import SharedTensorStaleHandleError
from shared_tensor.managed_object import SharedObjectHandle
from shared_tensor.provider import SharedTensorProvider
from shared_tensor.server import SharedTensorServer

__all__ = [
    "AsyncSharedTensorClient",
    "SharedTensorClient",
    "SharedObjectHandle",
    "SharedTensorStaleHandleError",
    "SharedTensorProvider",
    "SharedTensorServer",
    "TaskInfo",
    "TaskStatus",
]

__version__ = "0.2.16"
