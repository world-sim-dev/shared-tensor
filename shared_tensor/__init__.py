"""shared_tensor: local endpoint-oriented RPC for Python and PyTorch."""

from shared_tensor.async_client import AsyncSharedTensorClient
from shared_tensor.async_provider import AsyncSharedTensorProvider
from shared_tensor.async_task import TaskInfo, TaskStatus
from shared_tensor.client import SharedTensorClient
from shared_tensor.provider import SharedTensorProvider
from shared_tensor.server import SharedTensorServer

__all__ = [
    "AsyncSharedTensorClient",
    "AsyncSharedTensorProvider",
    "SharedTensorClient",
    "SharedTensorProvider",
    "SharedTensorServer",
    "TaskInfo",
    "TaskStatus",
]

__version__ = "0.2.0"
