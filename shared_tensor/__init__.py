"""shared_tensor: same-host same-GPU PyTorch CUDA IPC over local UDS RPC."""

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

__version__ = "0.2.12"
