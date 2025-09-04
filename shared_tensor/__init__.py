"""
Shared Tensor Library

A library for sharing GPU memory objects across processes using IPC mechanisms.
Enables model and inference engine separation architecture using JSON-RPC 2.0 protocol.
"""

from shared_tensor.provider import SharedTensorProvider
from shared_tensor.client import SharedTensorClient
from shared_tensor.server import SharedTensorServer
from shared_tensor.async_provider import AsyncSharedTensorProvider
from shared_tensor.async_client import AsyncSharedTensorClient
from shared_tensor.async_task import TaskStatus, TaskInfo

__version__ = "0.1.0"
__author__ = "Athena Team"

# Export main functionality
__all__ = [
    "SharedTensorProvider", 
    "SharedTensorClient",
    "SharedTensorServer",
    "AsyncSharedTensorProvider",
    "AsyncSharedTensorClient", 
    "TaskStatus",
    "TaskInfo",
]