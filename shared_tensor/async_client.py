"""
Async Shared Tensor Client

Supports long-running task execution without HTTP timeout limitations.
"""

import time
import logging
from typing import Any, Dict, Optional, Callable

import torch

from shared_tensor.errors import SharedTensorServerError
from shared_tensor.client import SharedTensorClient
from shared_tensor.async_task import TaskStatus, TaskInfo
from shared_tensor.utils import serialize_result


__all__ = ["AsyncSharedTensorClient", "execute_remote_function_async"]


logger = logging.getLogger(__name__)


class AsyncSharedTensorClient:
    """
    Async client for shared tensor operations
    
    Supports submitting long-running tasks and polling for results
    without being limited by HTTP timeouts.
    """
    
    def __init__(self, server_port: int = 2537, verbose_debug: bool = False, poll_interval: float = 1.0):
        """
        Initialize async client
        
        Args:
            server_port: Port of the shared tensor server
            verbose_debug: Whether to enable verbose debug logging
            poll_interval: Interval in seconds for polling task status
        """
        self.server_url = f"http://localhost:{server_port}"
        self.verbose_debug = verbose_debug
        self.poll_interval = poll_interval
        self._client = SharedTensorClient(server_port, verbose_debug=verbose_debug)
    
    def submit_task(self, function_path: str, args: tuple = (), kwargs: Dict[str, Any] = None, options: Dict[str, Any] = None) -> str:
        """
        Submit a task for async execution
        
        Args:
            function_path: Function path in format "module.submodule:function_name"
            args: Positional arguments
            kwargs: Keyword arguments
            options: Options for the task
            
        Returns:
            Task ID for tracking the execution
        """
        if kwargs is None:
            kwargs = {}
        
        args_hex = serialize_result(args).hex() if args else ""
        kwargs_hex = serialize_result(kwargs).hex() if kwargs else ""
        
        if self.verbose_debug:
            logger.debug(f"Submitting task with function path {function_path}, args {args}, kwargs {kwargs}, and options {options}")
        else:
            logger.debug(f"Submitting task with function path {function_path}")
        
        response = self._client._send_request(
            self._client._create_request("submit_task", {
                "function_path": function_path,
                "args": args_hex,
                "kwargs": kwargs_hex,
                "options": options,
                "encoding": "pickle_hex",
            })
        )
        
        if response.error:
            raise SharedTensorServerError(f"Failed to submit task: {response.error}")
        
        task_id = response.result.get("task_id")
        if not task_id:
            raise SharedTensorServerError("Server did not return task ID")
        
        logger.debug(f"Task submitted: {task_id}")
        return task_id
    
    def get_task_status(self, task_id: str) -> TaskInfo:
        """
        Get current status of a task
        
        Args:
            task_id: Task ID returned by submit_task
            
        Returns:
            TaskInfo object with current status
        """
        logger.debug(f"Getting task status for task {task_id}")
        response = self._client._send_request(
            self._client._create_request("get_task_status", {"task_id": task_id})
        )
        
        if response.error:
            logger.debug(f"Failed to get task status: {response.error}")
            raise SharedTensorServerError(f"Failed to get task status: {response.error}")
        
        task_data = response.result
        if not task_data:
            raise SharedTensorServerError(f"Task {task_id} not found")
        
        return TaskInfo.from_dict(task_data)
    
    def get_task_result(self, task_id: str) -> Any:
        """
        Get result of a completed task
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result (deserialized)
            
        Raises:
            RuntimeError: If task failed or not completed
        """
        task_info = self.get_task_status(task_id)
        
        if task_info.status == TaskStatus.FAILED:
            raise SharedTensorServerError(f"Task failed: {task_info.error_message}")
        
        if task_info.status != TaskStatus.COMPLETED:
            raise SharedTensorServerError(f"Task not completed, current status: {task_info.status.value}")
        
        if task_info.result_hex:
            result_bytes = bytes.fromhex(task_info.result_hex)
            return torch.multiprocessing.reducer.ForkingPickler.loads(result_bytes)
        return None
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None, 
                     callback: Optional[Callable[[TaskInfo], None]] = None) -> Any:
        """
        Wait for a task to complete and return its result
        
        Args:
            task_id: Task ID
            timeout: Maximum time to wait (None for no timeout)
            callback: Optional callback function called on each status update
            
        Returns:
            Task result
        """
        start_time = time.time()
        
        while True:
            task_info = self.get_task_status(task_id)
            
            # Call callback if provided
            if callback:
                try:
                    callback(task_info)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")
            
            # Check if completed
            if task_info.status == TaskStatus.COMPLETED:
                return self.get_task_result(task_id)
            
            # Check if failed
            if task_info.status == TaskStatus.FAILED:
                raise SharedTensorServerError(f"Task {task_id} failed: {task_info.error_message}")
            
            # Check if cancelled
            if task_info.status == TaskStatus.CANCELLED:
                raise SharedTensorServerError("Task was cancelled")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise SharedTensorServerError(f"Task {task_id} did not complete within {timeout} seconds")
            
            # Sleep before next poll
            time.sleep(self.poll_interval)
    
    def execute_function_async(self, function_path: str, args: tuple = (), 
                              kwargs: Dict[str, Any] = None, options: Dict[str, Any] = None, wait: bool = True,
                              timeout: Optional[float] = None,
                              callback: Optional[Callable[[TaskInfo], None]] = None) -> Any:
        """
        Execute a function asynchronously
        
        Args:
            function_path: Function path
            args: Positional arguments
            kwargs: Keyword arguments
            options: Options for the task
            wait: Whether to wait for completion
            timeout: Maximum time to wait if wait=True
            callback: Status update callback
            
        Returns:
            If wait=True: Function result
            If wait=False: Task ID
        """
        task_id = self.submit_task(function_path, args, kwargs, options)
        
        if wait:
            return self.wait_for_task(task_id, timeout, callback)
        else:
            return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task
        
        Args:
            task_id: Task ID
            
        Returns:
            True if successfully cancelled
        """
        response = self._client._send_request(
            self._client._create_request("cancel_task", {"task_id": task_id})
        )
        
        if response.error:
            logger.error(f"Failed to cancel task: {response.error}")
            return False
        
        return response.result.get("cancelled", False)
    
    def list_tasks(self, status: Optional[str] = None) -> Dict[str, TaskInfo]:
        """
        List tasks on the server
        
        Args:
            status: Optional status filter
            
        Returns:
            Dictionary of task ID -> TaskInfo
        """
        params = {}
        if status:
            params["status"] = status
        
        response = self._client._send_request(
            self._client._create_request("list_tasks", params)
        )
        
        if response.error:
            raise SharedTensorServerError(f"Failed to list tasks: {response.error}")
        
        tasks = {}
        for task_id, task_data in response.result.items():
            tasks[task_id] = TaskInfo.from_dict(task_data)
        
        return tasks
    
    def close(self):
        """Close the client"""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def execute_remote_function_async(
    function_path: str,
    args: tuple = (),
    kwargs: Dict[str, Any] = None,
    options: Dict[str, Any] = None,
    server_port: int = 2537,
    verbose_debug: bool = False,
    poll_interval: float = 1.0,
    wait: bool = True,
    timeout: Optional[float] = None,
    callback: Optional[Callable[[TaskInfo], None]] = None
) -> Any:
    """
    Convenience function to execute a remote function asynchronously
    
    Args:
        function_path: Function path
        args: Positional arguments
        kwargs: Keyword arguments
        options: Options for the task
        server_port: Port of the shared tensor server
        verbose_debug: Whether to enable verbose debug logging
        poll_interval: Interval in seconds for polling task status
        wait: Whether to wait for completion
        timeout: Maximum time to wait
        callback: Status update callback
        
    Returns:
        Function result if wait=True, task ID if wait=False
    """
    with AsyncSharedTensorClient(server_port, verbose_debug, poll_interval) as client:
        return client.execute_function_async(function_path, args, kwargs, options, wait, timeout, callback)
