"""
Async Provider for Shared Tensor

Extends the provider pattern to support async task execution
"""

import os
import logging
from functools import wraps
from typing import Any, Dict, Callable, Optional

from shared_tensor.errors import SharedTensorProviderError
from shared_tensor.provider import SharedTensorProvider
from shared_tensor.async_client import AsyncSharedTensorClient
from shared_tensor.async_task import TaskInfo


__all__ = ["AsyncSharedTensorProvider"]

logger = logging.getLogger(__name__)
global_rank = int(os.getenv("RANK", 0))


class AsyncSharedTensorProvider(SharedTensorProvider):
    """
    Async provider for shared tensor operations
    
    Supports both sync and async execution modes
    """
    
    def __init__(self, server_port: int = 2537 + global_rank, verbose_debug: bool = False, poll_interval: float = 1.0, default_enabled: bool = True):
        super().__init__(server_port=server_port, verbose_debug=verbose_debug, default_enabled=default_enabled)
        self.poll_interval = poll_interval
        logger.debug(f"AsyncSharedTensorProvider initialized with server port {server_port}, verbose debug {verbose_debug}, and poll interval {poll_interval}")
        self._async_client = None
    
    def _get_async_client(self) -> AsyncSharedTensorClient:
        """Get or create async client"""
        if self._async_client is None:
            logger.debug(f"Creating new async client with server port {self.server_port} and poll interval {self.poll_interval}")
            self._async_client = AsyncSharedTensorClient(self.server_port, self.verbose_debug, self.poll_interval)
            logger.debug(f"Async client created with server port {self.server_port} and poll interval {self.poll_interval}")
        return self._async_client
    
    def share(self, name: Optional[str] = None, wait: bool = True, singleton: bool = True, singleton_key_formatter: Optional[str] = None):
        """
        Decorator to register a function for async remote sharing
        
        Args:
            name: Optional custom name for the function
            wait: Whether to wait for completion by default
            singleton: Whether to use a singleton instance of the function result
            singleton_key_formatter: Formatter for cached results
        """
        def decorator(func: Callable):
            func_name = name or func.__name__

            if self.server_mode == "true":
                logger.debug(f"Server mode is true, returning function {func_name} without registering")
                return func
            
            if not self._enabled:
                logger.debug(f"SharedTensor is disabled, returning function {func_name} without registering")
                return func

            logger.debug(f"Server mode is false, registering function {func_name}")

            function_path = self._get_function_path(func)
            logger.debug(f"Function {func_name} registered with function path {function_path}")
            
            options = {
                'name': func_name,
                'singleton': singleton,
                'singleton_key_formatter': singleton_key_formatter,
            }

            function_info = {
                'name': func_name,
                'function_path': function_path,
                'options': options,
                'async_default_wait': wait
            }
            
            self._registered_functions[func_name] = function_info
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_async_function(func_name, args, kwargs, options)
            
            wrapper.submit_async = lambda *args, **kwargs: self._submit_async_function(func_name, args, kwargs, options)
            wrapper.execute_async = lambda *args, wait=wait, timeout=None, callback=None, **kwargs: \
                self._execute_async_function_with_options(func_name, args, kwargs, options, wait, timeout, callback)
            
            return wrapper
        return decorator
    
    def _submit_async_function(self, func_name: str, args: tuple, kwargs: dict, options: dict) -> str:
        """Submit function for async execution, return task ID"""
        try:
            if func_name not in self._registered_functions:
                raise SharedTensorProviderError(f"Function {func_name} not registered")
            
            function_info = self._registered_functions[func_name]
            function_path = function_info['function_path']
            
            async_client = self._get_async_client()
            logger.debug(f"Submitting async function {func_name} with function path {function_path} and options {options}")
            return async_client.submit_task(function_path, args, kwargs, options)
                
        except Exception as e:
            raise SharedTensorProviderError(f"Failed to submit async function {func_name}: {str(e)}")
    
    def _execute_async_function(self, func_name: str, args: tuple, kwargs: dict, options: dict) -> Any:
        """Execute function using default async settings"""
        function_info = self._registered_functions[func_name]
        wait = function_info.get('async_default_wait', True)
        if wait:
            return self._execute_async_function_with_options(func_name, args, kwargs, options, True, None, None)
        else:
            return self._submit_async_function(func_name, args, kwargs, options)
    
    def _execute_async_function_with_options(self, func_name: str, args: tuple, kwargs: dict, options: dict,
                                           wait: bool, timeout: Optional[float], 
                                           callback: Optional[Callable[[TaskInfo], None]]) -> Any:
        """Execute function with specific async options"""
        try:
            if func_name not in self._registered_functions:
                raise SharedTensorProviderError(f"Function {func_name} not registered")
            
            function_info = self._registered_functions[func_name]
            function_path = function_info['function_path']
            
            async_client = self._get_async_client()
            logger.debug(f"Executing async function {func_name} with function path {function_path} and options {options}")
            return async_client.execute_function_async(function_path, args, kwargs, options, wait, timeout, callback)
        except Exception as e:
            raise SharedTensorProviderError(f"Failed to execute async function {func_name}: {str(e)}")
    
    def get_task_status(self, task_id: str) -> TaskInfo:
        """Get status of a task"""
        async_client = self._get_async_client()
        logger.debug(f"Getting status of task {task_id}")
        return async_client.get_task_status(task_id)
    
    def get_task_result(self, task_id: str) -> Any:
        """Get result of a completed task"""
        async_client = self._get_async_client()
        logger.debug(f"Getting result of task {task_id}")
        return async_client.get_task_result(task_id)
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None,
                     callback: Optional[Callable[[TaskInfo], None]] = None) -> Any:
        """Wait for a task to complete"""
        async_client = self._get_async_client()
        logger.debug(f"Waiting for task {task_id} with timeout {timeout} and callback {callback}")
        return async_client.wait_for_task(task_id, timeout, callback)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        async_client = self._get_async_client()
        logger.debug(f"Cancelling task {task_id}")
        return async_client.cancel_task(task_id)
    
    def list_tasks(self, status: Optional[str] = None) -> Dict[str, TaskInfo]:
        """List tasks on the server"""
        async_client = self._get_async_client()
        logger.debug(f"Listing tasks with status {status}")
        return async_client.list_tasks(status)
    
    def close(self):
        """Close the provider and its clients"""
        super().close()
        if self._async_client:
            logger.debug(f"Closing async client")
            self._async_client.close()
            logger.debug(f"Async client closed")
            self._async_client = None