"""
Shared Tensor Provider

This module provides a provider for sharing functions across processes using JSON-RPC.
"""

import inspect
import os
import logging
from functools import wraps
from typing import Any, Dict, Callable, Optional
from shared_tensor.client import SharedTensorClient
from shared_tensor.errors import SharedTensorProviderError


__all__ = ["SharedTensorProvider"]

logger = logging.getLogger(__name__)
global_rank = int(os.getenv("RANK", 0))


class SharedTensorProvider:

    def __init__(self, server_port: int = 2537 + global_rank, verbose_debug: bool = False, default_enabled: bool = True):
        self.server_port: int = server_port
        self.server_mode = os.getenv("__SHARED_TENSOR_SERVER_MODE__", "false")
        self.verbose_debug = verbose_debug
        logger.debug(f"SharedTensorProvider initialized with server port {self.server_port}, server mode {self.server_mode}, and verbose debug {self.verbose_debug}")
        self._registered_functions: Dict[str, Dict[str, Any]] = {}
        self._enabled = os.getenv("__SHARED_TENSOR_ENABLED__", "true" if default_enabled else "false") == "true"
        self._client = None

    def _get_function_path(self, func: Callable) -> str:
        """Get the importable path of a function in format 'module.submodule:function_name'"""
        module = inspect.getmodule(func)
        
        if module is None:
            raise SharedTensorProviderError(f"Failed to get full qualified name for function {func.__name__}, function module is missing")
        
        module_name = module.__name__
        
        if module_name == "__main__":
            if hasattr(module, '__file__') and module.__file__:
                file_path = module.__file__
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                # For test files, we need to include the full path
                if 'tests' in file_path:
                    # Extract the module path from the file path
                    path_parts = file_path.split(os.sep)
                    if 'tests' in path_parts:
                        test_idx = path_parts.index('tests')
                        module_parts = path_parts[test_idx:]
                        # Remove .py extension from last part
                        module_parts[-1] = os.path.splitext(module_parts[-1])[0]
                        module_name = '.'.join(module_parts)
                    else:
                        module_name = file_name
                else:
                    module_name = file_name
            else:
                raise SharedTensorProviderError(f"Failed to get full qualified name for function {func.__name__}, function module file path is empty")
        
        if hasattr(func, '__qualname__'):
            qualname = func.__qualname__
            # Only reject true nested functions with <locals>, but allow test functions
            # Test functions might have qualname like "TestClass.test_method.<locals>.test_function"
            # but we can still use them by extracting the actual function name
            if '<locals>' in qualname:
                # For nested functions, try to extract the innermost function name
                # This allows test functions defined inside test methods to work
                func_path = func.__name__
                logger.warning(f"Function {func.__name__} appears to be nested (qualname: {qualname}), using function name only")
            else:
                func_path = qualname
        else:
            func_path = func.__name__
        
        return f"{module_name}:{func_path}"

    def share(self, name: Optional[str] = None, singleton: bool = True, singleton_key_formatter: Optional[str] = None):
        """Decorator to register a function for remote sharing
        
        Args:
            name: Optional custom name for the function
            singleton: Whether to use a singleton instance of the function result
            singleton_key_formatter: Formatter for cached results

        Returns:
            Decorator function that registers the function for remote sharing
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
            }
            self._registered_functions[func_name] = function_info
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_remote_function(func_name, args, kwargs, options)
            
            return wrapper
        return decorator

    def _get_client(self) -> SharedTensorClient:
        """Get or create JSON-RPC client"""
        if self._client is None:
            logger.debug(f"Creating new JSON-RPC client with server port {self.server_port}")
            self._client = SharedTensorClient(self.server_port, verbose_debug=self.verbose_debug)
            logger.debug(f"JSON-RPC client created with server port {self.server_port}")
        return self._client

    def _execute_remote_function(self, func_name: str, args: tuple, kwargs: dict, options: dict) -> Any:
        """Execute function remotely using JSON-RPC client"""
        try:
            if self.verbose_debug:
                logger.debug(f"Executing remote function {func_name} with args {args} and kwargs {kwargs}")
            else:
                logger.debug(f"Executing remote function {func_name}")

            if func_name not in self._registered_functions:
                raise SharedTensorProviderError(f"Function {func_name} not registered")
            
            function_info = self._registered_functions[func_name]
            function_path = function_info['function_path']
            client = self._get_client()
            logger.debug(f"Executing remote function {func_name} with function path {function_path} and options {options}")
            return client.execute_function(function_path, args, kwargs, options)
                
        except Exception as e:
            logger.warning(f"Failed to execute remote function {func_name}: {str(e)}")
            raise SharedTensorProviderError(f"Failed to execute remote function {func_name}: {str(e)}")
    
    def close(self):
        """Close the provider and its client connection"""
        if self._client:
            logger.debug(f"Closing JSON-RPC client with server port {self.server_port}")
            self._client.close()
            logger.debug(f"JSON-RPC client closed with server port {self.server_port}")
            self._client = None