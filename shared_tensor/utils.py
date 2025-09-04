"""
Remote function executor utility

This module provides utilities for importing and executing functions
from their string paths on the remote server side.
"""

import importlib
import logging
import io
import pickle
import inspect
from typing import Any, Callable

import torch


__all__ = ["serialize_result", "deserialize_args", "import_function_from_path", "format_cache_key"]


logger = logging.getLogger(__name__)


def serialize_result(obj: Any) -> bytes:
    """
    Serialize result using ForkingPickler if obj is a torch.nn.Module or torch.Tensor on GPU, otherwise use standard pickle.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serialized bytes
    """
    buffer = io.BytesIO()
    
    use_forking_pickler = False
    if isinstance(obj, torch.nn.Module):
        obj.cuda()
        obj.share_memory()
        use_forking_pickler = True
    elif isinstance(obj, torch.Tensor):
        obj.cuda()
        obj.share_memory_()
        use_forking_pickler = True

    if use_forking_pickler:
        torch.multiprocessing.reducer.ForkingPickler(buffer).dump(obj)
    else:
        pickle.dump(obj, buffer)
    return buffer.getvalue()


def deserialize_args(data_hex: str) -> Any:
    """
    Deserialize arguments from hex string.
    
    Args:
        data_hex: Hex-encoded serialized data
        
    Returns:
        Deserialized object
    """
    if not data_hex:
        return ()
    
    if isinstance(data_hex, bytes):
        data_bytes = data_hex
    else:
        data_bytes = bytes.fromhex(data_hex)
    return pickle.loads(data_bytes)


def import_function_from_path(function_path: str) -> Callable:
    """
    Import a function from its string path.
    
    Args:
        function_path: Function path in format "module.submodule:function_name"
                      e.g., "abc.addd:get_model" or "mypackage.models:create_model"
    
    Returns:
        The imported function object
    
    Raises:
        ImportError: If the module or function cannot be imported
        ValueError: If the function_path format is invalid
    """
    if ':' not in function_path:
        raise ValueError(f"Invalid function path format: {function_path}. Expected 'module:function'")
    
    module_path, func_name = function_path.split(':', 1)
    
    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Handle nested function names (e.g., "ClassName.method_name")
        func_obj = module
        for attr in func_name.split('.'):
            func_obj = getattr(func_obj, attr)
        
        if not callable(func_obj):
            raise ValueError(f"Object at {function_path} is not callable")
        
        return func_obj
        
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}': {e}")
    except AttributeError as e:
        raise ImportError(f"Cannot find function '{func_name}' in module '{module_path}': {e}")


def format_cache_key(name, func, args, kwargs, singleton_key_formatter):
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    fullkwargs = {}
    for param_name in sig.parameters:
        if param_name in bound_args.arguments:
            value = bound_args.arguments[param_name]
            fullkwargs[param_name] = value
    return singleton_key_formatter.format(_name_=name, **fullkwargs)