"""
Shared Tensor JSON-RPC Client

Handles communication with remote shared tensor servers using JSON-RPC 2.0 protocol.
"""

import pickle
import requests
import logging
from typing import Any, Dict, Optional

import torch

from shared_tensor.errors import SharedTensorClientError, SharedTensorServerError
from shared_tensor.jsonrpc import (
    JsonRpcRequest, 
    JsonRpcResponse, 
    JsonRpcErrorCodes,
    parse_response,
)
from shared_tensor.utils import serialize_result


__all__ = ["SharedTensorClient", "execute_remote_function"]

logger = logging.getLogger(__name__)


class SharedTensorClient:
    """
    JSON-RPC client for shared tensor communication
    """
    
    def __init__(self, server_port: int = 2537, timeout: int = 30, verbose_debug: bool = False):
        """
        Initialize the client
        
        Args:
            server_port: PORT of the shared tensor server
            timeout: Request timeout in seconds, critical when sync provider used
            verbose_debug: Whether to log detailed debug messages
        """
        self.server_url = f"http://localhost:{server_port}"
        self.verbose_debug = verbose_debug
        self.timeout = timeout

        logger.debug(f"SharedTensorClient initialized with server URL: {self.server_url} with timeout {self.timeout} and verbose_debug {self.verbose_debug}")

        self.session = requests.Session()
        
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SharedTensorClient/1.0'
        })
    
    def _send_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """
        Send a JSON-RPC request to the server
        
        Args:
            request: JsonRpcRequest object
            
        Returns:
            JsonRpcResponse object
            
        Raises:
            SharedTensorClientError: If unable to connect to server
            SharedTensorClientError: If request times out
            SharedTensorClientError: If server response is invalid
        """
        try:
            if self.verbose_debug:
                logger.debug(f"Sending request to server {self.server_url}: {request.to_json()}")
            else:
                logger.debug(f"Sending request to server {self.server_url}")

            response = self.session.post(
                f"{self.server_url}/jsonrpc",
                data=request.to_json(),
                timeout=self.timeout
            )
            if self.verbose_debug:
                logger.debug(f"Received response from server {self.server_url}: {response.text}")
            else:
                logger.debug(f"Received response from server {self.server_url}")

            if response.status_code != 200:
                logger.warning(f"Server {self.server_url} returned HTTP {response.status_code}: {response.text}")
                raise SharedTensorClientError(
                    f"Server {self.server_url} returned HTTP {response.status_code}: {response.text}"
                )
            try:
                rpc_response = parse_response(response.text)
                if self.verbose_debug:
                    logger.debug(f"Parsed JSON-RPC response: {rpc_response}")
                else:
                    logger.debug(f"Parsed JSON-RPC response from server {self.server_url}")
                return rpc_response
            except ValueError as e:
                logger.warning(f"Invalid JSON-RPC response: {e}")
                raise ValueError(f"Invalid JSON-RPC response: {e}")
                
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Unable to connect to server {self.server_url}: {e}")
            raise SharedTensorClientError(f"Unable to connect to server {self.server_url}: {e}")
        except requests.exceptions.Timeout as e:
            logger.warning(f"Request to server {self.server_url} timed out after {self.timeout} seconds: {e}")
            raise SharedTensorClientError(f"Request to server {self.server_url} timed out after {self.timeout} seconds: {e}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request to server {self.server_url} failed: {e}")
            raise SharedTensorClientError(f"Request to server {self.server_url} failed: {e}")
    
    def _create_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> JsonRpcRequest:
        """Create a JSON-RPC request"""
        return JsonRpcRequest(method=method, params=params)
    
    def execute_function(
        self, 
        function_path: str, 
        args: tuple = (), 
        kwargs: Dict[str, Any] = None,
        options: Dict[str, Any] = None
    ) -> Any:
        """
        Execute a function on the remote server
        
        Args:
            function_path: Function path in format "module.submodule:function_name"
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            options: Options for the function
            
        Returns:
            The result returned by the remote function
            
        Raises:
            SharedTensorClientError: If unable to communicate with server
            SharedTensorServerError: If remote execution fails
        """
        if kwargs is None:
            kwargs = {}
            
        try:
            serialized_args = serialize_result(args).hex()
            serialized_kwargs = serialize_result(kwargs).hex()
            
            request = JsonRpcRequest(
                method="execute_function",
                params={
                    "function_path": function_path,
                    "args": serialized_args,
                    "kwargs": serialized_kwargs,
                    "options": options,
                    "encoding": "pickle_hex"
                }
            )
            
            response = self._send_request(request)
            
            if response.error:
                error_code = response.error.get('code', JsonRpcErrorCodes.INTERNAL_ERROR)
                error_message = response.error.get('message', 'Unknown error')
                error_data = response.error.get('data')
                
                raise SharedTensorServerError(
                    f"Remote execution failed [{error_code}]: {error_message}"
                    + (f" - {error_data}" if error_data else "")
                )
            
            if response.result is None:
                return None
                
            result_data = response.result
            if not isinstance(result_data, dict) or 'handler' not in result_data:
                raise SharedTensorClientError("Invalid response format: missing 'handler' field")
            
            handler_hex = result_data['handler']
            handler_bytes = bytes.fromhex(handler_hex)
            return torch.multiprocessing.reducer.ForkingPickler.loads(handler_bytes)
            
        except (pickle.PickleError, ValueError) as e:
            raise SharedTensorClientError(f"Serialization error: {e}")
    
    def ping(self) -> bool:
        """
        Ping the server to check if it's alive
        
        Returns:
            True if server is responding, False otherwise
        """
        try:
            request = JsonRpcRequest(method="ping")
            response = self._send_request(request)
            return response.error is None
        except Exception:
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the server
        
        Returns:
            Dictionary containing server information
        """
        request = JsonRpcRequest(method="get_server_info")
        response = self._send_request(request)
        
        if response.error:
            raise SharedTensorServerError(f"Failed to get server info: {response.error}")
        
        return response.result or {}
    
    def list_functions(self) -> Dict[str, str]:
        """
        List all available functions on the server
        
        Returns:
            Dictionary mapping function names to their paths
        """
        request = JsonRpcRequest(method="list_functions")
        response = self._send_request(request)
        
        if response.error:
            raise SharedTensorServerError(f"Failed to list functions: {response.error}")
        
        return response.result or {}
    
    def close(self):
        """Close the client session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def execute_remote_function(
    function_path: str,
    args: tuple = (),
    kwargs: Dict[str, Any] = None,
    server_port: int = 2537,
    timeout: int = 30,
    verbose_debug: bool = False
) -> Any:
    """
    Convenience function to execute a remote function
    
    Args:
        function_path: Function path in format "module.submodule:function_name"
        args: Positional arguments
        kwargs: Keyword arguments
        server_port: Server port
        timeout: Request timeout in seconds
        verbose_debug: Whether to log detailed debug messages
    Returns:
        Function execution result
    """
    if verbose_debug:
        logger.debug(f"Executing remote function {function_path} with args {args} and kwargs {kwargs} on server {server_port} with timeout {timeout} and verbose_debug {verbose_debug}")
    else:
        logger.debug(f"Executing remote function {function_path} on server {server_port} with timeout {timeout}")
    with SharedTensorClient(server_port=server_port, timeout=timeout, verbose_debug=verbose_debug) as client:
        return client.execute_function(function_path, args, kwargs)
