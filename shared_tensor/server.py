"""
Shared Tensor JSON-RPC Server

Handles remote function execution requests using JSON-RPC 2.0 protocol.
"""

import os
import pickle
import logging
import threading
import time
import sys
from typing import Any, Dict
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import json

from shared_tensor.jsonrpc import (
    JsonRpcErrorCodes,
    parse_request,
    create_success_response,
    create_error_response
)
from shared_tensor.utils import import_function_from_path, serialize_result, deserialize_args, format_cache_key
from shared_tensor.async_task import get_task_manager, TaskStatus

logger = logging.getLogger(__name__)


class SharedTensorRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for JSON-RPC requests"""
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(format % args)
    
    def do_POST(self):
        """Handle POST requests"""
        # Only accept requests to /jsonrpc endpoint
        if self.path != '/jsonrpc':
            self.send_error(404, "Not Found")
            return
        
        # Check content type
        content_type = self.headers.get('Content-Type', '')
        if not content_type.startswith('application/json'):
            self.send_error(400, "Content-Type must be application/json")
            return
        
        # Read request body
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            request_data = self.rfile.read(content_length).decode('utf-8')
        except (ValueError, UnicodeDecodeError) as e:
            self.send_error(400, f"Invalid request body: {e}")
            return
        
        # Process JSON-RPC request
        response = self.server.process_jsonrpc_request(request_data)
        
        # Send response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))
    
    def do_GET(self):
        """Handle GET requests (for health checks, etc.)"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            response = json.dumps({"status": "healthy", "timestamp": time.time()})
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404, "Not Found")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server"""
    daemon_threads = True
    allow_reuse_address = True


class SharedTensorServer:
    """
    JSON-RPC server for shared tensor remote execution
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        """
        Initialize the server
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
        
        # Server statistics
        self.stats = {
            'start_time': None,
            'requests_processed': 0,
            'errors_encountered': 0,
            'functions_executed': {}
        }
        self.singleton_cache = {}
    
    def process_jsonrpc_request(self, request_data: str) -> str:
        """
        Process a JSON-RPC request
        
        Args:
            request_data: Raw JSON request string
            
        Returns:
            JSON response string
        """
        request_id = None
        
        try:
            # Parse JSON-RPC request
            try:
                request = parse_request(request_data)
                request_id = request.id
            except ValueError as e:
                logger.error(f"Invalid JSON-RPC request: {e}")
                response = create_error_response(
                    None, 
                    JsonRpcErrorCodes.INVALID_REQUEST, 
                    str(e)
                )
                return response.to_json()
            
            # Update statistics
            self.stats['requests_processed'] += 1
            
            # Route method calls
            method = request.method
            params = request.params or {}
            
            logger.info(f"Processing method: {method}, params: {params}")
            
            if method == "execute_function":
                result = self._handle_execute_function(params)
            elif method == "submit_task":
                result = self._handle_submit_task(params)
            elif method == "get_task_status":
                result = self._handle_get_task_status(params)
            elif method == "cancel_task":
                result = self._handle_cancel_task(params)
            elif method == "list_tasks":
                result = self._handle_list_tasks(params)
            elif method == "ping":
                result = {"pong": True, "timestamp": time.time()}
            elif method == "get_server_info":
                result = self._get_server_info()
            elif method == "list_functions":
                result = self._list_functions()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Create success response
            response = create_success_response(request_id, result)
            return response.to_json()
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.stats['errors_encountered'] += 1
            
            # Determine error code based on exception type
            if isinstance(e, ImportError):
                error_code = JsonRpcErrorCodes.FUNCTION_IMPORT_ERROR
            elif isinstance(e, (pickle.PickleError, ValueError)):
                error_code = JsonRpcErrorCodes.SERIALIZATION_ERROR
            else:
                error_code = JsonRpcErrorCodes.FUNCTION_EXECUTION_ERROR
            
            response = create_error_response(request_id, error_code, str(e))
            return response.to_json()
    
    def _handle_execute_function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle execute_function method"""
        # Validate parameters
        required_params = ['function_path', 'args', 'kwargs', 'encoding']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        function_path = params['function_path']
        args_hex = params['args']
        kwargs_hex = params['kwargs']
        options = params.get('options', {})
        encoding = params['encoding']
        
        cache_name = options.get('name')
        singleton = options.get('singleton', False)
        singleton_key_formatter = options.get('singleton_key_formatter') or '{_name_}'
        
        if encoding != 'pickle_hex':
            raise ValueError(f"Unsupported encoding: {encoding}")
        
        logger.info(f"Executing function: {function_path}")
        
        # Update function execution statistics
        if function_path not in self.stats['functions_executed']:
            self.stats['functions_executed'][function_path] = 0
        self.stats['functions_executed'][function_path] += 1
        
        try:
            func = import_function_from_path(function_path)
            
            # Deserialize arguments using ForkingPickler
            args = deserialize_args(args_hex)
            kwargs = deserialize_args(kwargs_hex) if kwargs_hex else {}
            
            cache_key = format_cache_key(cache_name, func, args, kwargs, singleton_key_formatter)
            logger.debug(f"Cache key: {cache_key}")

            if singleton and cache_key in self.singleton_cache:
                logger.info(f"Returning cached result for {cache_key}")
                return self.singleton_cache[cache_key]

            # Execute the function
            result = func(*args, **kwargs)
            
            # Serialize result using ForkingPickler
            result_bytes = serialize_result(result)
            
            result_info = {
                'handler': result_bytes.hex(),
                'function_path': function_path,
                'execution_time': time.time()
            }
            
            if singleton:
                logger.info(f"Caching result for {cache_key}")
                self.singleton_cache[cache_key] = result_info
            
            return result_info
        except Exception as e:
            logger.error(f"Function execution failed: {e}")
            raise
    
    def _handle_submit_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle submit_task method - submit task for async execution"""
        required_params = ['function_path', 'args', 'kwargs', 'encoding']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        function_path = params['function_path']
        args_hex = params['args']
        kwargs_hex = params['kwargs']
        options = params.get('options', {})
        encoding = params['encoding']
        
        if encoding != 'pickle_hex':
            raise ValueError(f"Unsupported encoding: {encoding}")
        
        logger.info(f"Submitting async task: {function_path}")
        
        # Deserialize arguments to validate them
        try:
            args = deserialize_args(args_hex)
            kwargs = deserialize_args(kwargs_hex) if kwargs_hex else {}
        except Exception as e:
            raise ValueError(f"Failed to deserialize arguments: {e}")
        
        # Submit task to task manager
        task_manager = get_task_manager()
        task_id = task_manager.submit_task(function_path, args, kwargs, options)
        
        return {
            'task_id': task_id,
            'function_path': function_path,
            'submitted_at': time.time()
        }
    
    def _handle_get_task_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_task_status method"""
        if 'task_id' not in params:
            raise ValueError("Missing required parameter: task_id")
        
        task_id = params['task_id']
        task_manager = get_task_manager()
        
        task_info = task_manager.get_task(task_id)
        if not task_info:
            raise ValueError(f"Task {task_id} not found")
        
        return task_info.to_dict()
    
    def _handle_cancel_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cancel_task method"""
        if 'task_id' not in params:
            raise ValueError("Missing required parameter: task_id")
        
        task_id = params['task_id']
        task_manager = get_task_manager()
        
        cancelled = task_manager.cancel_task(task_id)
        
        return {
            'task_id': task_id,
            'cancelled': cancelled,
            'timestamp': time.time()
        }
    
    def _handle_list_tasks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle list_tasks method"""
        status_filter = params.get('status')
        task_manager = get_task_manager()
        
        # Convert status string to enum if provided
        status_enum = None
        if status_filter:
            try:
                status_enum = TaskStatus(status_filter)
            except ValueError:
                raise ValueError(f"Invalid status: {status_filter}")
        
        tasks = task_manager.list_tasks(status_enum)
        
        # Convert to dict format
        result = {}
        for task_id, task_info in tasks.items():
            result[task_id] = task_info.to_dict()
        
        return result
    
    def _get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            'server': 'SharedTensorServer',
            'version': '1.0.0',
            'host': self.host,
            'port': self.port,
            'uptime': uptime,
            'stats': self.stats.copy(),
            'python_version': sys.version,
        }
    
    def _list_functions(self) -> Dict[str, Any]:
        """List available functions"""
        return {
            'executed_functions': list(self.stats['functions_executed'].keys()),
            'execution_counts': self.stats['functions_executed'].copy()
        }
    
    def start(self, blocking: bool = True):
        """
        Start the server
        
        Args:
            blocking: If True, block until server stops. If False, run in background.
        """
        if self.running:
            raise RuntimeError("Server is already running")
        
        try:
            # Create HTTP server
            self.server = ThreadedHTTPServer((self.host, self.port), SharedTensorRequestHandler)
            self.server.process_jsonrpc_request = self.process_jsonrpc_request
            
            self.running = True
            self.stats['start_time'] = time.time()
            
            logger.info(f"Starting SharedTensorServer on {self.host}:{self.port}")
            
            if blocking:
                self.server.serve_forever()
            else:
                self.server_thread = threading.Thread(target=self.server.serve_forever)
                self.server_thread.daemon = True
                self.server_thread.start()
                
        except Exception as e:
            self.running = False
            raise RuntimeError(f"Failed to start server: {e}")
    
    def stop(self):
        """Stop the server"""
        if not self.running:
            return
        
        logger.info("Stopping SharedTensorServer")
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
        
        self.running = False
        logger.info("Server stopped")
    
    def __enter__(self):
        self.start(blocking=False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def main():
    """Main entry point for running the server"""
    import argparse
    
    rank = int(os.getenv("RANK", 0))
    parser = argparse.ArgumentParser(description='Shared Tensor JSON-RPC Server')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2537 + rank, help='Server port (default: {})'.format(2537 + rank))
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start server
    server = SharedTensorServer(host=args.host, port=args.port)
    os.environ["__SHARED_TENSOR_SERVER_MODE__"] = "true"
    
    try:
        server.start(blocking=True)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        server.stop()
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
