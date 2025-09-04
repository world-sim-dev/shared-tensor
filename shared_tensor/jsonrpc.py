"""
JSON-RPC 2.0 implementation for shared tensor communication

Implements JSON-RPC 2.0 specification: https://www.jsonrpc.org/specification
"""

import json
import uuid
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 Request object"""
    method: str
    params: Optional[Union[Dict[str, Any], list]] = None
    id: Optional[Union[str, int]] = None
    jsonrpc: str = "2.0"
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Remove None params to keep the request clean
        if result['params'] is None:
            del result['params']
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 Response object"""
    id: Optional[Union[str, int]]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Only include result OR error, not both
        if self.error is not None:
            result.pop('result', None)
        else:
            result.pop('error', None)
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


@dataclass
class JsonRpcError:
    """JSON-RPC 2.0 Error object"""
    code: int
    message: str
    data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if result['data'] is None:
            del result['data']
        return result


# Standard JSON-RPC error codes
class JsonRpcErrorCodes:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom error codes for shared tensor
    FUNCTION_IMPORT_ERROR = -32001
    FUNCTION_EXECUTION_ERROR = -32002
    SERIALIZATION_ERROR = -32003


def create_success_response(request_id: Optional[Union[str, int]], result: Any) -> JsonRpcResponse:
    """Create a successful JSON-RPC response"""
    return JsonRpcResponse(id=request_id, result=result)


def create_error_response(
    request_id: Optional[Union[str, int]], 
    code: int, 
    message: str, 
    data: Optional[Any] = None
) -> JsonRpcResponse:
    """Create an error JSON-RPC response"""
    error = JsonRpcError(code=code, message=message, data=data)
    return JsonRpcResponse(id=request_id, error=error.to_dict())


def parse_request(json_str: str) -> JsonRpcRequest:
    """Parse JSON string into JsonRpcRequest object"""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    # Validate required fields
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    
    if data.get('jsonrpc') != '2.0':
        raise ValueError("Invalid jsonrpc version, must be '2.0'")
    
    if 'method' not in data:
        raise ValueError("Missing required field 'method'")
    
    return JsonRpcRequest(
        method=data['method'],
        params=data.get('params'),
        id=data.get('id'),
        jsonrpc=data['jsonrpc']
    )


def parse_response(json_str: str) -> JsonRpcResponse:
    """Parse JSON string into JsonRpcResponse object"""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    # Validate required fields
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    
    if data.get('jsonrpc') != '2.0':
        raise ValueError("Invalid jsonrpc version, must be '2.0'")
    
    if 'id' not in data:
        raise ValueError("Missing required field 'id'")
    
    # Must have either result or error
    has_result = 'result' in data
    has_error = 'error' in data
    
    if not has_result and not has_error:
        raise ValueError("Response must have either 'result' or 'error'")
    
    if has_result and has_error:
        raise ValueError("Response cannot have both 'result' and 'error'")
    
    return JsonRpcResponse(
        id=data['id'],
        result=data.get('result'),
        error=data.get('error'),
        jsonrpc=data['jsonrpc']
    )
