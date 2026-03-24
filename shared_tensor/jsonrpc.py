"""Minimal JSON-RPC 2.0 helpers used by the local HTTP transport."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from shared_tensor.errors import SharedTensorProtocolError


@dataclass(slots=True)
class JsonRpcRequest:
    method: str
    params: dict[str, Any] | None = None
    id: str | int | None = None
    jsonrpc: str = "2.0"

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["params"] is None:
            payload.pop("params")
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(slots=True)
class JsonRpcResponse:
    id: str | int | None
    result: Any | None = None
    error: dict[str, Any] | None = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.error is None:
            payload.pop("error")
        else:
            payload.pop("result")
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class JsonRpcErrorCodes:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    ENDPOINT_NOT_FOUND = -32001
    SERIALIZATION_ERROR = -32002
    CAPABILITY_ERROR = -32003
    TASK_ERROR = -32004
    REMOTE_ERROR = -32005


def create_success_response(request_id: str | int | None, result: Any) -> JsonRpcResponse:
    return JsonRpcResponse(id=request_id, result=result)


def create_error_response(
    request_id: str | int | None,
    code: int,
    message: str,
    data: Any | None = None,
) -> JsonRpcResponse:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return JsonRpcResponse(id=request_id, error=error)


def parse_request(raw: str) -> JsonRpcRequest:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SharedTensorProtocolError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SharedTensorProtocolError("Request must be a JSON object")
    if data.get("jsonrpc") != "2.0":
        raise SharedTensorProtocolError("Invalid jsonrpc version, expected '2.0'")
    if not isinstance(data.get("method"), str) or not data["method"]:
        raise SharedTensorProtocolError("Missing required field 'method'")
    params = data.get("params")
    if params is not None and not isinstance(params, dict):
        raise SharedTensorProtocolError("'params' must be an object when provided")
    return JsonRpcRequest(method=data["method"], params=params, id=data.get("id"), jsonrpc="2.0")


def parse_response(raw: str) -> JsonRpcResponse:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SharedTensorProtocolError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SharedTensorProtocolError("Response must be a JSON object")
    if data.get("jsonrpc") != "2.0":
        raise SharedTensorProtocolError("Invalid jsonrpc version, expected '2.0'")
    if "id" not in data:
        raise SharedTensorProtocolError("Missing required field 'id'")
    has_result = "result" in data
    has_error = "error" in data
    if has_result == has_error:
        raise SharedTensorProtocolError("Response must contain exactly one of 'result' or 'error'")
    return JsonRpcResponse(id=data["id"], result=data.get("result"), error=data.get("error"), jsonrpc="2.0")
