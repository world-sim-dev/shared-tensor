from __future__ import annotations

import pytest

from shared_tensor.errors import SharedTensorProtocolError
from shared_tensor.jsonrpc import (
    JsonRpcRequest,
    JsonRpcResponse,
    parse_request,
    parse_response,
)


def test_jsonrpc_request_round_trip() -> None:
    request = JsonRpcRequest(method="ping", params={"value": 1})
    parsed = parse_request(request.to_json())
    assert parsed.method == "ping"
    assert parsed.params == {"value": 1}


def test_jsonrpc_response_round_trip() -> None:
    response = JsonRpcResponse(id="abc", result={"ok": True})
    parsed = parse_response(response.to_json())
    assert parsed.id == "abc"
    assert parsed.result == {"ok": True}


def test_parse_request_rejects_invalid_version() -> None:
    with pytest.raises(SharedTensorProtocolError):
        parse_request('{"jsonrpc": "1.0", "method": "ping", "id": 1}')


def test_parse_response_requires_result_or_error() -> None:
    with pytest.raises(SharedTensorProtocolError):
        parse_response('{"jsonrpc": "2.0", "id": 1}')
