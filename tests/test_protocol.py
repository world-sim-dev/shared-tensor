from __future__ import annotations

import pytest

from shared_tensor.errors import SharedTensorProtocolError
from shared_tensor.jsonrpc import (
    JsonRpcRequest,
    JsonRpcResponse,
    create_error_response,
    create_success_response,
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


def test_jsonrpc_request_generates_id_and_omits_none_params() -> None:
    request = JsonRpcRequest(method="ping")

    payload = request.to_dict()

    assert isinstance(request.id, str)
    assert "params" not in payload


def test_jsonrpc_response_to_dict_prefers_error_shape() -> None:
    response = JsonRpcResponse(id="abc", error={"code": -1, "message": "boom"})

    payload = response.to_dict()

    assert payload == {
        "id": "abc",
        "error": {"code": -1, "message": "boom"},
        "jsonrpc": "2.0",
    }


def test_create_response_helpers() -> None:
    assert create_success_response("1", {"ok": True}).to_dict()["result"] == {"ok": True}
    assert create_error_response("1", -1, "boom", data={"x": 1}).to_dict()["error"] == {
        "code": -1,
        "message": "boom",
        "data": {"x": 1},
    }


def test_parse_request_rejects_invalid_version() -> None:
    with pytest.raises(SharedTensorProtocolError):
        parse_request('{"jsonrpc": "1.0", "method": "ping", "id": 1}')


def test_parse_request_rejects_invalid_json() -> None:
    with pytest.raises(SharedTensorProtocolError, match="Invalid JSON"):
        parse_request("{")


def test_parse_request_rejects_non_object_payload() -> None:
    with pytest.raises(SharedTensorProtocolError, match="JSON object"):
        parse_request('[]')


def test_parse_request_rejects_missing_method() -> None:
    with pytest.raises(SharedTensorProtocolError, match="Missing required field 'method'"):
        parse_request('{"jsonrpc": "2.0", "id": 1}')


def test_parse_request_rejects_non_object_params() -> None:
    with pytest.raises(SharedTensorProtocolError, match="'params' must be an object"):
        parse_request('{"jsonrpc": "2.0", "method": "ping", "params": []}')


def test_parse_response_requires_result_or_error() -> None:
    with pytest.raises(SharedTensorProtocolError):
        parse_response('{"jsonrpc": "2.0", "id": 1}')


def test_parse_response_rejects_invalid_json() -> None:
    with pytest.raises(SharedTensorProtocolError, match="Invalid JSON"):
        parse_response("{")


def test_parse_response_rejects_non_object_payload() -> None:
    with pytest.raises(SharedTensorProtocolError, match="JSON object"):
        parse_response('[]')


def test_parse_response_rejects_missing_id() -> None:
    with pytest.raises(SharedTensorProtocolError, match="Missing required field 'id'"):
        parse_response('{"jsonrpc": "2.0", "result": {}}')


def test_parse_response_rejects_both_result_and_error() -> None:
    with pytest.raises(SharedTensorProtocolError, match="exactly one"):
        parse_response('{"jsonrpc": "2.0", "id": 1, "result": {}, "error": {}}')
