from __future__ import annotations

import multiprocessing as mp

import pytest

from shared_tensor.errors import SharedTensorCapabilityError
from shared_tensor.utils import (
    CONTROL_ENCODING,
    TORCH_ENCODING,
    build_cache_key,
    deserialize_payload,
    serialize_call_payloads,
    serialize_empty_payload,
    serialize_payload,
)

torch = pytest.importorskip("torch")


def test_plain_python_payload_is_rejected() -> None:
    with pytest.raises(SharedTensorCapabilityError):
        serialize_payload({"a": [1, 2, 3]})


def test_cpu_tensor_payload_is_rejected() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    with pytest.raises(SharedTensorCapabilityError):
        serialize_payload(tensor)


def test_cpu_module_payload_is_rejected() -> None:
    module = torch.nn.Linear(2, 2)
    with pytest.raises(SharedTensorCapabilityError):
        serialize_payload(module)


def test_empty_control_payload_round_trip() -> None:
    args_encoding, args_payload = serialize_empty_payload(())
    kwargs_encoding, kwargs_payload = serialize_empty_payload({})
    assert args_encoding == CONTROL_ENCODING
    assert kwargs_encoding == CONTROL_ENCODING
    assert deserialize_payload(args_encoding, args_payload) == ()
    assert deserialize_payload(kwargs_encoding, kwargs_payload) == {}


def test_serialize_call_payloads_empty_uses_control_encoding() -> None:
    encoding, args_payload, kwargs_payload = serialize_call_payloads((), {})
    assert encoding == CONTROL_ENCODING
    assert deserialize_payload(encoding, args_payload) == ()
    assert deserialize_payload(encoding, kwargs_payload) == {}


def test_cache_key_is_stable_for_equivalent_inputs() -> None:
    left = build_cache_key("sum", (1, 2), {"scale": 3})
    right = build_cache_key("sum", (1, 2), {"scale": 3})
    assert left == right


def _produce_cuda_payload(kind: str, payload_queue, release_queue) -> None:
    if kind == "tensor":
        obj = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
        expected = obj.detach().cpu()
    elif kind == "module":
        obj = torch.nn.Linear(2, 2).cuda()
        expected = obj.weight.detach().cpu()
    else:  # pragma: no cover - defensive
        raise AssertionError(f"Unknown kind: {kind}")
    encoding, payload = serialize_payload(obj)
    payload_queue.put((encoding, payload.hex(), expected))
    release_queue.get(timeout=10)


def _round_trip_cuda_payload(kind: str):
    ctx = mp.get_context("spawn")
    payload_queue = ctx.Queue()
    release_queue = ctx.Queue()
    process = ctx.Process(target=_produce_cuda_payload, args=(kind, payload_queue, release_queue))
    process.start()
    try:
        encoding, payload_hex, expected = payload_queue.get(timeout=10)
        result = deserialize_payload(encoding, payload_hex)
    finally:
        release_queue.put(True)
        process.join(timeout=10)
    assert process.exitcode == 0
    return encoding, result, expected


@pytest.mark.gpu
def test_cuda_tensor_round_trip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    encoding, result, expected = _round_trip_cuda_payload("tensor")

    assert encoding == TORCH_ENCODING
    assert result.is_cuda
    assert torch.equal(result.cpu(), expected)


@pytest.mark.gpu
def test_cuda_module_round_trip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    encoding, result, expected = _round_trip_cuda_payload("module")

    assert encoding == TORCH_ENCODING
    assert isinstance(result, torch.nn.Linear)
    assert next(result.parameters()).is_cuda
    assert torch.equal(result.weight.detach().cpu(), expected)
