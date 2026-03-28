from __future__ import annotations

import multiprocessing as mp
import multiprocessing.reduction as mp_reduction
import sys
from pathlib import Path

import pytest

from shared_tensor.errors import SharedTensorCapabilityError, SharedTensorConfigurationError
from shared_tensor.utils import (
    CONTROL_ENCODING,
    SHARED_TENSOR_BASE_PATH_ENV,
    TORCH_ENCODING,
    TRANSFORMERS_MODEL_ENCODING,
    _normalize_for_cache,
    _validate_call_payload,
    build_cache_key,
    deserialize_payload,
    format_cache_key,
    payload_uses_cuda,
    resolve_device_index,
    resolve_runtime_socket_path,
    resolve_server_base_path,
    serialize_call_payloads,
    serialize_empty_payload,
    serialize_payload,
    validate_call_payload_for_transport,
    validate_payload_for_transport,
)

torch = pytest.importorskip("torch")

try:
    import transformers as _transformers
except ImportError:  # pragma: no cover - optional dependency
    _transformers = None


if _transformers is not None:
    class _TinyTransformersConfig(_transformers.PretrainedConfig):
        model_type = "tiny-transformers-test"

        def __init__(self, hidden_size: int = 4, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = hidden_size


    class _TinyParametrizedPreTrainedModel(_transformers.PreTrainedModel):
        config_class = _TinyTransformersConfig

        def __init__(self, config: _TinyTransformersConfig):
            super().__init__(config)
            self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            torch.nn.utils.parametrizations.weight_norm(self.proj, name="weight")
            with torch.no_grad():
                self.proj.parametrizations.weight.original0.fill_(1.0)
                self.proj.parametrizations.weight.original1.fill_(0.25)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.proj(values)

else:  # pragma: no cover - optional dependency
    _TinyTransformersConfig = None
    _TinyParametrizedPreTrainedModel = None


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


def test_cache_key_can_use_format_string() -> None:
    def op(value, scale=1):
        return value

    left = build_cache_key("sum", (1,), {"scale": 3}, func=op, cache_format_key="{scale}")
    right = build_cache_key("sum", (99,), {"scale": 3}, func=op, cache_format_key="{scale}")
    assert left == right


def test_format_cache_key_defaults_from_function_signature() -> None:
    def op(value, scale=7):
        return value

    assert format_cache_key(op, (1,), {}, "{scale}") == "7"


def test_resolve_server_base_path_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SHARED_TENSOR_BASE_PATH_ENV, "/tmp/shared-tensor-test")
    assert resolve_server_base_path("/tmp/default") == "/tmp/shared-tensor-test"


def test_resolve_runtime_socket_path_offsets_device_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shared_tensor.utils.resolve_device_index", lambda device_index=None: 4)
    assert resolve_runtime_socket_path("/tmp/shared-tensor") == "/tmp/shared-tensor-4.sock"


def test_resolve_device_index_rejects_negative_value() -> None:
    with pytest.raises(SharedTensorConfigurationError, match="must be >= 0"):
        resolve_device_index(-1)


def test_resolve_device_index_returns_explicit_value() -> None:
    assert resolve_device_index(3) == 3


def test_torch_forking_pickler_falls_back_to_stdlib(monkeypatch: pytest.MonkeyPatch) -> None:
    reductions = torch.multiprocessing.reductions
    monkeypatch.delattr(reductions, "ForkingPickler", raising=False)
    from shared_tensor.utils import _torch_forking_pickler

    assert _torch_forking_pickler() is mp_reduction.ForkingPickler


def test_validate_call_payload_allows_scalar_kwargs_with_cuda_tensors() -> None:
    payload = {"tensor": torch.ones(1, device="cuda") if torch.cuda.is_available() else None, "version": 3}
    if payload["tensor"] is None:
        pytest.skip("CUDA is not available")
    _validate_call_payload(payload, allow_dict_keys=True)


def test_validate_payload_for_transport_rejects_non_string_dict_key() -> None:
    value = torch.arange(1, dtype=torch.float32) if not torch.cuda.is_available() else torch.ones(1, device="cuda")
    with pytest.raises(SharedTensorCapabilityError, match="Dictionary payload keys must be strings"):
        validate_payload_for_transport({1: value}, allow_dict_keys=True)


def test_validate_call_payload_for_transport_allows_scalars() -> None:
    validate_call_payload_for_transport({"version": 3}, allow_dict_keys=True)


def test_serialize_empty_payload_rejects_non_empty_values() -> None:
    with pytest.raises(SharedTensorCapabilityError, match="Only empty args/kwargs"):
        serialize_empty_payload((1,))


def test_build_cache_key_requires_func_when_format_key_is_used() -> None:
    with pytest.raises(SharedTensorConfigurationError, match="func is required"):
        build_cache_key("op", (), {}, cache_format_key="fixed")


def test_normalize_for_cache_supports_bytes_sets_and_repr_fallback() -> None:
    class Token:
        def __repr__(self) -> str:
            return "Token()"

    normalized = {
        "bytes": _normalize_for_cache(b"ab"),
        "set": _normalize_for_cache({2, 1}),
        "obj": _normalize_for_cache(Token()),
    }

    assert normalized["bytes"] == {"__bytes__": "6162"}
    assert normalized["set"] == [1, 2]
    assert normalized["obj"] == {"__repr__": "Token()", "__type__": "Token"}


def test_payload_uses_cuda_returns_false_for_plain_objects() -> None:
    assert payload_uses_cuda({"value": 1, "items": ["x"]}) is False


def _produce_cuda_payload(kind: str, payload_queue, release_queue) -> None:
    if kind == "tensor":
        obj = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
        expected = obj.detach().cpu()
    elif kind == "module":
        obj = torch.nn.Linear(2, 2).cuda()
        expected = obj.weight.detach().cpu()
    elif kind == "transformers_parametrized_module":
        if _TinyParametrizedPreTrainedModel is None or _TinyTransformersConfig is None:  # pragma: no cover - optional dependency
            raise AssertionError("transformers is not installed")
        obj = _TinyParametrizedPreTrainedModel(_TinyTransformersConfig(hidden_size=4)).cuda().eval()
        sample = torch.ones(1, 4, device="cuda")
        expected = obj(sample).detach().cpu()
    else:  # pragma: no cover - defensive
        raise AssertionError(f"Unknown kind: {kind}")
    encoding, payload = serialize_payload(obj)
    payload_queue.put((encoding, payload, expected))
    release_queue.get(timeout=10)


def _round_trip_cuda_payload(kind: str):
    ctx = mp.get_context("spawn")
    payload_queue = ctx.Queue()
    release_queue = ctx.Queue()
    process = ctx.Process(target=_produce_cuda_payload, args=(kind, payload_queue, release_queue))
    process.start()
    try:
        encoding, payload, expected = payload_queue.get(timeout=10)
        result = deserialize_payload(encoding, payload)
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


@pytest.mark.gpu
def test_cuda_transformers_parametrized_module_round_trip() -> None:
    if _TinyParametrizedPreTrainedModel is None:
        pytest.skip("transformers is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    encoding, result, expected = _round_trip_cuda_payload("transformers_parametrized_module")

    assert encoding == TRANSFORMERS_MODEL_ENCODING
    assert isinstance(result, _TinyParametrizedPreTrainedModel)
    assert next(result.parameters()).is_cuda
    sample = torch.ones(1, 4, device="cuda")
    actual = result(sample).detach().cpu()
    assert torch.allclose(actual, expected)


@pytest.mark.gpu
def test_cuda_transformers_round_trip_recovers_missing_client_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if _TinyParametrizedPreTrainedModel is None:
        pytest.skip("transformers is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import shared_tensor.utils as utils

    module_name = _TinyParametrizedPreTrainedModel.__module__
    module = sys.modules[module_name]
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        pytest.skip("test module file is unavailable")

    fake_root = tmp_path / "transformers_modules" / "demo"
    fake_root.mkdir(parents=True)
    fake_module_path = fake_root / "modeling_demo.py"
    fake_module_path.write_text(Path(module_file).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(module, "__file__", str(fake_module_path))

    model = _TinyParametrizedPreTrainedModel(_TinyTransformersConfig(hidden_size=4)).cuda().eval()
    encoding, payload = serialize_payload(model)

    monkeypatch.delitem(sys.modules, module_name, raising=False)

    result = deserialize_payload(encoding, payload)

    assert type(result).__name__ == type(model).__name__
    assert next(result.parameters()).is_cuda
    staged_root = Path(utils.tempfile.gettempdir()) / "shared-tensor-transformers-modules"
    assert (staged_root / module_name.replace(".", "/")).with_suffix(".py").exists()


@pytest.mark.gpu
def test_cuda_transformers_round_trip_preserves_named_state_layout() -> None:
    if _TinyParametrizedPreTrainedModel is None:
        pytest.skip("transformers is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    model = _TinyParametrizedPreTrainedModel(_TinyTransformersConfig(hidden_size=4)).cuda().eval()
    expected_parameters = list(model.named_parameters(remove_duplicate=False))
    expected_buffers = list(model.named_buffers(remove_duplicate=False))

    encoding, payload = serialize_payload(model)
    result = deserialize_payload(encoding, payload)

    actual_parameters = list(result.named_parameters(remove_duplicate=False))
    actual_buffers = list(result.named_buffers(remove_duplicate=False))

    assert [name for name, _ in actual_parameters] == [name for name, _ in expected_parameters]
    assert [name for name, _ in actual_buffers] == [name for name, _ in expected_buffers]
    for (expected_name, expected_value), (actual_name, actual_value) in zip(
        expected_parameters, actual_parameters, strict=True
    ):
        assert expected_name == actual_name
        assert actual_value.is_cuda
        assert torch.equal(actual_value.detach().cpu(), expected_value.detach().cpu())
