"""Utility helpers shared by the provider, server, and clients."""

from __future__ import annotations

import hashlib
import inspect
import io
import multiprocessing.reduction as mp_reduction
import os
import pickle
from collections.abc import Callable
from typing import Any, cast

from shared_tensor.errors import (
    SharedTensorCapabilityError,
    SharedTensorConfigurationError,
    SharedTensorSerializationError,
)

try:
    import torch as _torch
except ImportError:  # pragma: no cover - exercised only without torch installed.
    _torch = cast(Any, None)


TORCH_MODULE: Any | None = _torch


CONTROL_ENCODING = "control_pickle"
TORCH_ENCODING = "torch_cuda_ipc"
SHARED_TENSOR_BASE_PORT_ENV = "SHARED_TENSOR_BASE_PORT"
_EMPTY_TUPLE = ()
_EMPTY_DICT: dict[str, Any] = {}


def _torch_forking_pickler() -> type | None:
    if TORCH_MODULE is None:
        return None
    reductions = TORCH_MODULE.multiprocessing.reductions
    init_reductions = getattr(reductions, "init_reductions", None)
    if callable(init_reductions):
        try:
            init_reductions()
        except Exception:
            return cast(type, mp_reduction.ForkingPickler)
    pickler = getattr(reductions, "ForkingPickler", None)
    if pickler is not None:
        return cast(type, pickler)
    return cast(type, mp_reduction.ForkingPickler)


def _raise_unsupported_payload(message: str) -> None:
    raise SharedTensorCapabilityError(
        f"{message}. shared_tensor only supports same-host CUDA torch.Tensor and torch.nn.Module payloads"
    )


def _validate_module_device(module: Any) -> None:
    tensors = list(module.parameters()) + list(module.buffers())
    if not tensors:
        _raise_unsupported_payload("torch.nn.Module payloads must own at least one parameter or buffer")
    if any(not tensor.is_cuda for tensor in tensors):
        _raise_unsupported_payload("torch.nn.Module payloads must stay on CUDA")
    device_indexes = {int(cast(int, tensor.device.index)) for tensor in tensors}
    if len(device_indexes) != 1:
        _raise_unsupported_payload("torch.nn.Module payloads must live on exactly one CUDA device")


def _validate_torch_payload(obj: Any, *, allow_dict_keys: bool = False) -> None:
    if TORCH_MODULE is None:
        raise SharedTensorCapabilityError("PyTorch is required for shared_tensor")

    if isinstance(obj, TORCH_MODULE.Tensor):
        if not obj.is_cuda:
            _raise_unsupported_payload("CPU torch.Tensor payloads are not supported")
        return

    if isinstance(obj, TORCH_MODULE.nn.Module):
        _validate_module_device(obj)
        return

    if isinstance(obj, tuple):
        for item in obj:
            _validate_torch_payload(item)
        return

    if isinstance(obj, list):
        for item in obj:
            _validate_torch_payload(item)
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            if allow_dict_keys:
                if not isinstance(key, str):
                    _raise_unsupported_payload("Dictionary payload keys must be strings")
            else:
                _validate_torch_payload(key)
            _validate_torch_payload(value)
        return

    _raise_unsupported_payload(f"Unsupported payload type: {type(obj).__name__}")


def _validate_call_payload(obj: Any, *, allow_dict_keys: bool = False) -> None:
    if TORCH_MODULE is None:
        raise SharedTensorCapabilityError("PyTorch is required for shared_tensor")

    if isinstance(obj, (str, int, float, bool, type(None), bytes)):
        return

    if isinstance(obj, TORCH_MODULE.Tensor):
        if not obj.is_cuda:
            _raise_unsupported_payload("CPU torch.Tensor payloads are not supported")
        return

    if isinstance(obj, TORCH_MODULE.nn.Module):
        _validate_module_device(obj)
        return

    if isinstance(obj, tuple):
        for item in obj:
            _validate_call_payload(item)
        return

    if isinstance(obj, list):
        for item in obj:
            _validate_call_payload(item)
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            if allow_dict_keys:
                if not isinstance(key, str):
                    _raise_unsupported_payload("Dictionary payload keys must be strings")
            else:
                _validate_call_payload(key)
            _validate_call_payload(value)
        return

    _raise_unsupported_payload(f"Unsupported payload type: {type(obj).__name__}")


def validate_payload_for_transport(obj: Any, *, allow_dict_keys: bool = False) -> None:
    """Validate that a payload fits the supported CUDA torch transport contract."""
    _validate_torch_payload(obj, allow_dict_keys=allow_dict_keys)


def validate_call_payload_for_transport(obj: Any, *, allow_dict_keys: bool = False) -> None:
    """Validate RPC call args/kwargs, allowing scalar controls alongside CUDA payloads."""
    _validate_call_payload(obj, allow_dict_keys=allow_dict_keys)


def _torch_serialize(obj: Any) -> bytes:
    pickler_cls = _torch_forking_pickler()
    if pickler_cls is None:
        raise SharedTensorCapabilityError("PyTorch serialization support is unavailable")
    buffer = io.BytesIO()
    pickler_cls(buffer, pickle.HIGHEST_PROTOCOL).dump(obj)
    return buffer.getvalue()


def _serialize_ipc_payload(obj: Any) -> bytes:
    if TORCH_MODULE is not None and isinstance(obj, (TORCH_MODULE.Tensor, TORCH_MODULE.nn.Module)):
        return _torch_serialize(obj)
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def serialize_payload(obj: Any) -> tuple[str, bytes]:
    """Serialize a CUDA torch payload using PyTorch's IPC-aware pickler."""
    try:
        _validate_torch_payload(obj, allow_dict_keys=isinstance(obj, dict))
        return TORCH_ENCODING, _serialize_ipc_payload(obj)
    except SharedTensorCapabilityError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise SharedTensorSerializationError(f"Failed to serialize payload: {exc}") from exc


def serialize_empty_payload(obj: tuple[Any, ...] | dict[str, Any]) -> tuple[str, bytes]:
    """Serialize empty args/kwargs control payloads."""
    if obj != _EMPTY_TUPLE and obj != _EMPTY_DICT:
        raise SharedTensorCapabilityError(
            "Only empty args/kwargs control payloads may use control encoding"
        )
    return CONTROL_ENCODING, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def serialize_call_payloads(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str, bytes, bytes]:
    """Serialize RPC call args and kwargs with a single compatible encoding."""
    if not args and not kwargs:
        _, args_payload = serialize_empty_payload(_EMPTY_TUPLE)
        return CONTROL_ENCODING, args_payload, serialize_empty_payload(_EMPTY_DICT)[1]

    try:
        _validate_call_payload(args)
        _validate_call_payload(kwargs, allow_dict_keys=True)
        return TORCH_ENCODING, pickle.dumps(args, protocol=pickle.HIGHEST_PROTOCOL), pickle.dumps(
            kwargs, protocol=pickle.HIGHEST_PROTOCOL
        )
    except SharedTensorCapabilityError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise SharedTensorSerializationError(f"Failed to serialize call payloads: {exc}") from exc


def deserialize_payload(encoding: str, data: bytes | str) -> Any:
    """Deserialize a payload produced by :func:`serialize_payload`."""
    payload = bytes.fromhex(data) if isinstance(data, str) else data
    try:
        if encoding == CONTROL_ENCODING:
            return pickle.loads(payload)
        if encoding == TORCH_ENCODING:
            return pickle.loads(payload)
        raise SharedTensorSerializationError(f"Unsupported payload encoding: {encoding}")
    except SharedTensorCapabilityError:
        raise
    except Exception as exc:
        raise SharedTensorSerializationError(f"Failed to deserialize payload: {exc}") from exc


def resolve_server_base_port(default_port: int) -> int:
    """Resolve the configured base port from the environment."""
    raw = os.getenv(SHARED_TENSOR_BASE_PORT_ENV)
    if raw is None:
        return default_port
    try:
        return int(raw)
    except ValueError as exc:
        raise SharedTensorConfigurationError(
            f"{SHARED_TENSOR_BASE_PORT_ENV} must be an integer"
        ) from exc


def resolve_device_index(device_index: int | None = None) -> int:
    """Resolve the CUDA device index for same-GPU port selection."""
    if device_index is not None:
        if device_index < 0:
            raise SharedTensorConfigurationError("device_index must be >= 0")
        return device_index
    if TORCH_MODULE is None or not TORCH_MODULE.cuda.is_available():
        return 0
    return int(TORCH_MODULE.cuda.current_device())


def resolve_runtime_port(base_port: int, device_index: int | None = None) -> int:
    """Resolve the runtime port by offsetting the base port by CUDA device index."""
    return int(base_port) + resolve_device_index(device_index)


def build_cache_key(
    endpoint_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    func: Callable[..., Any] | None = None,
    cache_format_key: str | None = None,
) -> str:
    """Build a deterministic cache key for endpoint calls."""
    digest = hashlib.sha256()
    digest.update(endpoint_name.encode("utf-8"))
    if cache_format_key is not None:
        if func is None:
            raise SharedTensorConfigurationError(
                "func is required when cache_format_key is configured"
            )
        formatted_key = format_cache_key(func, args, kwargs, cache_format_key)
        payload = {"cache_format_key": formatted_key}
    else:
        payload = {
            "args": _normalize_for_cache(args),
            "kwargs": _normalize_for_cache(kwargs),
        }
    digest.update(pickle.dumps(payload, protocol=4))
    return digest.hexdigest()


def format_cache_key(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    cache_format_key: str,
) -> str:
    """Format a cache key from bound function arguments."""
    try:
        bound = inspect.signature(func).bind(*args, **kwargs)
    except TypeError as exc:
        raise SharedTensorConfigurationError(
            f"Unable to bind arguments for cache_format_key on {func.__qualname__}: {exc}"
        ) from exc
    bound.apply_defaults()
    try:
        return cache_format_key.format(**bound.arguments)
    except KeyError as exc:
        missing = exc.args[0]
        raise SharedTensorConfigurationError(
            f"cache_format_key for {func.__qualname__} references unknown argument '{missing}'"
        ) from exc


def _normalize_for_cache(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, bytes):
        return {"__bytes__": obj.hex()}
    if isinstance(obj, dict):
        return {
            str(key): _normalize_for_cache(value)
            for key, value in sorted(obj.items(), key=lambda item: str(item[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_cache(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_normalize_for_cache(item) for item in obj)
    if TORCH_MODULE is not None and isinstance(obj, TORCH_MODULE.Tensor):
        cpu_tensor = obj.detach().cpu().contiguous()
        raw = cpu_tensor.numpy().tobytes()
        return {
            "__tensor__": True,
            "dtype": str(cpu_tensor.dtype),
            "shape": list(cpu_tensor.shape),
            "device": str(obj.device),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if TORCH_MODULE is not None and isinstance(obj, TORCH_MODULE.nn.Module):
        devices = sorted({str(tensor.device) for tensor in list(obj.parameters()) + list(obj.buffers())})
        state = obj.state_dict()
        return {
            "__module__": type(obj).__name__,
            "devices": devices,
            "state": {
                name: _normalize_for_cache(tensor)
                for name, tensor in sorted(state.items(), key=lambda item: item[0])
            },
        }
    return {"__repr__": repr(obj), "__type__": type(obj).__name__}
def capability_snapshot() -> dict[str, Any]:
    snapshot = {
        "torch_available": TORCH_MODULE is not None,
        "cuda_available": False,
        "transport": "same-host-cuda-torch-ipc",
    }
    if TORCH_MODULE is not None:
        snapshot["cuda_available"] = bool(TORCH_MODULE.cuda.is_available())
    return snapshot


def payload_uses_cuda(obj: Any) -> bool:
    if TORCH_MODULE is None:
        return False
    if isinstance(obj, TORCH_MODULE.Tensor):
        return bool(obj.is_cuda)
    if isinstance(obj, TORCH_MODULE.nn.Module):
        return any(tensor.is_cuda for tensor in list(obj.parameters()) + list(obj.buffers()))
    if isinstance(obj, dict):
        return any(payload_uses_cuda(value) for value in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(payload_uses_cuda(item) for item in obj)
    return False
