"""Utility helpers shared by the provider, server, and clients."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
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
_EMPTY_TUPLE = ()
_EMPTY_DICT: dict[str, Any] = {}


def _torch_forking_pickler() -> type | None:
    if TORCH_MODULE is None:
        return None
    return cast(type, TORCH_MODULE.multiprocessing.reductions.ForkingPickler)


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


def validate_payload_for_transport(obj: Any, *, allow_dict_keys: bool = False) -> None:
    """Validate that a payload fits the supported CUDA torch transport contract."""
    _validate_torch_payload(obj, allow_dict_keys=allow_dict_keys)


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
        _validate_torch_payload(args)
        _validate_torch_payload(kwargs, allow_dict_keys=True)
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


def build_cache_key(endpoint_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Build a deterministic cache key for endpoint calls."""
    digest = hashlib.sha256()
    digest.update(endpoint_name.encode("utf-8"))
    digest.update(pickle.dumps((_normalize_for_cache(args), _normalize_for_cache(kwargs)), protocol=4))
    return digest.hexdigest()


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


def resolve_legacy_endpoint_name(function_path: str) -> str:
    """Map an old-style function path to the endpoint name portion."""
    if not function_path:
        raise SharedTensorConfigurationError("function_path must not be empty")
    if ":" not in function_path:
        return function_path
    _, endpoint = function_path.rsplit(":", 1)
    return endpoint.split(".")[-1]


def load_object(target: str) -> Any:
    """Load ``module:attribute`` references for CLI entrypoints."""
    if ":" not in target:
        raise SharedTensorConfigurationError(
            f"Invalid target '{target}'. Expected 'module:attribute'."
        )
    module_name, attr_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for attr in attr_name.split("."):
        obj = getattr(obj, attr)
    return obj


def infer_function_path(func: Callable[..., Any]) -> str:
    """Return an importable path for debugging and compatibility only."""
    module = inspect.getmodule(func)
    if module is None or not getattr(module, "__name__", None):
        raise SharedTensorConfigurationError(f"Unable to resolve module for {func!r}")
    module_name = module.__name__
    module_file = getattr(module, "__file__", None)
    if module_name == "__main__" and module_file:
        module_name = inspect.getmodulename(module_file) or module_name
    return f"{module_name}:{func.__qualname__}"


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
