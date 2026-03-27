"""Utility helpers shared by the provider, server, and clients."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import multiprocessing.reduction as mp_reduction
import os
import pickle
import sys
import tempfile
from pathlib import Path
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

try:
    from transformers import PreTrainedModel as _transformers_pretrained_model
except ImportError:  # pragma: no cover - exercised when transformers is not installed.
    _transformers_pretrained_model = cast(Any, None)


TORCH_MODULE: Any | None = _torch
TRANSFORMERS_PRETRAINED_MODEL: Any | None = _transformers_pretrained_model


CONTROL_ENCODING = "control_pickle"
TORCH_ENCODING = "torch_cuda_ipc"
TRANSFORMERS_MODEL_ENCODING = "torch_transformers_pretrained_model"
SHARED_TENSOR_BASE_PATH_ENV = "SHARED_TENSOR_BASE_PATH"
DEFAULT_SOCKET_BASE_PATH = "/tmp/shared-tensor"
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


def _serialize_transformers_model(module: Any) -> bytes:
    config = getattr(module, "config", None)
    if config is None:
        raise SharedTensorSerializationError("transformers model is missing a config")
    module_sources = _collect_transformers_module_sources(type(module).__module__, type(config).__module__)
    payload = {
        "module_sources": module_sources,
        "module_bytes": _torch_serialize(module),
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _serialize_ipc_payload(obj: Any) -> tuple[str, bytes]:
    if (
        TORCH_MODULE is not None
        and TRANSFORMERS_PRETRAINED_MODEL is not None
        and isinstance(obj, TRANSFORMERS_PRETRAINED_MODEL)
    ):
        return TRANSFORMERS_MODEL_ENCODING, _serialize_transformers_model(obj)
    if TORCH_MODULE is not None and isinstance(obj, (TORCH_MODULE.Tensor, TORCH_MODULE.nn.Module)):
        return TORCH_ENCODING, _torch_serialize(obj)
    return CONTROL_ENCODING, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def serialize_payload(obj: Any) -> tuple[str, bytes]:
    """Serialize a CUDA torch payload using PyTorch's IPC-aware pickler."""
    try:
        _validate_torch_payload(obj, allow_dict_keys=isinstance(obj, dict))
        return _serialize_ipc_payload(obj)
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


def _bundle_module_files(module_name: str) -> dict[str, str] | None:
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    module_path = Path(module_file).resolve()
    if module_path.suffix != ".py":
        return None
    transformers_root = None
    if TRANSFORMERS_PRETRAINED_MODEL is not None:
        transformers_file = getattr(sys.modules.get("transformers"), "__file__", None)
        if transformers_file is not None:
            transformers_root = Path(transformers_file).resolve().parent
    if transformers_root is not None:
        try:
            module_path.relative_to(transformers_root)
            if not module_name.startswith("transformers_modules."):
                return None
        except ValueError:
            pass
    files = {module_path.name: module_path.read_text(encoding="utf-8")}
    get_relative_import_files = getattr(importlib.import_module("transformers.dynamic_module_utils"), "get_relative_import_files", None)
    if callable(get_relative_import_files):
        for needed in get_relative_import_files(str(module_path)):
            needed_path = Path(needed).resolve()
            files[needed_path.name] = needed_path.read_text(encoding="utf-8")
    return files


def _collect_transformers_module_sources(*module_names: str) -> dict[str, dict[str, str]]:
    bundles: dict[str, dict[str, str]] = {}
    for module_name in module_names:
        bundle = _bundle_module_files(module_name)
        if bundle:
            bundles[module_name] = bundle
    return bundles


def _stage_transformers_module_sources(module_sources: dict[str, dict[str, str]]) -> None:
    if not module_sources:
        return
    staging_root = Path(tempfile.gettempdir()) / "shared-tensor-transformers-modules"
    staging_root.mkdir(parents=True, exist_ok=True)
    root_str = str(staging_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    for module_name, files in module_sources.items():
        parts = module_name.split(".")
        package_dir = staging_root.joinpath(*parts[:-1])
        package_dir.mkdir(parents=True, exist_ok=True)
        current = staging_root
        for part in parts[:-1]:
            current = current / part
            init_file = current / "__init__.py"
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")
        for filename, content in files.items():
            (package_dir / filename).write_text(content, encoding="utf-8")
    importlib.invalidate_caches()


def _import_transformers_module(module_name: str, module_sources: dict[str, dict[str, str]]) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        _stage_transformers_module_sources(module_sources)
        return importlib.import_module(module_name)


def _deserialize_transformers_model(payload: bytes) -> Any:
    encoded = pickle.loads(payload)
    module_sources = cast(dict[str, dict[str, str]], encoded.get("module_sources", {}))
    for module_name in module_sources:
        _import_transformers_module(module_name, module_sources)
    module_bytes = encoded.get("module_bytes")
    if not isinstance(module_bytes, (bytes, bytearray)):
        raise SharedTensorSerializationError("transformers module payload is invalid")
    model = pickle.loads(module_bytes)
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()
    _validate_module_device(model)
    return model


def deserialize_payload(encoding: str, data: bytes | str) -> Any:
    """Deserialize a payload produced by :func:`serialize_payload`."""
    payload = bytes.fromhex(data) if isinstance(data, str) else data
    try:
        if encoding == CONTROL_ENCODING:
            return pickle.loads(payload)
        if encoding == TORCH_ENCODING:
            return pickle.loads(payload)
        if encoding == TRANSFORMERS_MODEL_ENCODING:
            return _deserialize_transformers_model(payload)
        raise SharedTensorSerializationError(f"Unsupported payload encoding: {encoding}")
    except SharedTensorCapabilityError:
        raise
    except Exception as exc:
        raise SharedTensorSerializationError(f"Failed to deserialize payload: {exc}") from exc


def resolve_device_index(device_index: int | None = None) -> int:
    """Resolve the CUDA device index for same-GPU socket selection."""
    if device_index is not None:
        if device_index < 0:
            raise SharedTensorConfigurationError("device_index must be >= 0")
        return device_index
    if TORCH_MODULE is None or not TORCH_MODULE.cuda.is_available():
        return 0
    return int(TORCH_MODULE.cuda.current_device())


def resolve_server_base_path(default_base_path: str = DEFAULT_SOCKET_BASE_PATH) -> str:
    """Resolve the configured UDS base path from the environment."""
    raw = os.getenv(SHARED_TENSOR_BASE_PATH_ENV)
    if raw is None or not raw.strip():
        return default_base_path
    return raw.strip()


def resolve_runtime_socket_path(
    base_path: str = DEFAULT_SOCKET_BASE_PATH,
    device_index: int | None = None,
) -> str:
    """Resolve the runtime UDS path by suffixing the CUDA device index."""
    return f"{base_path}-{resolve_device_index(device_index)}.sock"


def unlink_socket_path(socket_path: str) -> None:
    """Best-effort removal of a stale unix socket path."""
    try:
        Path(socket_path).unlink()
    except FileNotFoundError:
        return


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
