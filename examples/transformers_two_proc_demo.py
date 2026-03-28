"""One file, two processes, same-code transformers CUDA IPC example."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from shared_tensor import SharedObjectHandle, SharedTensorProvider

provider = SharedTensorProvider(timeout=float(os.getenv("SHARED_TENSOR_TIMEOUT", "600.0")))
MODEL_ID = "bert-base-uncased"


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")


def _resolve_model_path() -> Path:
    try:
        return Path(
            snapshot_download(
                repo_id=MODEL_ID,
                local_files_only=True,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve cached Hugging Face model {MODEL_ID}. "
            "Prefetch it with `hf download bert-base-uncased` or `transformers.from_pretrained` first"
        ) from exc


def _prepare_model_path(model_path: Path) -> Path:
    resolved_model_path = model_path.resolve()
    path_str = str(resolved_model_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return resolved_model_path


def _load_model(model_path: Path):
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    model = model.cuda()
    model.eval()
    return model


def _shared_load_metrics(model_path: Path):
    before_mem_mb = _cuda_mem_mb()
    first_started_at = time.perf_counter()
    first_result = load_model(model_path=str(model_path))
    first_elapsed_ms = round((time.perf_counter() - first_started_at) * 1000, 2)
    after_first_mem_mb = _cuda_mem_mb()

    second_started_at = time.perf_counter()
    second_result = load_model(model_path=str(model_path))
    second_elapsed_ms = round((time.perf_counter() - second_started_at) * 1000, 2)
    after_second_mem_mb = _cuda_mem_mb()

    return {
        "first_result": first_result,
        "second_result": second_result,
        "first_elapsed_ms": first_elapsed_ms,
        "second_elapsed_ms": second_elapsed_ms,
        "before_mem_mb": before_mem_mb,
        "after_first_mem_mb": after_first_mem_mb,
        "after_second_mem_mb": after_second_mem_mb,
    }


def _cuda_mem_mb() -> float:
    return round(torch.cuda.memory_allocated() / 1024 / 1024, 2)


@provider.share(
    execution="task",
    managed=True,
    concurrency="serialized",
    cache=True,
    cache_format_key="transformers:{model_path}",
)
def load_model(model_path: str | None = None):
    _require_cuda()
    resolved_model_path = _prepare_model_path(Path(model_path) if model_path else _resolve_model_path())
    return _load_model(resolved_model_path)


if __name__ == "__main__":
    _require_cuda()
    resolved_model_path = _prepare_model_path(_resolve_model_path())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    metrics = _shared_load_metrics(resolved_model_path)
    result = metrics["first_result"]
    second_result = metrics["second_result"]
    after_mem_mb = metrics["after_first_mem_mb"]
    handle = result if isinstance(result, SharedObjectHandle) else None
    model = handle.value if handle is not None else result
    second_handle = second_result if isinstance(second_result, SharedObjectHandle) else None
    print(
        "LOAD_STATS",
        {
            "mode": provider.execution_mode,
            "load_model_ms": metrics["first_elapsed_ms"],
            "reload_model_ms": metrics["second_elapsed_ms"],
            "before_mem_mb": metrics["before_mem_mb"],
            "after_mem_mb": after_mem_mb,
            "after_reload_mem_mb": metrics["after_second_mem_mb"],
            "delta_mem_mb": round(after_mem_mb - metrics["before_mem_mb"], 2),
            "delta_reload_mem_mb": round(metrics["after_second_mem_mb"] - after_mem_mb, 2),
            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        },
        flush=True,
    )
    if provider.execution_mode == "client":
        print(
            "LOAD_HINT",
            {
                "meaning": "load_model_ms includes the first shared_tensor reopen in this client process; reload_model_ms shows same-process warm reopen after class resolution/import has already happened",
            },
            flush=True,
        )
    print(
        "MODEL_READY",
        {
            "mode": provider.execution_mode,
            "auto_class": AutoModel.__name__,
            "model_path": str(resolved_model_path),
            "handle": handle is not None,
            "type": type(model).__name__,
            "module": type(model).__module__,
            "device": str(next(model.parameters()).device),
        },
        flush=True,
    )
    if provider.execution_mode == "server":
        runtime = provider.get_runtime_info()
        print(
            "SERVER_READY",
            {
                "socket_path": runtime["server_socket_path"],
                "model_path": str(resolved_model_path),
            },
            flush=True,
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    if handle is not None:
        handle.release()
    if second_handle is not None:
        second_handle.release()
