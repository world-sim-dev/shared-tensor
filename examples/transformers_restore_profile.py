from __future__ import annotations

import io
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

from shared_tensor.utils import _torch_forking_pickler


MODEL_ID = "bert-base-uncased"
HF_CACHE_DIR = os.getenv("HF_HUB_CACHE", "/home/niubility2/pretrained_models/huggingface/hub")


def _torch_serialize(obj):
    pickler = _torch_forking_pickler()
    if pickler is None:
        raise RuntimeError("Torch IPC pickler is unavailable")
    buffer = io.BytesIO()
    pickler(buffer, pickle.HIGHEST_PROTOCOL).dump(obj)
    return buffer.getvalue()


def _resolve_parent_module(root, parent_name: str):
    if not parent_name:
        return root
    current = root
    for part in parent_name.split("."):
        current = getattr(current, part)
    return current


def _rebind_named_parameter(module, name: str, value) -> None:
    parent_name, _, local_name = name.rpartition(".")
    parent = _resolve_parent_module(module, parent_name)
    parent._parameters[local_name] = value


def _rebind_named_buffer(module, name: str, value) -> None:
    parent_name, _, local_name = name.rpartition(".")
    parent = _resolve_parent_module(module, parent_name)
    parent._buffers[local_name] = value


def _producer(queue, model_path: str) -> None:
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
    ).cuda().eval()
    legacy_payload = pickle.dumps(
        {"module_bytes": _torch_serialize(model)},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    state_payload = pickle.dumps(
        {
            "config_bytes": pickle.dumps(model.config, protocol=pickle.HIGHEST_PROTOCOL),
            "model_module": type(model).__module__,
            "model_qualname": type(model).__qualname__,
            "state_bytes": _torch_serialize(
                {
                    "parameters": dict(model.named_parameters(remove_duplicate=False)),
                    "buffers": dict(model.named_buffers(remove_duplicate=False)),
                }
            ),
            "training": bool(model.training),
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    queue.put((legacy_payload, state_payload))
    time.sleep(60)


def _resolve_model_path() -> Path:
    return Path(
        snapshot_download(
            repo_id=MODEL_ID,
            local_files_only=True,
        )
    )


def main() -> None:
    model_path = str(_resolve_model_path())
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=_producer, args=(queue, model_path))
    process.start()
    legacy_payload, state_payload = queue.get(timeout=180)
    print(
        {
            "event": "payload_sizes_mb",
            "legacy": round(len(legacy_payload) / 1024 / 1024, 3),
            "state": round(len(state_payload) / 1024 / 1024, 3),
        },
        flush=True,
    )

    started = time.perf_counter()
    legacy_outer = pickle.loads(legacy_payload)
    after_outer = time.perf_counter()
    legacy_model = pickle.loads(legacy_outer["module_bytes"])
    after_legacy = time.perf_counter()
    print(
        {
            "event": "legacy",
            "outer_sec": round(after_outer - started, 3),
            "module_unpickle_sec": round(after_legacy - after_outer, 3),
            "total_sec": round(after_legacy - started, 3),
            "type": type(legacy_model).__name__,
        },
        flush=True,
    )

    state_started = time.perf_counter()
    encoded = pickle.loads(state_payload)
    after_state_outer = time.perf_counter()
    config = pickle.loads(encoded["config_bytes"])
    after_config = time.perf_counter()
    module = __import__(encoded["model_module"], fromlist=["_dummy"])
    model_cls = module
    for part in encoded["model_qualname"].split("."):
        model_cls = getattr(model_cls, part)
    with torch.device("meta"):
        shell = model_cls(config)
    after_meta_shell = time.perf_counter()
    state = pickle.loads(encoded["state_bytes"])
    after_state_unpickle = time.perf_counter()
    for name, value in state["parameters"].items():
        _rebind_named_parameter(shell, name, value)
    for name, value in state["buffers"].items():
        _rebind_named_buffer(shell, name, value)
    shell.eval()
    after_rebind = time.perf_counter()
    print(
        {
            "event": "state_path",
            "outer_sec": round(after_state_outer - state_started, 3),
            "config_sec": round(after_config - after_state_outer, 3),
            "meta_shell_sec": round(after_meta_shell - after_config, 3),
            "state_unpickle_sec": round(after_state_unpickle - after_meta_shell, 3),
            "rebind_sec": round(after_rebind - after_state_unpickle, 3),
            "total_sec": round(after_rebind - state_started, 3),
            "type": type(shell).__name__,
        },
        flush=True,
    )

    process.terminate()
    process.join(timeout=5)


if __name__ == "__main__":
    main()
