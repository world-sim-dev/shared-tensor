from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModel

from shared_tensor import SharedTensorClient, SharedTensorProvider, SharedTensorServer
import shared_tensor.utils as utils

BASE_PATH = "/tmp/shared-tensor-trace"
MODEL_ROOT = Path("/home/niubility2/pretrained_models/huggingface/hub/models--bert-base-uncased")
SNAPSHOTS = MODEL_ROOT / "snapshots"
MODEL_PATH = str(sorted([p for p in SNAPSHOTS.iterdir() if p.is_dir()])[-1] if SNAPSHOTS.is_dir() else MODEL_ROOT)


def _patch_server() -> None:
    original = utils._serialize_transformers_model

    def wrapped(module):
        started = time.perf_counter()
        payload = original(module)
        elapsed = time.perf_counter() - started
        encoded = pickle.loads(payload)
        print(
            {
                "event": "server_serialize",
                "sec": round(elapsed, 3),
                "payload_kb": round(len(payload) / 1024, 3),
                "state_kb": round(len(encoded["state_bytes"]) / 1024, 3),
                "config_kb": round(len(encoded["config_bytes"]) / 1024, 3),
            },
            flush=True,
        )
        return payload

    utils._serialize_transformers_model = wrapped


def _patch_client() -> None:
    original = utils._deserialize_transformers_model

    def wrapped(payload):
        started = time.perf_counter()
        value = original(payload)
        elapsed = time.perf_counter() - started
        print(
            {
                "event": "client_deserialize",
                "sec": round(elapsed, 3),
                "type": type(value).__name__,
            },
            flush=True,
        )
        return value

    utils._deserialize_transformers_model = wrapped


def run_server() -> int:
    _patch_server()
    provider = SharedTensorProvider(execution_mode="server", base_path=BASE_PATH, timeout=120.0)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
    ).cuda().eval()

    @provider.share(
        execution="task",
        managed=True,
        concurrency="serialized",
        cache=True,
        cache_format_key="transformers:{model_path}",
    )
    def load_model(model_path: str | None = None):
        return model

    load_model(model_path=MODEL_PATH)
    SharedTensorServer(provider).start(blocking=True)
    return 0


def run_client() -> int:
    _patch_client()
    client = SharedTensorClient(base_path=BASE_PATH, timeout=120.0)
    started = time.perf_counter()
    obj = client.call("load_model", model_path=MODEL_PATH)
    elapsed = time.perf_counter() - started
    print(
        {
            "event": "client_call",
            "sec": round(elapsed, 3),
            "type": type(getattr(obj, "value", obj)).__name__,
            "cuda_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 3),
        },
        flush=True,
    )
    if hasattr(obj, "release"):
        obj.release()
    return 0


if __name__ == "__main__":
    role = sys.argv[1]
    raise SystemExit(run_server() if role == "server" else run_client())
