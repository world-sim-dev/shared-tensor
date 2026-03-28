"""Validate that client-side parameter mutation is visible on the server."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

from shared_tensor import SharedTensorClient, SharedTensorProvider, SharedTensorServer

MODEL_ID = "bert-base-uncased"
HF_CACHE_DIR = os.getenv("HF_HUB_CACHE", "/home/niubility2/pretrained_models/huggingface/hub")
BASE_PATH = "/tmp/shared-tensor-transformers-mutation"


def _resolve_model_path() -> str:
    return snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )


def first_param_stats(model):
    param = next(model.parameters())
    flat = param.detach().view(-1)
    return {
        "device": str(param.device),
        "first": float(flat[0].item()),
        "sum16": float(flat[:16].sum().item()),
    }


def server_main(ready_q, request_q, result_q, stop_event):
    os.environ["SHARED_TENSOR_ROLE"] = "server"
    provider = SharedTensorProvider(execution_mode="server", base_path=BASE_PATH, timeout=120.0)
    model_path = _resolve_model_path()
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
    ).cuda().eval()

    @provider.share(execution="task", managed=True, concurrency="serialized", cache=True)
    def load_model():
        return model

    server = SharedTensorServer(provider)
    server.start(blocking=False)
    ready_q.put(first_param_stats(model))
    if request_q.get() == "read_stats":
        result_q.put(first_param_stats(model))
    stop_event.wait()
    server.stop(wait_for_tasks=False)


if __name__ == "__main__":
    try:
        os.remove(f"{BASE_PATH}-0.sock")
    except FileNotFoundError:
        pass
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    request_q = ctx.Queue()
    result_q = ctx.Queue()
    stop_event = ctx.Event()
    proc = ctx.Process(target=server_main, args=(ready_q, request_q, result_q, stop_event))
    proc.start()
    client = None
    handle = None
    try:
        before = ready_q.get(timeout=180)
        client = SharedTensorClient(base_path=BASE_PATH, timeout=120.0)
        handle = client.call("load_model")
        client_before = first_param_stats(handle.value)
        with torch.no_grad():
            param = next(handle.value.parameters())
            param.view(-1)[:16].add_(1.0)
        client_after = first_param_stats(handle.value)
        request_q.put("read_stats")
        server_after = result_q.get(timeout=30)
        print(
            {
                "server_before": before,
                "client_before": client_before,
                "client_after": client_after,
                "server_after": server_after,
            },
            flush=True,
        )
    finally:
        if handle is not None:
            handle.release()
        if client is not None:
            client.close()
        stop_event.set()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
