"""Validate that client-side parameter mutation is visible on the server."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModel

from shared_tensor import SharedTensorClient, SharedTensorProvider, SharedTensorServer

MODEL_ROOT = os.environ["TRANSFORMERS_MODEL_ROOT"]
SNAPSHOTS = os.path.join(MODEL_ROOT, "snapshots")
if os.path.isdir(SNAPSHOTS):
    entries = sorted(
        os.path.join(SNAPSHOTS, name)
        for name in os.listdir(SNAPSHOTS)
        if os.path.isdir(os.path.join(SNAPSHOTS, name))
    )
    MODEL_PATH = entries[-1]
else:
    MODEL_PATH = MODEL_ROOT
BASE_PATH = "/tmp/shared-tensor-transformers-mutation"


def first_param_stats(model):
    param = next(model.parameters())
    flat = param.detach().view(-1)
    return {
        "device": str(param.device),
        "first": float(flat[0].item()),
        "sum16": float(flat[:16].sum().item()),
    }


def server_main(ready_q, command_q, result_q):
    os.environ["SHARED_TENSOR_ROLE"] = "server"
    provider = SharedTensorProvider(execution_mode="server", base_path=BASE_PATH, timeout=120.0)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
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
    try:
        while True:
            try:
                command = command_q.get(timeout=0.1)
            except queue.Empty:
                command = None
            if command == "read_stats":
                result_q.put(first_param_stats(model))
            elif command == "stop":
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        os.remove(f"{BASE_PATH}-0.sock")
    except FileNotFoundError:
        pass
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    command_q = ctx.Queue()
    result_q = ctx.Queue()
    proc = ctx.Process(target=server_main, args=(ready_q, command_q, result_q))
    proc.start()
    try:
        before = ready_q.get(timeout=180)
        client = SharedTensorClient(base_path=BASE_PATH, timeout=120.0)
        handle = client.call("load_model")
        client_before = first_param_stats(handle.value)
        with torch.no_grad():
            param = next(handle.value.parameters())
            param.view(-1)[:16].add_(1.0)
        client_after = first_param_stats(handle.value)
        command_q.put("read_stats")
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
        handle.release()
    finally:
        command_q.put("stop")
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
