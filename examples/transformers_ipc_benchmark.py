from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import BertConfig, BertModel

from shared_tensor import SharedTensorClient, SharedTensorProvider, SharedTensorServer

BASE_PATH = "/tmp/shared-tensor-transformers-bench"


def cuda_mem_mb() -> float:
    return round(torch.cuda.memory_allocated() / 1024 / 1024, 2)


def build_model() -> BertModel:
    model = BertModel(
        BertConfig(
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=16,
            num_hidden_layers=12,
            vocab_size=32000,
            max_position_embeddings=1024,
        )
    ).cuda()
    model.eval()
    return model


def run_server() -> int:
    provider = SharedTensorProvider(execution_mode="server", base_path=BASE_PATH)
    model = build_model()
    print(
        json.dumps(
            {
                "event": "server_model_ready",
                "pid": os.getpid(),
                "mem_mb": cuda_mem_mb(),
            }
        ),
        flush=True,
    )

    @provider.share(execution="task", cache=False, managed=False)
    def load_model() -> BertModel:
        return model

    SharedTensorServer(provider).start(blocking=True)
    return 0


def run_client() -> int:
    torch.cuda.empty_cache()
    before_mb = cuda_mem_mb()
    client = SharedTensorClient(base_path=BASE_PATH, timeout=120.0)
    first_started = time.perf_counter()
    model = client.call("load_model")
    first_call_ms = round((time.perf_counter() - first_started) * 1000, 2)
    after_first_load_mb = cuda_mem_mb()
    second_started = time.perf_counter()
    model_2 = client.call("load_model")
    second_call_ms = round((time.perf_counter() - second_started) * 1000, 2)
    after_second_load_mb = cuda_mem_mb()
    input_ids = torch.randint(0, 100, (1, 32), device="cuda", dtype=torch.long)
    output = model(input_ids=input_ids)
    after_forward_mb = cuda_mem_mb()
    del model_2
    print(
        json.dumps(
            {
                "event": "client_result",
                "pid": os.getpid(),
                "first_call_ms": first_call_ms,
                "second_call_ms": second_call_ms,
                "before_mb": before_mb,
                "after_first_load_mb": after_first_load_mb,
                "after_second_load_mb": after_second_load_mb,
                "after_forward_mb": after_forward_mb,
                "delta_first_load_mb": round(after_first_load_mb - before_mb, 2),
                "delta_second_load_mb": round(after_second_load_mb - after_first_load_mb, 2),
                "delta_forward_mb": round(after_forward_mb - after_second_load_mb, 2),
                "device": str(next(model.parameters()).device),
                "shape": list(output.last_hidden_state.shape),
            }
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["server", "client"])
    args = parser.parse_args()
    return run_server() if args.role == "server" else run_client()


if __name__ == "__main__":
    raise SystemExit(main())
