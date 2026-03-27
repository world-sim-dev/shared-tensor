from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import BertConfig, BertModel

from shared_tensor import SharedTensorClient, SharedTensorProvider, SharedTensorServer

BASE_PATH = "/tmp/shared-tensor-transformers-e2e"


def build_model() -> BertModel:
    model = BertModel(
        BertConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            vocab_size=32,
        )
    ).cuda()
    model.eval()
    return model


def run_server() -> int:
    provider = SharedTensorProvider(execution_mode="server", base_path=BASE_PATH)

    @provider.share(execution="task", cache=False, managed=False)
    def load_model() -> BertModel:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        return build_model()

    SharedTensorServer(provider).start(blocking=True)
    return 0


def run_client() -> int:
    client = SharedTensorClient(base_path=BASE_PATH, timeout=60.0)
    model = client.call("load_model")
    output = model(input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.long))
    print(
        "CLIENT_OK",
        {
            "type": type(model).__name__,
            "device": str(next(model.parameters()).device),
            "last_hidden_state_shape": list(output.last_hidden_state.shape),
        },
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
