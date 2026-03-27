from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import BertConfig, AutoModel

from shared_tensor import SharedTensorClient, SharedTensorProvider, SharedTensorServer

BASE_PATH = "/tmp/shared-tensor-transformers-e2e"


def wait_for_socket(socket_path: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                sock.connect(socket_path)
            return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for server socket {socket_path}")


def build_model() -> torch.nn.Module:
    config = BertConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
    )
    model = AutoModel.from_config(config).cuda()
    model.eval()
    return model


def build_provider() -> SharedTensorProvider:
    provider = SharedTensorProvider(
        execution_mode="server",
        base_path=BASE_PATH,
        verbose_debug=True,
    )

    @provider.share(execution="task", cache=False, managed=False)
    def load_transformers_model() -> torch.nn.Module:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        return build_model()

    return provider


def run_server() -> int:
    provider = build_provider()
    server = SharedTensorServer(provider)
    print(f"SERVER_READY socket={server.socket_path}", flush=True)
    try:
        server.start(blocking=True)
    except KeyboardInterrupt:
        print("SERVER_STOPPED", flush=True)
    return 0


def run_client() -> int:
    client = SharedTensorClient(base_path=BASE_PATH, timeout=60.0, verbose_debug=True)
    wait_for_socket(client.socket_path)
    model = client.call("load_transformers_model")
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda", dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    first_param = next(model.parameters())
    print(
        "CLIENT_OK",
        {
            "type": type(model).__name__,
            "module": type(model).__module__,
            "device": str(first_param.device),
            "last_hidden_state_shape": list(output.last_hidden_state.shape),
            "training": model.training,
        },
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["server", "client"])
    args = parser.parse_args()
    if args.role == "server":
        return run_server()
    return run_client()


if __name__ == "__main__":
    raise SystemExit(main())
