from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import torch
from safetensors.torch import load_file
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerConfig

from shared_tensor import SharedTensorClient, SharedTensorProvider, SharedTensorServer

MODEL_PATH = "/home/niubility2/pretrained_models/huggingface/hub/models--ASLP-lab--SongFormer/snapshots/5ac5227fccf286519464fdf211e15b606898408e"
BASE_PATH = "/tmp/shared-tensor-songformer-e2e"


def configure_songformer_env() -> None:
    os.environ["SONGFORMER_LOCAL_DIR"] = MODEL_PATH
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if current_pythonpath:
        os.environ["PYTHONPATH"] = f"{MODEL_PATH}:{current_pythonpath}"
    else:
        os.environ["PYTHONPATH"] = MODEL_PATH
    if MODEL_PATH not in sys.path:
        sys.path.insert(0, MODEL_PATH)


def patch_musicfm_offline_config() -> None:
    original_from_pretrained = Wav2Vec2ConformerConfig.from_pretrained

    def offline_from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        if pretrained_model_name_or_path == "facebook/wav2vec2-conformer-rope-large-960h-ft":
            return cls(
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                position_embeddings_type="rotary",
            )
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    Wav2Vec2ConformerConfig.from_pretrained = classmethod(offline_from_pretrained)


def load_songformer_local() -> torch.nn.Module:
    configure_songformer_env()
    patch_musicfm_offline_config()

    from configuration_songformer import SongFormerConfig
    from modeling_songformer import SongFormerModel

    config = SongFormerConfig.from_pretrained(MODEL_PATH)
    model = SongFormerModel(config)
    state = load_file(f"{MODEL_PATH}/model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"SongFormer state load mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    return model


def wait_for_socket(socket_path: str, timeout: float = 60.0) -> None:
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


def build_provider() -> SharedTensorProvider:
    provider = SharedTensorProvider(
        execution_mode="server",
        base_path=BASE_PATH,
        verbose_debug=True,
    )

    @provider.share(execution="task", cache=False, managed=False, concurrency="serialized")
    def load_songformer() -> torch.nn.Module:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        model = load_songformer_local()
        model = model.cuda()
        model.eval()
        return model

    return provider


def run_server() -> int:
    configure_songformer_env()
    provider = build_provider()
    server = SharedTensorServer(provider)
    print(f"SERVER_READY socket={server.socket_path}", flush=True)
    try:
        server.start(blocking=True)
    except KeyboardInterrupt:
        print("SERVER_STOPPED", flush=True)
    return 0


def run_client() -> int:
    configure_songformer_env()
    patch_musicfm_offline_config()
    client = SharedTensorClient(base_path=BASE_PATH, timeout=180.0, verbose_debug=True)
    wait_for_socket(client.socket_path)
    model = client.call("load_songformer")
    first_param = next(model.parameters())
    print(
        "CLIENT_OK",
        {
            "type": type(model).__name__,
            "module": type(model).__module__,
            "device": str(first_param.device),
            "param_count": sum(param.numel() for param in model.parameters()),
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
