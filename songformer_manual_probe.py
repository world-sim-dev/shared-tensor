from __future__ import annotations

import os
import sys

from safetensors.torch import load_file
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerConfig

MODEL_PATH = "/home/niubility2/pretrained_models/huggingface/hub/models--ASLP-lab--SongFormer/snapshots/5ac5227fccf286519464fdf211e15b606898408e"


def main() -> int:
    os.environ["SONGFORMER_LOCAL_DIR"] = MODEL_PATH
    if MODEL_PATH not in sys.path:
        sys.path.insert(0, MODEL_PATH)

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

    from configuration_songformer import SongFormerConfig
    from modeling_songformer import SongFormerModel

    config = SongFormerConfig.from_pretrained(MODEL_PATH)
    model = SongFormerModel(config)
    state = load_file(f"{MODEL_PATH}/model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)

    first_param = next(model.parameters())
    print("STATE_KEYS", len(state))
    print("MISSING", len(missing))
    print("MISSING_SAMPLE", missing[:20])
    print("UNEXPECTED", len(unexpected))
    print("UNEXPECTED_SAMPLE", unexpected[:20])
    print("FIRST_PARAM", str(first_param.device), str(first_param.dtype), tuple(first_param.shape))
    print("META_COUNT", sum(1 for _, p in model.named_parameters() if getattr(p, "is_meta", False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
