import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# Tokenizer files copied when normalizing v5-style list extra_special_tokens for v4.x.
_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
)


@dataclass
class LocalModel:
    model: PreTrainedModel
    tokenizer: AutoTokenizer
    config: AutoConfig
    device: str
    n_layers: int
    d_model: int
    d_mlp: int


def _needs_extra_special_tokens_v4_patch(model_path: str) -> bool:
    """Return True when tokenizer_config uses list-style extra_special_tokens."""
    cfg_path = Path(model_path) / "tokenizer_config.json"
    if not cfg_path.is_file():
        return False
    with cfg_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return isinstance(data.get("extra_special_tokens"), list)


@contextmanager
def _tokenizer_pretrained_path(model_path: str):
    """Yield a tokenizer path compatible with transformers 4.x loading."""
    root = Path(model_path)
    if not _needs_extra_special_tokens_v4_patch(model_path):
        yield model_path
        return

    with tempfile.TemporaryDirectory(prefix="hf_tok_compat_") as tmp:
        dest = Path(tmp)
        for name in _TOKENIZER_ARTIFACTS:
            src = root / name
            if src.is_file():
                shutil.copy2(src, dest / name)

        cfg_path = dest / "tokenizer_config.json"
        with cfg_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        data["extra_special_tokens"] = {
            token: token for token in data["extra_special_tokens"]
        }
        with cfg_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        yield str(dest)


def load_local_model(model_path: str, device: str = "cpu") -> LocalModel:
    print(f"Loading model from {model_path}...")
    # Fail fast when the user clearly meant a *local* directory but it is missing.
    # Hugging Face repo ids also contain "/", so do not gate on slash alone — only
    # reject missing dirs for absolute paths and explicit ./ ../ relative roots.
    is_explicit_local_root = (
        os.path.isabs(model_path)
        or model_path.startswith("./")
        or model_path.startswith("../")
    )
    if is_explicit_local_root and not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Local model directory does not exist: {model_path!r}. "
            "Check MODEL_PATH / the path argument (a stale exported MODEL_PATH "
            "in the submitting shell is a common cause)."
        )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval().to(device)

    with _tokenizer_pretrained_path(model_path) as tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded: {config.num_hidden_layers} layers, "
          f"d_model={config.hidden_size}, d_mlp={config.intermediate_size}")

    return LocalModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        n_layers=config.num_hidden_layers,
        d_model=config.hidden_size,
        d_mlp=config.intermediate_size,
    )
