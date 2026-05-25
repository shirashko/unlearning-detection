import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, PreTrainedModel
from dataclasses import dataclass


@dataclass
class LocalModel:
    model: PreTrainedModel
    tokenizer: AutoTokenizer
    config: AutoConfig
    device: str
    n_layers: int
    d_model: int
    d_mlp: int


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

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
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
