import json
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


def _list_hub_model_files(repo_id: str) -> frozenset[str] | None:
    """Return top-level filenames for a Hub model repo, or None if not applicable."""
    if os.path.isdir(repo_id) or os.path.isfile(repo_id):
        return None
    if "/" not in repo_id.strip("/"):
        return None
    try:
        from huggingface_hub import HfApi

        names = HfApi().list_repo_files(repo_id.strip(), repo_type="model")
        return frozenset(os.path.basename(f) for f in names)
    except Exception:
        return None


def _is_peft_adapter_only_layout(model_path: str) -> bool:
    """True when snapshot looks like LoRA/PEFT-only (adapter weights + no full config.json)."""
    if os.path.isdir(model_path):
        ad = os.path.join(model_path, "adapter_config.json")
        cf = os.path.join(model_path, "config.json")
        return os.path.isfile(ad) and not os.path.isfile(cf)
    files = _list_hub_model_files(model_path)
    if not files:
        return False
    return "adapter_config.json" in files and "config.json" not in files


def _load_peft_adapter_merged(model_path: str, device: str) -> LocalModel:
    """Load base CausalLM + merge LoRA adapter into dense weights."""
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            "PEFT adapter checkpoint requires the `peft` package "
            "(pip install peft). See requirements.txt."
        ) from e

    from huggingface_hub import hf_hub_download

    print(f"Loading PEFT adapter layout from {model_path}...")
    if os.path.isdir(model_path):
        adapter_root = model_path
        with open(os.path.join(model_path, "adapter_config.json"), encoding="utf-8") as f:
            adapter_cfg = json.load(f)
    else:
        ac_path = hf_hub_download(repo_id=model_path.strip(), filename="adapter_config.json")
        with open(ac_path, encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        adapter_root = model_path.strip()

    base_path = (
        os.environ.get("PEFT_BASE_MODEL_OVERRIDE", "").strip()
        or adapter_cfg.get("base_model_name_or_path")
    )
    if not base_path:
        raise ValueError(
            f"adapter_config.json for {model_path!r} has no base_model_name_or_path "
            "and PEFT_BASE_MODEL_OVERRIDE is unset."
        )
    if os.environ.get("PEFT_BASE_MODEL_OVERRIDE", "").strip():
        print(f"  PEFT_BASE_MODEL_OVERRIDE={base_path!r}")
    else:
        print(f"  base_model_name_or_path from adapter: {base_path!r}")

    config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        attn_implementation="eager",
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_root)
    model = model.merge_and_unload()
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_root,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cfg = model.config
    print(
        f"Model loaded (merged PEFT): {cfg.num_hidden_layers} layers, "
        f"d_model={cfg.hidden_size}, d_mlp={cfg.intermediate_size}"
    )

    return LocalModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        n_layers=cfg.num_hidden_layers,
        d_model=cfg.hidden_size,
        d_mlp=cfg.intermediate_size,
    )


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
    if _is_peft_adapter_only_layout(model_path):
        return _load_peft_adapter_merged(model_path, device=device)

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
        d_mlp=config.intermediate_size
    )
