from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Type, TypeVar, Union, get_args
import yaml

from experiments.train.train import parse_int_list

DEFAULT_CONTEXT_RARE_ZIPF_CUTOFF: float = 5.5

RankBy = Literal["rel_delta", "abs_rel_delta"]

"""
mlp_intermediate corresponds to the matrix A (n_tokens, d_mlp), representing the MLP activations 
post-activation function (the input to down_proj), where d_mlp corresponds to the model hidden state size.
The SNMF basis Z (d_mlp, K), where K is the factorization rank.
"""
ActivationMode = Literal["mlp_intermediate"]

def _rank_by_choices() -> tuple[str, ...]: return get_args(RankBy)
def _mode_choices() -> tuple[str, ...]: return get_args(ActivationMode)

T = TypeVar("T")

def _expand_path_fields(s: str) -> str:
    """Expand ${VAR} / ~ so YAML can use env vars set by audit_runner_env.sh."""
    if not s:
        return s
    return os.path.expandvars(os.path.expanduser(str(s)))


def _pick_dataclass_kwargs(dc_type: Type[T], data: Dict[str, Any]) -> Dict[str, Any]:
    names = {f.name for f in fields(dc_type)}
    return {k: v for k, v in data.items() if k in names and v is not None}


def _reject_unknown_nested_keys(dc_type: Type[Any], patch: Dict[str, Any]) -> None:
    """Typos in nested YAML sections must fail fast (root-level unknown keys already do)."""
    allowed = {f.name for f in fields(dc_type)}
    unknown = sorted(set(patch.keys()) - allowed)
    if unknown:
        sec = getattr(dc_type, "__name__", str(dc_type))
        raise ValueError(f"Unknown keys in {sec} config section: {unknown}")


def _merge_dataclass(dc_type: Type[T], defaults: T, patch: Optional[Dict[str, Any]]) -> T:
    if patch is None:
        return defaults
    if not isinstance(patch, dict):
        raise ValueError(
            f"Expected a mapping for {getattr(dc_type, '__name__', dc_type)}, got {type(patch).__name__}"
        )
    _reject_unknown_nested_keys(dc_type, patch)
    return replace(defaults, **_pick_dataclass_kwargs(dc_type, patch))


@dataclass
class AuditRuntimeConfig:
    max_prompts: int = 400
    batch_size: int = 8
    device: str = "auto"
    seed: int = 42

@dataclass
class SNMFConfig:
    mode: ActivationMode = "mlp_intermediate"
    ridge_lambda: float = 1e-4
    top_k_global: int = 20
    top_k_per_layer: int = 15
    rank_by: RankBy = "rel_delta"
    contexts_per_feature: int = 8
    context_window: int = 15

@dataclass
class LogitLensConfig:
    vocab_lens_top_k: int = 15
    skip_vocab_lens: bool = False
    lens_center_unembed: bool = True
    lens_mask_special_tokens: bool = True
    vocab_lens_aggregate_top_k: int = 20
    lens_delta_weighted: bool = False

@dataclass
class ContextRareConfig:
    context_rare_top_n: int = 15
    skip_context_rare_words: bool = False
    context_rare_zipf_cutoff: float = DEFAULT_CONTEXT_RARE_ZIPF_CUTOFF
    context_rare_min_len: int = 3

@dataclass
class JudgeConfig:
    judge_model: str = "gemini-2.5-flash"
    judge_temperature: float = 0.0
    judge_max_output_tokens: int = 8192
    skip_judge: bool = False
    judge_api_key_env: str = "GOOGLE_API_KEY"

LayersSpec = Optional[Union[str, List[int]]]

@dataclass
class AuditConfig:
    base_model_path: str = ""
    candidate_model_path: str = ""
    snmf_dir: str = ""
    data_path: str = ""
    output_dir: str = ""
    layers: LayersSpec = None

    runtime: AuditRuntimeConfig = field(default_factory=AuditRuntimeConfig)
    snmf: SNMFConfig = field(default_factory=SNMFConfig)
    lens: LogitLensConfig = field(default_factory=LogitLensConfig)
    rare: ContextRareConfig = field(default_factory=ContextRareConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def layers_wanted(self) -> Optional[Set[int]]:
        spec = self.layers
        if spec is None:
            return None
        if isinstance(spec, str):
            return set(parse_int_list(spec))
        return {int(x) for x in spec}

_NESTED_SECTION_TYPES: Dict[str, Type[Any]] = {
    "runtime": AuditRuntimeConfig,
    "snmf": SNMFConfig,
    "lens": LogitLensConfig,
    "rare": ContextRareConfig,
    "judge": JudgeConfig,
}

_AUDIT_ROOT_SCALAR_FIELDS: FrozenSet[str] = frozenset(
    f.name for f in fields(AuditConfig) if f.name not in _NESTED_SECTION_TYPES
)

def apply_argparse_namespace_overrides(
    cfg: AuditConfig,
    ns: argparse.Namespace,
    *,
    skip_names: FrozenSet[str] = frozenset({"config"}),
) -> None:
    """Apply CLI overrides: ``vars(ns)`` keys must match dataclass field names (argparse ``dest``).

    Top-level ``AuditConfig`` fields and each nested section (``runtime``, ``snmf``, …) are
    updated in place. Add a new flag by (1) defining the field on the right dataclass and
    (2) adding ``p.add_argument(..., dest=<field_name>, default=SUPPRESS)`` — no edits here.
    """
    raw = vars(ns)
    for name in _AUDIT_ROOT_SCALAR_FIELDS:
        if name in skip_names or name not in raw:
            continue
        setattr(cfg, name, raw[name])

    for attr, dc_type in _NESTED_SECTION_TYPES.items():
        patch = {fn: raw[fn] for fn in (f.name for f in fields(dc_type)) if fn in raw}
        if patch:
            setattr(cfg, attr, replace(getattr(cfg, attr), **patch))

def audit_config_from_dict(raw: Optional[Dict[str, Any]]) -> AuditConfig:
    if not raw:
        return AuditConfig()
    root = dict(raw)

    cfg = AuditConfig(
        base_model_path=_expand_path_fields(str(root.pop("base_model_path", "") or "")),
        candidate_model_path=_expand_path_fields(str(root.pop("candidate_model_path", "") or "")),
        snmf_dir=_expand_path_fields(str(root.pop("snmf_dir", "") or "")),
        data_path=_expand_path_fields(str(root.pop("data_path", "") or "")),
        output_dir=_expand_path_fields(str(root.pop("output_dir", "") or "")),
        layers=root.pop("layers", None),
    )

    # Resolve direct section mappings
    cfg.runtime = _merge_dataclass(AuditRuntimeConfig, cfg.runtime, root.pop("runtime", None))
    cfg.snmf = _merge_dataclass(SNMFConfig, cfg.snmf, root.pop("snmf", None))
    cfg.lens = _merge_dataclass(LogitLensConfig, cfg.lens, root.pop("lens", None))
    cfg.rare = _merge_dataclass(ContextRareConfig, cfg.rare, root.pop("rare", None))
    cfg.judge = _merge_dataclass(JudgeConfig, cfg.judge, root.pop("judge", None))

    # Clean up flat keys that belong to nested dataclasses
    for attr, dc_type in _NESTED_SECTION_TYPES.items():
        patch = _pick_dataclass_kwargs(dc_type, root)
        if patch:
            setattr(cfg, attr, replace(getattr(cfg, attr), **patch))
            for k in patch:
                root.pop(k)

    if root:
        raise ValueError(f"Unknown config keys: {sorted(root.keys())}")

    return cfg

def load_audit_config_yaml(path: Union[str, Path]) -> AuditConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return AuditConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}")
    return audit_config_from_dict(raw)

def _is_explicit_local_model_path(model_path: str) -> bool:
    s = model_path.strip()
    return Path(s).is_absolute() or s.startswith("./") or s.startswith("../")

def _require_local_model_dir(label: str, model_path: str) -> None:
    if not _is_explicit_local_model_path(model_path):
        return
    p = Path(model_path).expanduser()
    if not p.is_dir():
        raise ValueError(f"{label} must be an existing local directory for path {model_path!r}")

def validate_audit_config(cfg: AuditConfig) -> None:
    missing = [
        name for name, val in (
            ("base_model_path", cfg.base_model_path),
            ("candidate_model_path", cfg.candidate_model_path),
            ("snmf_dir", cfg.snmf_dir),
            ("data_path", cfg.data_path),
            ("output_dir", cfg.output_dir),
        ) if not str(val).strip()
    ]
    if missing:
        raise ValueError("Missing required config fields: " + ", ".join(missing))
    if cfg.snmf.rank_by not in _rank_by_choices():
        raise ValueError(f"Invalid rank_by={cfg.snmf.rank_by!r}; expected one of {_rank_by_choices()}")
    if cfg.snmf.mode not in _mode_choices():
        raise ValueError(f"Invalid mode={cfg.snmf.mode!r}; expected one of {_mode_choices()}")

    out_dir = Path(cfg.output_dir).expanduser()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Cannot create output_dir {cfg.output_dir!r}: {e}") from e

    snmf_dir = Path(cfg.snmf_dir).expanduser()
    if not snmf_dir.is_dir():
        raise ValueError(f"snmf_dir is not an existing directory: {cfg.snmf_dir!r}")

    data_file = Path(cfg.data_path).expanduser()
    if not data_file.is_file():
        raise ValueError(f"data_path is not an existing file: {cfg.data_path!r}")

    _require_local_model_dir("base_model_path", cfg.base_model_path)
    _require_local_model_dir("candidate_model_path", cfg.candidate_model_path)

def audit_config_to_nested_dict(cfg: AuditConfig) -> Dict[str, Any]:
    def _conv(obj: Any) -> Any:
        if is_dataclass(obj):
            return {k: _conv(v) for k, v in asdict(obj).items()}
        return obj
    return {
        "base_model_path": cfg.base_model_path,
        "candidate_model_path": cfg.candidate_model_path,
        "snmf_dir": cfg.snmf_dir,
        "data_path": cfg.data_path,
        "output_dir": cfg.output_dir,
        "layers": cfg.layers,
        "runtime": _conv(cfg.runtime),
        "snmf": _conv(cfg.snmf),
        "lens": _conv(cfg.lens),
        "rare": _conv(cfg.rare),
        "judge": _conv(cfg.judge),
    }

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Label-free SNMF unlearning audit with LLM-judge verdict.")
    p.add_argument("--config", type=str, default=None, help="YAML file with audit settings.")
    p.add_argument("--base-model-path", type=str, default=argparse.SUPPRESS)
    p.add_argument("--candidate-model-path", type=str, default=argparse.SUPPRESS)
    p.add_argument("--snmf-dir", type=str, default=argparse.SUPPRESS)
    p.add_argument("--data-path", type=str, default=argparse.SUPPRESS)
    p.add_argument("--output-dir", type=str, default=argparse.SUPPRESS)
    p.add_argument("--layers", type=str, default=argparse.SUPPRESS)

    # Runtime / SNMF
    p.add_argument("--mode", type=str, default=argparse.SUPPRESS, choices=list(_mode_choices()))
    p.add_argument("--max-prompts", type=int, default=argparse.SUPPRESS)
    p.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--device", type=str, default=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--ridge-lambda", type=float, default=argparse.SUPPRESS)
    p.add_argument("--top-k-global", type=int, default=argparse.SUPPRESS)
    p.add_argument("--top-k-per-layer", type=int, default=argparse.SUPPRESS)
    p.add_argument("--rank-by", type=str, default=argparse.SUPPRESS, choices=list(_rank_by_choices()))
    p.add_argument("--contexts-per-feature", type=int, default=argparse.SUPPRESS)
    p.add_argument("--context-window", type=int, default=argparse.SUPPRESS)
    
    # Logit Lens Config
    p.add_argument("--vocab-lens-top-k", type=int, default=argparse.SUPPRESS)
    p.add_argument("--skip-vocab-lens", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    p.add_argument("--lens-center-unembed", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    p.add_argument("--lens-mask-special-tokens", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    p.add_argument("--vocab-lens-aggregate-top-k", type=int, default=argparse.SUPPRESS)
    p.add_argument("--lens-delta-weighted", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    # Rare Context Words
    p.add_argument("--context-rare-top-n", type=int, default=argparse.SUPPRESS)
    p.add_argument("--context-rare-zipf-cutoff", type=float, default=argparse.SUPPRESS)
    p.add_argument("--context-rare-min-len", type=int, default=argparse.SUPPRESS)
    p.add_argument("--skip-context-rare-words", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)

    # LLM Judge
    p.add_argument("--judge-model", type=str, default=argparse.SUPPRESS)
    p.add_argument("--judge-temperature", type=float, default=argparse.SUPPRESS)
    p.add_argument("--judge-max-output-tokens", type=int, default=argparse.SUPPRESS)
    p.add_argument("--skip-judge", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    p.add_argument("--judge-api-key-env", type=str, default=argparse.SUPPRESS)
    return p

def parse_args_to_config(argv: Optional[List[str]] = None) -> AuditConfig:
    p = build_arg_parser()
    ns = p.parse_args(argv)
    cfg = load_audit_config_yaml(ns.config) if ns.config else AuditConfig()
    apply_argparse_namespace_overrides(cfg, ns)
    try:
        validate_audit_config(cfg)
    except ValueError as e:
        p.error(str(e))
    return cfg