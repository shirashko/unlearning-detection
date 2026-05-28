#!/usr/bin/env python3
"""Extract optimal unlearning configurations into a nested YAML file.

Writes ``optimal_unlearning_hyperparams.yaml`` from pre-selected winner rows in
``optimal_hyperparams/*.csv``. Each slice ``<Model> -> <Method> -> <Concept>``
is split into:
  - ``hyperparameters``: values required to rerun the unlearning job
  - ``evaluation``: downstream metrics retained for analysis / auditing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Experimental configuration block
# ---------------------------------------------------------------------------
TARGET_CONCEPTS = ["Golf", "Ancient Rome", "Uranium"]
TARGET_MODELS = ["Gemma", "Llama"]
METHODS = ["PISCES", "RMU", "CRISP", "SNMF"]

METHOD_EVAL_TYPE = {
    "PISCES": "gen",
    "RMU": "mc",
    "CRISP": "mc",
    "SNMF": "mc",
}

# Structural columns from source CSVs (never emitted in either output bucket).
STRUCTURAL_COLUMNS = frozenset({"model", "concept", "method"})

# Downstream evaluation / audit metrics (not needed to rerun unlearning).
EVALUATION_COLUMNS = frozenset(
    {
        "mmlu_acc",
        "mmlu_frac",
        "mmlu_invalid",
        "qa_acc",
        "qa_frac",
        "qa_invalid",
        "forget_acc",
        "retain_acc",
        "simdom_acc",
        "simdom_frac",
        "simdom_invalid",
        "efficacy",
        "specificity",
        "harmonic",
        "alpaca_instr",
        "alp_instr_frac",
        "alpaca_flu",
        "alp_flu_frac",
        "coherence",
        "harmonic_alpaca",
        "relearning_qa_mc",
        "relearning_qa_open",
    }
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HYPERPARAMS_DIR = SCRIPT_DIR / "optimal_hyperparams"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "optimal_unlearning_hyperparams.yaml"

ROUND_DECIMALS = 3

logger = logging.getLogger(__name__)


def normalize_string(value: str) -> str:
    """Normalize concept labels for robust matching."""
    return str(value).strip().lower().replace(" pandemic", "")


def match_concepts(target: str, source: str) -> bool:
    """Return True when two concept labels refer to the same target concept."""
    left = normalize_string(target)
    right = normalize_string(source)
    return left == right or left.startswith(right) or right.startswith(left)


def concept_config_key(concept: str) -> str:
    """Return the canonical concept key used in the output YAML tree."""
    return "COVID-19" if "covid" in concept.lower() else concept


def find_row_for_concept(frame: pd.DataFrame, concept: str) -> pd.Series | None:
    """Extract the single pre-selected winner row for a target concept."""
    concept_col = next((col for col in frame.columns if col.lower() == "concept"), None)
    if concept_col is None:
        return None

    mask = frame[concept_col].astype(str).apply(lambda value: match_concepts(concept, value))
    subset = frame.loc[mask]
    
    if subset.empty:
        return None

    return subset.iloc[0]


def cast_value(value: object, *, should_round: bool = False) -> object:
    """Convert numpy/pandas scalars to native Python types for clean YAML output."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if should_round:
            rounded = round(value, ROUND_DECIMALS)
            if rounded == int(rounded):
                return int(rounded)
            return rounded
        return float(value)
    return str(value)


def split_row_payload(row: pd.Series) -> dict[str, dict[str, object]]:
    """Split one selected CSV row into rerun hyperparameters and evaluation metrics."""
    hyperparameters: dict[str, object] = {}
    evaluation: dict[str, object] = {}

    for column, value in row.to_dict().items():
        canonical = column.lower()
        if canonical in STRUCTURAL_COLUMNS:
            continue

        if canonical in EVALUATION_COLUMNS:
            evaluation[column] = cast_value(value, should_round=True)
        else:
            hyperparameters[column] = cast_value(value)

    return {
        "hyperparameters": hyperparameters,
        "evaluation": evaluation,
    }


def build_optimal_unlearning_config(
    hyperparams_dir: Path = DEFAULT_HYPERPARAMS_DIR,
    output_path: Path = DEFAULT_OUTPUT_FILE,
) -> dict[str, Any]:
    """Build and persist the nested optimal unlearning hyperparameters YAML."""
    optimal_config: dict[str, Any] = {}

    for model in TARGET_MODELS:
        optimal_config[model] = {}

        for method in METHODS:
            eval_type = METHOD_EVAL_TYPE[method]
            filename = f"{method}_{model}_{eval_type}.csv"
            file_path = hyperparams_dir / filename

            if not file_path.is_file():
                logger.warning(
                    "File %s not found. Skipping slice: %s -> %s",
                    filename,
                    model,
                    method,
                )
                continue

            frame = pd.read_csv(file_path)
            frame.columns = [col.strip() for col in frame.columns]

            method_configs: dict[str, dict[str, dict[str, object]]] = {}
            for concept in TARGET_CONCEPTS:
                row = find_row_for_concept(frame, concept)
                if row is None:
                    logger.warning(
                        "No matching concept '%s' in %s, skipping slice.",
                        concept,
                        filename,
                    )
                    continue

                method_configs[concept_config_key(concept)] = split_row_payload(row)

            if method_configs:
                optimal_config[model][method] = method_configs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(optimal_config, handle, default_flow_style=False, sort_keys=False)

    logger.info("Successfully generated optimal unlearning hyperparams at: %s", output_path)
    return optimal_config


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_optimal_unlearning_config()


if __name__ == "__main__":
    main()
