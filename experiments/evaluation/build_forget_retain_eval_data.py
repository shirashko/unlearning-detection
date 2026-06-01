#!/usr/bin/env python3
"""Build forget/retain JSON corpora for target evaluation from SNMF-Erasure data.

Output format (consumed by ``run_target_evaluation.py``)::

    {
      "forget": ["...", "..."],
      "retain": ["...", "..."]
    }

Forget sentences come from ``300_sample_sentences`` in wikipedia_sentences_samples.json
(same pool used for SNMF-Erasure unlearning training). Retain sentences come from the
shared neutral pool.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_SNMF_DATA_ROOT = Path(
    os.getenv(
        "SNMF_ERASURE_DATA_ROOT",
        "/home/morg/students/rashkovits/SNMF-Erasure/data",
    )
)

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "eval"

FORGET_POOL_KEY = "300_sample_sentences"


def concept_slug(name: str) -> str:
    """Map ``Ancient Rome`` -> ``ancient_rome`` for output filenames."""
    s = name.strip().lower().replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def normalize_concept_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def concepts_match(requested: str, candidate: str) -> bool:
    left = normalize_concept_label(requested).replace("_", " ")
    right = normalize_concept_label(candidate).replace("_", " ")
    return left == right


def resolve_data_file(data_root: Path, *relative_paths: str) -> Path:
    for rel in relative_paths:
        path = data_root / rel
        if path.is_file():
            return path
    tried = ", ".join(str(data_root / rel) for rel in relative_paths)
    raise FileNotFoundError(f"Could not find data file. Tried: {tried}")


@lru_cache(maxsize=4)
def load_json_cached(path: Path) -> Any:
    """Cache loaded JSONs to eliminate redundant disk I/O bottlenecks within loops."""
    logger.debug("Disk I/O: Reading corporate file %s", path.name)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_concept_record(
    records: Sequence[Dict[str, Any]],
    concept: str,
    *,
    concept_field: str,
) -> Dict[str, Any]:
    for record in records:
        label = record.get(concept_field)
        if label is not None and concepts_match(concept, str(label)):
            return record
    available = sorted(
        {str(record.get(concept_field, "")).strip() for record in records}
        - {""},
    )
    raise KeyError(
        f"Concept {concept!r} not found. Available options: {', '.join(available)}",
    )


def extract_forget_sentences(*, concept: str, data_root: Path) -> List[str]:
    path = resolve_data_file(
        data_root,
        "Forgetdata/wikipedia_sentences_samples.json",
        "wikipedia_sentences_samples.json",
    )
    records = load_json_cached(path)
    record = find_concept_record(records, concept, concept_field="concept")
    if FORGET_POOL_KEY not in record:
        raise KeyError(
            f"Pool {FORGET_POOL_KEY!r} missing for concept {concept!r} in {path}."
        )
    raw = record[FORGET_POOL_KEY]

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected list of sentences for {FORGET_POOL_KEY!r}, got {type(raw).__name__}.",
        )
    sentences = [str(item).strip() for item in raw if str(item).strip()]
    if not sentences:
        raise ValueError(f"No non-empty forget sentences for concept {concept!r}.")
    return sentences


def extract_retain_sentences(data_root: Path) -> List[str]:
    path = resolve_data_file(
        data_root,
        "Retaindata/neutral_sentences_300.json",
        "neutral_sentences_300.json",
    )
    raw = load_json_cached(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}, got {type(raw).__name__}.")

    sentences: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            sentences.append(item.strip())
        elif isinstance(item, dict):
            text = item.get("sentence") or item.get("text") or item.get("prompt")
            if text is not None and str(text).strip():
                sentences.append(str(text).strip())
    if not sentences:
        raise ValueError(f"No retain sentences parsed from {path}.")
    return sentences


def subsample(
    items: Sequence[str],
    max_samples: Optional[int],
    rng: random.Random,
) -> List[str]:
    if max_samples is None or max_samples >= len(items):
        return list(items)
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}.")
    return rng.sample(list(items), k=max_samples)


def build_forget_retain_payload(
    *,
    concept: str,
    data_root: Path,
    max_samples: Optional[int],
    seed: int,
) -> Dict[str, List[str]]:
    forget_all = extract_forget_sentences(concept=concept, data_root=data_root)
    retain_all = extract_retain_sentences(data_root)

    coupled_n = min(len(forget_all), len(retain_all))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}.")
        coupled_n = min(coupled_n, max_samples)
    if coupled_n <= 0:
        raise ValueError(
            f"Cannot build coupled corpus for concept {concept!r}: "
            f"forget count={len(forget_all)}, retain count={len(retain_all)}."
        )

    forget_rng = random.Random(seed)
    retain_rng = random.Random(seed)

    forget = subsample(forget_all, coupled_n, forget_rng)
    retain = subsample(retain_all, coupled_n, retain_rng)
    
    if len(forget) != len(retain):
        raise RuntimeError("Coupled baseline validation size mismatch.")
    return {"forget": forget, "retain": retain}


def list_available_concepts(data_root: Path) -> List[str]:
    path = resolve_data_file(
        data_root,
        "Forgetdata/wikipedia_sentences_samples.json",
        "wikipedia_sentences_samples.json",
    )
    records = load_json_cached(path)
    return sorted(
        {str(record["concept"]).strip() for record in records if record.get("concept")},
    )


def write_eval_file(
    payload: Dict[str, List[str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", 
        encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build forget/retain evaluation JSON from SNMF-Erasure data.",
    )
    p.add_argument(
        "--concept",
        action="append",
        default=[],
        help="Concept label. Can be repeated.",
    )
    p.add_argument(
        "--all-concepts",
        action="store_true",
        help="Build eval files for every available concept.",
    )
    p.add_argument(
        "--snmf-data-root",
        type=Path,
        default=DEFAULT_SNMF_DATA_ROOT,
        help=f"Data directory root location.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Enforce precise coupled token limit constraints.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random generation tracking seed controller.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target dump directory path.",
    )
    p.add_argument(
        "--list-concepts",
        action="store_true",
        help="Print catalog tracking details and exit.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    data_root = args.snmf_data_root.expanduser().resolve()

    if args.list_concepts:
        for name in list_available_concepts(data_root):
            print(f"{name}\t->\t{concept_slug(name)}_forget_retain.json")
        return

    if args.all_concepts:
        concepts = list_available_concepts(data_root)
    elif args.concept:
        concepts = args.concept
    else:
        build_arg_parser().error("Provide --concept or --all-concepts flags.")

    output_dir = args.output_dir.expanduser().resolve()
    for concept in concepts:
        try:
            payload = build_forget_retain_payload(
                concept=concept,
                data_root=data_root,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            slug = concept_slug(concept)
            out_path = output_dir / f"{slug}_forget_retain.json"
            write_eval_file(payload, out_path)
            logger.info(
                "Wrote %s (concept=%r, coupled_n=%d)",
                out_path.name,
                concept,
                len(payload["forget"]),
            )
        except (KeyError, ValueError, RuntimeError) as err:
            logger.error("Skipping data build for concept %r due to error: %s", concept, err)


if __name__ == "__main__":
    main()