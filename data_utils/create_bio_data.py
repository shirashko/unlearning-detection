import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DATASET_PATH = "/home/morg/students/rashkovits/Localized-UNDO/datasets"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create labeled bio dataset JSON file(s) with bio_forget / bio_retain / neutral. "
            "Use --num-files > 1 to write several disjoint batches of --samples-per-label "
            "per label (clipped if sources are too small)."
        )
    )
    parser.add_argument(
        "--remove-path",
        type=Path,
        default=Path(
            f"{BASE_DATASET_PATH}/wmdp/qa/wmdp-bio_remove_dataset-combined.jsonl"
        ),
        help="Path to remove split JSONL (mapped to label 'bio_forget').",
    )
    parser.add_argument(
        "--retain-path",
        type=Path,
        default=Path(
            f"{BASE_DATASET_PATH}/wmdp/qa/wmdp-bio_retain_dataset-combined.jsonl"
        ),
        help="Path to retain split JSONL (mapped to label 'bio_retain').",
    )
    parser.add_argument(
        "--neutral-path",
        type=Path,
        default=Path(f"{BASE_DATASET_PATH}/pretrain/train_eng.jsonl"),
        help="Path to neutral JSONL (mapped to label 'neutral').",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/bio_data.json"),
        help=(
            "Output path when --num-files=1. When --num-files>1, writes "
            "<stem>_partNNN<suffix> in the same directory (NNN zero-padded)."
        ),
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=1,
        help=(
            "How many JSON files to create. Each file has --samples-per-label rows per label. "
            "Batches are disjoint (no row reused across files). If you ask for more files than "
            "the smallest source allows, the count is clipped and a warning is printed."
        ),
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=400,
        help="How many samples per label in each output file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Keep only the first N whitespace tokens from each sampled text (0 disables truncation).",
    )
    return parser.parse_args()


def load_texts(jsonl_path: Path) -> List[str]:
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = (
                record.get("qa", {}).get("question")
                if isinstance(record.get("qa"), dict)
                else None
            )
            if not isinstance(text, str) or not text.strip():
                text = record.get("question")
            if not isinstance(text, str) or not text.strip():
                text = record.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def load_qa_rows(jsonl_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qa = record.get("qa", {})
            if not isinstance(qa, dict):
                continue

            question = qa.get("question")
            answer = qa.get("answer")
            question_text = question.strip() if isinstance(question, str) else ""
            answer_text = answer.strip() if isinstance(answer, str) else ""
            if question_text and answer_text:
                rows.append((question_text, answer_text))
    return rows


def truncate_to_first_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def _qa_and_answer_counts(k: int) -> Tuple[int, int]:
    """Half (rounded) from question+answer strings, remainder from answer-only pool."""
    qa_k = k // 2
    answer_only_k = k - qa_k
    return qa_k, answer_only_k


def _batches_available(pool: int, need: int) -> int:
    """How many disjoint draws of `need` rows from `pool` (need==0 → no constraint)."""
    if need < 0:
        return 0
    if need == 0:
        return 10**18
    return pool // need


def _max_disjoint_files(k: int, n_forget_rows: int, n_retain_rows: int, n_neutral: int) -> int:
    """How many files of size k per label can be built without reusing source rows."""
    if k < 1:
        return 0
    if n_forget_rows < k:
        return 0
    if n_retain_rows < k:
        return 0
    if n_neutral < k:
        return 0
    m_forget = n_forget_rows // k
    m_retain = n_retain_rows // k
    m_neutral = n_neutral // k
    return min(m_forget, m_retain, m_neutral)


def part_output_paths(output_path: Path, num_parts: int) -> List[Path]:
    if num_parts == 1:
        return [output_path]
    parent = output_path.parent
    stem = output_path.stem
    suffix = output_path.suffix
    return [parent / f"{stem}_part{idx + 1}{suffix}" for idx in range(num_parts)]


def _pct_str(used: int, total: int) -> str:
    if total <= 0:
        return f"{used}/{total} (n/a)"
    return f"{used}/{total} ({100.0 * used / total:.2f}%)"


def main() -> None:
    args = parse_args()
    print("Starting bio data creation...")
    if args.num_files < 1:
        raise SystemExit("--num-files must be >= 1")
    k = args.samples_per_label
    if k < 1:
        raise SystemExit("--samples-per-label must be >= 1")

    forget_rows = load_qa_rows(args.remove_path)
    retain_rows = load_qa_rows(args.retain_path)
    neutral_texts = load_texts(args.neutral_path)

    qa_k, ans_k = _qa_and_answer_counts(k)
    max_files = _max_disjoint_files(
        k,
        len(forget_rows),
        len(retain_rows),
        len(neutral_texts),
    )
    if max_files == 0:
        raise ValueError(
            f"Sources are too small for one file of {k} samples per label "
            f"(need forget: ≥{k} QA rows with both question+answer, "
            f"retain: ≥{k} QA rows, and ≥{k} neutral rows)."
        )

    n_files = min(args.num_files, max_files)
    if n_files < args.num_files:
        print(
            f"WARNING: requested {args.num_files} files but only {max_files} disjoint batches "
            f"of {k} per label are possible; writing {n_files} file(s)."
        )

    rng = random.Random(args.seed)
    rng.shuffle(forget_rows)
    rng.shuffle(retain_rows)
    rng.shuffle(neutral_texts)

    out_paths = part_output_paths(args.output_path, n_files)
    for i, out_path in enumerate(out_paths):
        forget_chunk = forget_rows[i * k : (i + 1) * k]
        retain_chunk = retain_rows[i * k : (i + 1) * k]

        # Pick disjoint source rows for QA vs answer-only inside each file.
        random.Random(args.seed + 10007 * i).shuffle(forget_chunk)
        random.Random(args.seed + 10007 * i + 1).shuffle(retain_chunk)

        f_qa_rows = forget_chunk[:qa_k]
        f_ans_rows = forget_chunk[qa_k : qa_k + ans_k]
        combined_forget = [f"{q} {a}" for q, a in f_qa_rows] + [a for _, a in f_ans_rows]
        random.Random(args.seed + 20011 * i).shuffle(combined_forget)

        r_qa_rows = retain_chunk[:qa_k]
        r_ans_rows = retain_chunk[qa_k : qa_k + ans_k]
        combined_retain = [f"{q} {a}" for q, a in r_qa_rows] + [a for _, a in r_ans_rows]
        random.Random(args.seed + 20011 * i + 1).shuffle(combined_retain)

        neu = neutral_texts[i * k : (i + 1) * k]

        output_data = {
            "bio_forget": [
                truncate_to_first_tokens(t, args.max_tokens) for t in combined_forget
            ],
            "bio_retain": [
                truncate_to_first_tokens(t, args.max_tokens) for t in combined_retain
            ],
            "neutral": [
                truncate_to_first_tokens(t, args.max_tokens) for t in neu
            ],
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(
            f"Part {i:03d}: wrote {out_path} "
            f"({k} per label; disjoint batches, seed={args.seed})."
        )

    used_f_rows = n_files * k
    used_r_rows = n_files * k
    used_neu = n_files * k
    n_f_rows = len(forget_rows)
    n_r_rows = len(retain_rows)
    n_neu = len(neutral_texts)

    print("")
    print("Source pool coverage (rows used across all output files / available in each JSONL):")
    print(
        f"  bio_forget ({args.remove_path}): "
        f"source rows {_pct_str(used_f_rows, n_f_rows)} "
        f"(within each file: {qa_k} question+answer + {ans_k} answer-only, no overlap by source row)"
    )
    print(
        f"  bio_retain ({args.retain_path}): "
        f"source rows {_pct_str(used_r_rows, n_r_rows)} "
        f"(within each file: {qa_k} question+answer + {ans_k} answer-only, no overlap by source row)"
    )
    print(
        f"  neutral      ({args.neutral_path}): "
        f"text lines {_pct_str(used_neu, n_neu)}"
    )

    if n_files == 1:
        print(f"Done: {3 * k} rows total ({k} per label).")
    else:
        print(
            f"Done: {n_files} files × {k} per label (max possible without reuse: {max_files})."
        )


if __name__ == "__main__":
    main()
