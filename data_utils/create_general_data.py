import argparse
import json
import random
from pathlib import Path
from typing import List

BASE_DATASET_PATH = "/home/morg/students/rashkovits/Localized-UNDO/datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an unlabeled pretrain JSON file (flat list of texts) sampled from "
            "train_eng.jsonl and train_wikitext.jsonl. Use --num-files > 1 to write "
            "several disjoint batches of --samples-per-source per source."
        )
    )
    parser.add_argument(
        "--eng-path",
        type=Path,
        default=Path(f"{BASE_DATASET_PATH}/pretrain/train_eng.jsonl"),
        help="Path to English pretrain JSONL.",
    )
    parser.add_argument(
        "--wikitext-path",
        type=Path,
        default=Path(f"{BASE_DATASET_PATH}/pretrain/train_wikitext.jsonl"),
        help="Path to WikiText pretrain JSONL.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/pretrain_data.json"),
        help=(
            "Output path when --num-files=1. When --num-files>1, writes "
            "<stem>_partNNN<suffix> in the same directory."
        ),
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=1,
        help=(
            "How many JSON files to create. Each file has --samples-per-source rows per source. "
            "Batches are disjoint (no row reused across files). If you ask for more files than "
            "the smallest source allows, the count is clipped and a warning is printed."
        ),
    )
    parser.add_argument(
        "--samples-per-source",
        type=int,
        default=400,
        help="How many samples per source in each output file.",
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
            text = record.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def truncate_to_first_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


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
    print("Starting pretrain data creation...")
    if args.num_files < 1:
        raise SystemExit("--num-files must be >= 1")
    k = args.samples_per_source
    if k < 1:
        raise SystemExit("--samples-per-source must be >= 1")

    eng_texts = load_texts(args.eng_path)
    wiki_texts = load_texts(args.wikitext_path)

    max_files = min(len(eng_texts) // k, len(wiki_texts) // k)
    if max_files == 0:
        raise ValueError(
            f"Sources are too small for one file of {k} samples per source "
            f"(need ≥{k} rows from each of eng and wikitext)."
        )

    n_files = min(args.num_files, max_files)
    if n_files < args.num_files:
        print(
            f"WARNING: requested {args.num_files} files but only {max_files} disjoint batches "
            f"of {k} per source are possible; writing {n_files} file(s)."
        )

    rng = random.Random(args.seed)
    rng.shuffle(eng_texts)
    rng.shuffle(wiki_texts)

    out_paths = part_output_paths(args.output_path, n_files)
    for i, out_path in enumerate(out_paths):
        eng_chunk = eng_texts[i * k : (i + 1) * k]
        wiki_chunk = wiki_texts[i * k : (i + 1) * k]

        combined = list(eng_chunk) + list(wiki_chunk)
        random.Random(args.seed + 30013 * i).shuffle(combined)

        output_data = [
            truncate_to_first_tokens(t, args.max_tokens) for t in combined
        ]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(
            f"Part {i:03d}: wrote {out_path} "
            f"({k} per source, {len(output_data)} total; seed={args.seed})."
        )

    used_eng = n_files * k
    used_wiki = n_files * k

    print("")
    print("Source pool coverage (rows used across all output files / available in each JSONL):")
    print(f"  eng      ({args.eng_path}): {_pct_str(used_eng, len(eng_texts))}")
    print(f"  wikitext ({args.wikitext_path}): {_pct_str(used_wiki, len(wiki_texts))}")

    if n_files == 1:
        print(f"Done: {2 * k} rows total ({k} per source).")
    else:
        print(
            f"Done: {n_files} files × {2 * k} rows each "
            f"({k} per source; max possible without reuse: {max_files})."
        )


if __name__ == "__main__":
    main()
