# General Unlabeled Dataset Generation

This module handles the stochastic, stratified generation of unlabeled text batches from wide-distribution text sources (for example, Wikipedia and The Pile).

Under realistic audit scenarios, the specific ground-truth concepts or datasets targeted during unlearning are entirely unknown to the auditor. To evaluate an unlearned candidate model against its baseline without target-domain access (e.g., biosecurity or cyber-warfare vectors), this pipeline synthesizes a broad, general, label-free text pool. This approach relies on the hypothesis that systemic feature omissions and latent anomalies will naturally surface using robust feature detection methods.

This general distribution is used to collect activations from both the base model and its unlearned counterpart. These activations serve a dual purpose: first, to train the SNMF dictionary base, and subsequently, to project the unlearned model's activations onto this discovered feature space. This audit framework isolates latent tracing anomalies to reverse-engineer and discover the original unlearning target.

## Configuration Specification & Command-Line API

### Argument Topology Reference


| CLI Parameter          | Datatype | Default Setting           | Functional Mapping                                                        |
| ---------------------- | -------- | ------------------------- | ------------------------------------------------------------------------- |
| `--source1-path`       | `Path`   | `None`                    | Filepath to the first JSONL text pool.                                    |
| `--source2-path`       | `Path`   | `None`                    | Filepath to the second JSONL text pool.                                   |
| `--output-path`        | `Path`   | `data/general_data.json` | Default target JSON export path (or root name for multi-part runs).       |
| `--num-files`          | `int`    | `1`                       | Denotes how many disjoint JSON files to synthesize across the slice pool. |
| `--samples-per-source` | `int`    | `400`                     | Sample size (k) extracted from *each* domain source per file.             |
| `--max-tokens`         | `int`    | `256`                     | Token count truncation floor to bound contextual input widths.            |
| `--seed`               | `int`    | `42`                      | Global pseudo-random initialization factor for reproducibility.           |


### Source JSONL format (`--source1-path`, `--source2-path`)

Each path must point to a UTF-8 **JSON Lines** file: one JSON object per line. Each object must contain a non-empty string field `"text"`, extra keys are ignored.

```json
{"text": "First document body as plain text."}
{"text": "Second document. Numbers and punctuation are fine.", "extra": "ignored"}
{"text": "Unicode also works: café 中文"}
```

### CLI Invocation Examples

#### 1. Standard Production Run (Single Batch)

Generates a singular target file containing 800 total entries (400 English, 400 WikiText), keeping only the first 256 structural whitespace tokens per sequence:

```bash

python data_utils/create_general_data.py \
  --source1-path "pile.jsonl" \
  --source2-path "wikitext.jsonl" \
  --output-path data/audit_general_pretrain_800.json \
  --samples-per-source 400 \
  --max-tokens 256 \
  --seed 42
```

#### 2. Scaled Scalability Run (Multi-Part Sweep Generation)

Synthesizes 5 completely distinct, disjoint data partitions saved in the same output target directory (e.g., `data/general_data_part1.json` through `part5.json`). Each file contains 1,000 unique sequences (500 per source pool):

```bash
python data_utils/create_general_data.py \
  --source1-path "pile.jsonl" \
  --source2-path "wikitext.jsonl" \
  --output-path data/general_data.json \
  --num-files 5 \
  --samples-per-source 500 \
  --max-tokens 256 \
  --seed 1337
```


## Pipeline Architecture & Execution Flow

The generation process isolates components to guarantee reproducible, disjoint data batches across consecutive experimental runs.

### Core Execution Phases:

 1. **Linear Line Processing (`load_texts`):** Raw `.jsonl` source tracking files are parsed sequentially. Blank lines are skipped, but each non-empty line must be a valid JSON object containing a non-empty string `"text"` field. Invalid JSON, non-object JSON values, or records without a usable `"text"` field are not normalized by `load_texts` and should be removed from the source file beforehand to avoid downstream failures.
2. **Deterministic Permutation (`rng.shuffle`):** Both pools undergo pseudo-random in-place shuffling governed by a global random seed parameter, ensuring that even under disjoint array slicing, sample sequence boundaries are completely randomized.
3. **Disjoint Stratified Chunking:** The pipeline extracts identical sample counts ($k$) from both domains. Consecutive file iterations ($i$) calculate absolute sliding windows mathematically using a half-open interval:

$$\text{Slice Window} = [i \cdot k \quad , \quad (i + 1) \cdot k)$$

This guarantees that **zero sequence duplication or overlap** occurs across partitioned output file parts (`part_001.json`, `part_002.json`, etc.).

1. **Tokenization Alignment (`truncate_to_first_tokens`):** Extracted strings are mapped into a whitespace-split array token layout and truncated to a hard boundary (`--max-tokens`). This standardizes activation context windows across sequence arrays before they hit local model tokenizers during activation generation.

### Methodological Rationale: Dual-Source Selection

Relying on a single text corpus introduces severe **domain bias**, forcing the SNMF dictionary to overfit to specific structural or stylistic invariants rather than learning generalized language semantics. 

To mitigate this, this pipeline constructs a balanced control distribution by blending highly structured encyclopedic prose  with colloquial web-crawl text in a stratified, 50/50 disjoint mix.


## Technical Considerations & Safety Checks

- **Strict Non-Reuse Verification:** The execution script asserts that total requested rows do not exceed structural resource boundaries. If `--num-files` * `--samples-per-source` is greater than the total records available in the smallest source collection on disk, the system cleanly clips execution to the maximum possible disjoint factor (`max_files`), printing a runtime warning to standard output instead of falling back to unsafe duplicate item sampling.

