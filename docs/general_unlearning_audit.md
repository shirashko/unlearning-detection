# General (label-free) SNMF unlearning audit

Use this when:

- you want to know whether *something* was unlearned between two checkpoints, without committing to a particular forget concept up-front, and
- what concept the change most plausibly targets.

## Pipeline

1. Load Z = F per layer from `$SNMF_DIR/layer_*/snmf_factors.pt` (basis trained on M_base).
2. Run M_base and M_candidate on the **same** unlabeled prompts; collect `mlp_intermediate` activations per layer.
3. Project both onto Z via ridge least squares → Y_base, Y_candidate.
4. Per-feature unlabeled signal:
   - `delta_i` = E[Y_base_max,i] − E[Y_candidate_max,i] (over all prompts, per-prompt peak across tokens, same as the supervised audit)
   - `rel_delta_i` = delta_i / (E[Y_base_max,i] + epsilon) — the fractional drop in mean peak activation. This better captures “surgical” unlearning of niche concepts: a feature that goes from 0.05 → 0 (`rel_delta=1.0`) is ranked above a feature that drops from 5.0 → 4.0 (`rel_delta=0.2`) even though the latter has a larger absolute delta.
5. Globally rank latents by `rank_by`: `rel_delta` (default) = fractional decrease in mean peak activation; `abs_rel_delta` = magnitude of that fractional change (either direction).
6. For each top-K latent, pull the top-N most-activating tokens (from M_base’s Y on the audit prompts) and quote their local windows with the peak token marked **like_this**.
7. Logit-lens each top-K latent through M_base’s `mlp.down_proj` + `final_norm` + `lm_head`:

   ```
   r_i      = W_down_L @ F_L[:, i]       # in residual space
   logits_i = lm_head(final_norm(r_i))   # in vocab space
   ```

   Take the top `--vocab-lens-top-k` tokens of `logits_i`. This gives the **output** side of each feature (what tokens it writes), complementing the **input** side (`top_contexts`).

8. Aggregate logit-lens (joint signal) at two scopes:
   - **Per-layer:** `r_L = W_down_L @ ( Σ_i w_i · F_L[:, i] )` over the layer’s top-decreased latents.
   - **Global:** `r_global = Σ_L W_down_L @ ( Σ_i w_i · F_L[:, i] )` over all (layer, latent) pairs in the cross-layer top set.

   Both go through the same `final_norm` + `lm_head` + topk. By default, `w_i` is the feature’s `rank_by` score (e.g. rel_delta), so the aggregate emphasizes the most-changed features. `--no-lens-delta-weighted` switches to a uniform sum.

9. Pack a single message for a judge LLM (Gemini 2.5 Flash by default) asking two things:
   - (a) what concept does this most plausibly look like the unlearned one
   - (b) confidence (0–100 %) that unlearning actually happened

## Outputs (under `--output-dir`)

| Path | Description |
|------|-------------|
| `layer_<i>/audit.json` | Per-latent profile (delta, mean coefs, top contexts, `top_vocab_base`, plus the per-layer `top_vocab_base_sum` aggregate) |
| `layer_<i>/audit_features.csv` | Flat per-latent table for that layer |
| `audit_summary.json` | Per-layer aggregates + global top-K + `per_layer_aggregate_vocab` + `global_aggregate_vocab` + judge verdict |
| `judge_prompt.txt` | Exact prompt sent to the judge LLM |
| `judge_response.json` | Parsed verdict from the judge |
| `judge_response_raw.txt` | Raw text returned by the judge (for debugging) |

## Run

```bash
python experiments/audit/general_unlearning_audit.py \
    --base-model-path  /path/to/base \
    --candidate-model-path /path/to/maybe_unlearned \
    --snmf-dir         outputs/.../results_data_partN \
    --data-path        data/general_data_part1.json \
    --layers           10-18 \
    --output-dir       outputs/audit_general/run_xyz \
    --top-k-global     20 \
    --contexts-per-feature 8
```

Or load settings from YAML (CLI overrides the file):

```bash
python experiments/audit/general_unlearning_audit.py \
    --config configs/audit/example_general_audit.yaml
```

## Dataset format

The JSON dataset can be either:

- a flat list of strings (e.g. `data/general_data_partN.json`), or
- a dict of `{label: [strings, ...]}` (labels are ignored here).
