from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


from experiments.audit.judge_constants import (
    JUDGE_RESPONSE_INSTRUCTIONS,
    JUDGE_SYSTEM_PROMPT,
)
from llm_utils.gemini_client import GeminiClient
from llm_utils.utils import format_audited_layers


class UnlearningAuditReporter:
    """
    Domain manager for mechanistic audit reports.

    Aggregates SNMF / logit-lens evidence into the master investigation prompt
    and parses the unlearning-target verdict JSON.
    """

    def __init__(self, client: GeminiClient) -> None:
        self.client = client

    @staticmethod
    def format_rare_words(words: Sequence[Dict[str, Any]]) -> str:
        """Serialize rare-word rows ``{word, count, zipf, ...}`` for the judge."""
        if not words:
            return ""
        chunks: List[str] = []
        for entry in words:
            w = entry.get("word", "")
            n = int(entry.get("count", 0))
            z = float(entry.get("zipf", 0.0))
            chunks.append(f"{w}(n={n}, z={z:.2f})")
        return "  ".join(chunks)

@staticmethod
    def _append_layer_summary(parts: List[str], summary: List[Dict[str, Any]]) -> None:
        parts.append("--- Per-layer reconstruction & delta summary ---")
        parts.append(
            "(residual = || A - Z Y ||_F^2 / || A ||_F^2, rel_delta = per-latent "
            "fractional change in mean peak coefficient, "
            "(E[Y_base]-E[Y_cand])/(E[Y_base]+1e-9).)"
        )
        for row in summary:
            parts.append(
                f"  L{row['layer']:02d} | "
                f"residual (base / cand): {row['residual_base']:.4f} / {row['residual_candidate']:.4f} | "
                f"residual Δ: {row['residual_delta']:+.4f} | "
                f"rel_delta (max / mean): {row['rel_delta_max']:+.4f} / {row['rel_delta_mean']:+.4f}"
            )
        parts.append("")

    @staticmethod
    def _append_top_features(
        parts: List[str], global_top: List[Dict[str, Any]], rank_by: str,
    ) -> None:
        parts.append(
            f"--- Top-{len(global_top)} most-changed features (ranked by {rank_by}) ---\n"
        )
        fmt_rare = UnlearningAuditReporter.format_rare_words
        for k, rec in enumerate(global_top, start=1):
            parts.append(
                f"[{k}] layer L{rec['layer']}, latent {rec['latent_idx']} | "
                f"rel_delta={rec.get('rel_delta', 0.0):+.4f}  "
                f"abs_rel_delta={rec.get('abs_rel_delta', 0.0):.4f} | "
                f"mean_base={rec['mean_Y_base']:.4f}  "
                f"mean_candidate={rec['mean_Y_candidate']:.4f}"
            )

            if ctxs := rec.get("top_contexts"):
                for j, ctx in enumerate(ctxs, start=1):
                    parts.append(
                        f"    {j:>2}. act={ctx['activation']:.3f}  | {ctx['context']}"
                    )
            else:
                parts.append("    (no contexts recorded)")

            if vocab := rec.get("top_vocab_base"):
                tok_strs = [
                    f"{json.dumps(t['token'])} ({t['logit']:+.2f})" for t in vocab
                ]
                parts.append(
                    "    tokens-most-promoted (logit-lens via M_base): "
                    + "  ".join(tok_strs)
                )

            if rare_str := fmt_rare(rec.get("rare_context_words") or []):
                parts.append(
                    "    rare-context words (count*(cutoff-zipf), "
                    "rarer/more-recurring first): " + rare_str
                )
            parts.append("")

    @staticmethod
    def _append_layer_vocab(
        parts: List[str], layers_vocab: Optional[List[Dict[str, Any]]],
    ) -> None:
        if not layers_vocab:
            return
        parts.append(
            "--- Per-layer AGGREGATE logit-lens (sum of that layer's "
            "top-decreased features through W_down) ---"
        )
        parts.append(
            "This is the joint signal: tokens promoted by the *direction-sum* "
            "of the layer's most-changed features.\n"
        )

        for row in layers_vocab:
            tag = "delta-weighted" if row.get("delta_weighted") else "uniform-sum"
            parts.append(
                f"  L{int(row['layer']):02d}  ({tag}, "
                f"n_features={row.get('n_features_summed')}, "
                f"residual_norm={row.get('residual_norm', 0.0):.3f}):"
            )
            tok_strs = [
                f"{json.dumps(t['token'])} ({t['logit']:+.2f})"
                for t in (row.get("tokens") or [])
            ]
            parts.append("      " + ("  ".join(tok_strs) if tok_strs else "(no tokens)"))
        parts.append("")

    @staticmethod
    def _append_global_vocab(
        parts: List[str],
        global_vocab: Optional[Dict[str, Any]],
        top_count: int,
    ) -> None:
        if not global_vocab:
            return
        tag = "delta-weighted" if global_vocab.get("delta_weighted") else "uniform-sum"
        parts.append(
            f"--- GLOBAL AGGREGATE logit-lens (sum across all top-{top_count} "
            "cross-layer features) ---"
        )
        parts.append(
            f"  ({tag}, n_features_summed={global_vocab.get('n_features_summed')}, "
            f"n_layers_spanned={global_vocab.get('n_layers_spanned')}, "
            f"residual_norm={global_vocab.get('residual_norm', 0.0):.3f}):"
        )
        tok_strs = [
            f"{json.dumps(t['token'])} ({t['logit']:+.2f})"
            for t in (global_vocab.get("tokens") or [])
        ]
        parts.append("    " + ("  ".join(tok_strs) if tok_strs else "(no tokens)"))
        parts.append("")

    @staticmethod
    def _append_layer_rare_words(
        parts: List[str], layers_rare: Optional[List[Dict[str, Any]]],
    ) -> None:
        if not layers_rare:
            return
        parts.append(
            "--- Per-layer AGGREGATE rare-context words (rare/topical vocabulary "
            "recurring across top-decreased features) ---\n"
        )
        fmt_rare = UnlearningAuditReporter.format_rare_words

        for row in layers_rare:
            words_str = fmt_rare(row.get("words") or [])
            parts.append(
                f"  L{int(row['layer']):02d}  (n_features={row.get('n_features_pooled')}, "
                f"n_contexts={row.get('n_contexts')}, "
                f"zipf_cutoff={float(row.get('zipf_cutoff', 0.0)):.2f}):"
            )
            parts.append("      " + (words_str or "(no rare words above cutoff)"))
        parts.append("")

    @staticmethod
    def _append_global_rare_words(
        parts: List[str],
        global_rare: Optional[Dict[str, Any]],
        top_count: int,
    ) -> None:
        if not global_rare:
            return
        parts.append(
            f"--- GLOBAL AGGREGATE rare-context words (pooled across all "
            f"top-{top_count} cross-layer features) ---"
        )
        parts.append(
            f"  (n_features={global_rare.get('n_features_pooled')}, "
            f"n_contexts={global_rare.get('n_contexts')}, "
            f"zipf_cutoff={float(global_rare.get('zipf_cutoff', 0.0)):.2f}):"
        )
        words_str = UnlearningAuditReporter.format_rare_words(
            global_rare.get("words") or []
        )
        parts.append("    " + (words_str or "(no rare words above cutoff)"))
        parts.append("")

    @staticmethod
    def build_audit_report_prompt(
        *,
        n_prompts: int,
        layers: List[int],
        rank_by: str,
        global_top: List[Dict[str, Any]],
        per_layer_summary: List[Dict[str, Any]],
        per_layer_aggregate_vocab: Optional[List[Dict[str, Any]]] = None,
        global_aggregate_vocab: Optional[Dict[str, Any]] = None,
        per_layer_rare_words: Optional[List[Dict[str, Any]]] = None,
        global_rare_words: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the full audit text payload for the judge."""
        parts = [
            JUDGE_SYSTEM_PROMPT,
            "=" * 72, "AUDIT REPORT", "=" * 72,
            f"n_audit_prompts:      {n_prompts}",
            f"audited_layers:       {format_audited_layers(layers)}",
            f"rank_by:              {rank_by}\n",
        ]

        UnlearningAuditReporter._append_layer_summary(parts, per_layer_summary)
        UnlearningAuditReporter._append_top_features(parts, global_top, rank_by)
        UnlearningAuditReporter._append_layer_vocab(parts, per_layer_aggregate_vocab)
        UnlearningAuditReporter._append_global_vocab(
            parts, global_aggregate_vocab, len(global_top),
        )
        UnlearningAuditReporter._append_layer_rare_words(parts, per_layer_rare_words)
        UnlearningAuditReporter._append_global_rare_words(
            parts, global_rare_words, len(global_top),
        )

        parts.extend(["=" * 72, JUDGE_RESPONSE_INSTRUCTIONS])
        return "\n".join(parts)

    def run_prompt(
        self, prompt: str,
    ) -> Tuple[Dict[str, Any], str, Optional[str], Optional[str]]:
        """
        Send a pre-built audit prompt and parse the verdict JSON.

        Returns:
            ``(verdict, raw_text, error, finish_reason)``
        """
        raw_text, error, finish_reason = self.client.generate_text(prompt)
        if error or not raw_text:
            return (
                {"_parse_error": f"Audit execution failed. {error or 'empty response'}"},
                raw_text or "",
                error or "empty response",
                finish_reason,
            )
        return parse_judge_json(raw_text), raw_text, None, finish_reason

    def execute_audit(
        self,
        *,
        n_prompts: int,
        layers: List[int],
        rank_by: str,
        global_top: List[Dict[str, Any]],
        per_layer_summary: List[Dict[str, Any]],
        per_layer_aggregate_vocab: Optional[List[Dict[str, Any]]] = None,
        global_aggregate_vocab: Optional[Dict[str, Any]] = None,
        per_layer_rare_words: Optional[List[Dict[str, Any]]] = None,
        global_rare_words: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any], str, Optional[str], Optional[str]]:
        """
        Build the audit prompt, call Gemini, and parse the verdict.

        Returns:
            ``(prompt, verdict, raw_text, error, finish_reason)``
        """
        prompt = self.build_audit_report_prompt(
            n_prompts=n_prompts,
            layers=layers,
            rank_by=rank_by,
            global_top=global_top,
            per_layer_summary=per_layer_summary,
            per_layer_aggregate_vocab=per_layer_aggregate_vocab,
            global_aggregate_vocab=global_aggregate_vocab,
            per_layer_rare_words=per_layer_rare_words,
            global_rare_words=global_rare_words,
        )
        verdict, raw_text, error, finish_reason = self.run_prompt(prompt)
        return prompt, verdict, raw_text, error, finish_reason



def _strip_code_fence(text: str) -> str:
    """
    Strip at most one leading markdown code fence.

    Removes a trailing `` ``` `` only when it appears as an actual closing
    fence at end-of-string (regex), avoiding blind ``s[:-3]`` truncation when
    the model omits the closer or the payload ends with unrelated `` ``` ``.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```[\w-]*\s*\n?", "", s, count=1)
    s = re.sub(r"\s*```\s*$", "", s, count=1)
    return s.strip()


def parse_judge_json(text: str) -> Dict[str, Any]:
    """Best-effort: parse the model's output as a JSON object, fall back to regex search."""
    cleaned = _strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {
        "_parse_error": "Failed to parse JSON from judge response.",
        "_raw_text": text,
    }

