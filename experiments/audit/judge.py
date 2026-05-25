from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from experiments.audit.judge_constants import (
    JUDGE_RESPONSE_INSTRUCTIONS,
    JUDGE_SYSTEM_PROMPT,
)


def _finish_reason_label(response: Any) -> Optional[str]:
    """Extract a clean finish reason (e.g. ``MAX_TOKENS``)"""
    try:
        candidate = response.candidates[0]
        fr = getattr(candidate, "finish_reason", None)
        if fr is None:
            return None
        fr_str = str(fr.name if getattr(fr, "name", None) else fr).strip()
        if not fr_str:
            return None
        return fr_str.split(".")[-1].upper()
    except (AttributeError, IndexError, TypeError):
        return None


class GeminiAuditJudge:
    """
    An automated, blind evaluation framework that leverages Gemini LLM 
    to audit and diagnose LLM unlearning.

    This judge ingests a structured report containing mechanistic evidence—specifically 
    Semi-Nonnegative Matrix Factorization (SNMF) activation deltas, token-level logit 
    lenses, and corpus-niche rare words. Without prior exposure to the true forget 
    objective, it evaluates whether a candidate model exhibits unlearning 
    signatures relative to its base model, quantifies its evaluation as a confidence metric, 
    and reconstructs the latent targeted concept within a structured JSON schema.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        api_key_env: str = "GOOGLE_API_KEY",
    ) -> None:
        """Initialize the audit judge framework.

        Raises:
            ValueError: If the API key referenced by ``api_key_env`` is missing
                or blank.

        Args:
            model: Gemini model identifier (e.g. ``gemini-2.5-pro``).
            temperature: Controls generation randomness, ``0.0`` is greedy.
            max_output_tokens: Token budget for the judge completion.
            api_key_env: Environment variable name holding the API key.
        """
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ValueError(
                f"{api_key_env} is not set or empty; skipping judge call."
            )

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_key_env = api_key_env
        self._api_key = api_key

    @staticmethod
    def format_rare_words(words: Sequence[Dict[str, Any]]) -> str:
        """Serialize rare-word rows ``{word, count, zipf, ...}`` for the judge.

        Example output: ``cascode(n=12, z=1.45)`` from structured entries.
        """
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
            "(residual = || A - Z Y ||_F^2 / || A ||_F^2 ; "
            "rel_delta and rel_delta_max/mean summarize per-latent fractional change "
            "in mean peak coefficient, (E[Y_base]-E[Y_cand])/(E[Y_base]+1e-9).)"
        )
        for row in summary:
            parts.append(
                f"  L{row['layer']:02d}  residual_base={row['residual_base']:.4f}  "
                f"residual_candidate={row['residual_candidate']:.4f}  "
                f"residual_delta={row['residual_delta']:+.4f}  "
                f"rel_delta_max={row['rel_delta_max']:+.4f}  "
                f"rel_delta_mean={row['rel_delta_mean']:+.4f}"
            )
        parts.append("")

    @staticmethod
    def _append_top_features(parts: List[str], global_top: List[Dict[str, Any]], rank_by: str) -> None:
        parts.append(f"--- Top-{len(global_top)} most-changed features (ranked by {rank_by}) ---\n")
        fmt_rare = GeminiAuditJudge.format_rare_words
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
                tok_strs = [f"{json.dumps(t['token'])} ({t['logit']:+.2f})" for t in vocab]
                parts.append("    tokens-most-promoted (logit-lens via M_base): " + "  ".join(tok_strs))
                
            if rare_str := fmt_rare(rec.get("rare_context_words") or []):
                parts.append("    rare-context words (count*(cutoff-zipf), rarer/more-recurring first): " + rare_str)
            parts.append("")

    @staticmethod
    def _append_layer_vocab(parts: List[str], layers_vocab: Optional[List[Dict[str, Any]]]) -> None:
        if not layers_vocab:
            return
        parts.append("--- Per-layer AGGREGATE logit-lens (sum of that layer's top-decreased features through W_down) ---")
        parts.append("This is the joint signal: tokens promoted by the *direction-sum* of the layer's most-changed features.\n")
        
        for row in layers_vocab:
            tag = "delta-weighted" if row.get("delta_weighted") else "uniform-sum"
            parts.append(f"  L{int(row['layer']):02d}  ({tag}, n_features={row.get('n_features_summed')}, residual_norm={row.get('residual_norm', 0.0):.3f}):")
            tok_strs = [f"{json.dumps(t['token'])} ({t['logit']:+.2f})" for t in (row.get("tokens") or [])]
            parts.append("      " + ("  ".join(tok_strs) if tok_strs else "(no tokens)"))
        parts.append("")

    @staticmethod
    def _append_global_vocab(parts: List[str], global_vocab: Optional[Dict[str, Any]], top_count: int) -> None:
        if not global_vocab:
            return
        tag = "delta-weighted" if global_vocab.get("delta_weighted") else "uniform-sum"
        parts.append(f"--- GLOBAL AGGREGATE logit-lens (sum across all top-{top_count} cross-layer features) ---")
        parts.append(f"  ({tag}, n_features_summed={global_vocab.get('n_features_summed')}, n_layers_spanned={global_vocab.get('n_layers_spanned')}, residual_norm={global_vocab.get('residual_norm', 0.0):.3f}):")
        tok_strs = [f"{json.dumps(t['token'])} ({t['logit']:+.2f})" for t in (global_vocab.get("tokens") or [])]
        parts.append("    " + ("  ".join(tok_strs) if tok_strs else "(no tokens)"))
        parts.append("")

    @staticmethod
    def _append_layer_rare_words(parts: List[str], layers_rare: Optional[List[Dict[str, Any]]]) -> None:
        if not layers_rare:
            return
        parts.append("--- Per-layer AGGREGATE rare-context words (rare/topical vocabulary recurring across top-decreased features) ---\n")
        fmt_rare = GeminiAuditJudge.format_rare_words
        
        for row in layers_rare:
            words_str = fmt_rare(row.get("words") or [])
            parts.append(f"  L{int(row['layer']):02d}  (n_features={row.get('n_features_pooled')}, n_contexts={row.get('n_contexts')}, zipf_cutoff={float(row.get('zipf_cutoff', 0.0)):.2f}):")
            parts.append("      " + (words_str or "(no rare words above cutoff)"))
        parts.append("")

    @staticmethod
    def _append_global_rare_words(parts: List[str], global_rare: Optional[Dict[str, Any]], top_count: int) -> None:
        if not global_rare:
            return
        parts.append(f"--- GLOBAL AGGREGATE rare-context words (pooled across all top-{top_count} cross-layer features) ---")
        parts.append(f"  (n_features={global_rare.get('n_features_pooled')}, n_contexts={global_rare.get('n_contexts')}, zipf_cutoff={float(global_rare.get('zipf_cutoff', 0.0)):.2f}):")
        words_str = GeminiAuditJudge.format_rare_words(global_rare.get("words") or [])
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
        """
        Builds the full audit text payload for the judge.
        """
        # 1. Header Metadata Section
        parts = [
            JUDGE_SYSTEM_PROMPT,
            "=" * 72, "AUDIT REPORT", "=" * 72,
            f"n_audit_prompts:      {n_prompts}",
            f"audited_layers:       {layers}",
            f"rank_by:              {rank_by}\n"
        ]

        # 2. Append Sections to the prompt
        GeminiAuditJudge._append_layer_summary(parts, per_layer_summary)
        GeminiAuditJudge._append_top_features(parts, global_top, rank_by)
        GeminiAuditJudge._append_layer_vocab(parts, per_layer_aggregate_vocab)
        GeminiAuditJudge._append_global_vocab(parts, global_aggregate_vocab, len(global_top))
        GeminiAuditJudge._append_layer_rare_words(parts, per_layer_rare_words)
        GeminiAuditJudge._append_global_rare_words(parts, global_rare_words, len(global_top))

        # 3. Footer System Core Context
        parts.extend(["=" * 72, JUDGE_RESPONSE_INSTRUCTIONS])
        return "\n".join(parts)


    def call_gemini(self, prompt: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Invoke Gemini with this instance's model and generation settings.

        Uses the API key validated at construction time (see ``__init__``).

        Returns:
            ``(text, error, finish_reason)``. ``finish_reason`` is set when the
            SDK returns a response with candidates (e.g. ``STOP``, ``MAX_TOKENS``).
        """
        errors: List[str] = []

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore

            client = genai.Client(api_key=self._api_key)
            gen_cfg = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
            )
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gen_cfg,
            )
            text = getattr(resp, "text", "") or ""
            return text, None, _finish_reason_label(resp)
        except ImportError:
            errors.append("google-genai not installed")
        except Exception as e:
            errors.append(f"google-genai call failed: {e}")

        try:
            import google.generativeai as legacy  # type: ignore

            legacy.configure(api_key=self._api_key)
            gm = legacy.GenerativeModel(self.model)
            resp = gm.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                    "response_mime_type": "application/json",
                },
            )
            text = getattr(resp, "text", "") or ""
            return text, None, _finish_reason_label(resp)
        except ImportError:
            errors.append(
                "google-generativeai not installed; install one of "
                "'google-genai' or 'google-generativeai' to enable the judge step "
                "(e.g. `pip install google-genai`)."
            )
        except Exception as e:
            errors.append(f"google-generativeai call failed: {e}")

        return "", " | ".join(errors), None


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
    """Best-effort: parse the model's output as JSON; fall back to regex search."""
    cleaned = _strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"_parse_error": True, "_raw_text": text}