from __future__ import annotations
import re
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch

from experiments.audit.context_windows import _marked_context_text
from experiments.audit.special_tokens import RARE_WORD_SPECIAL_TOKEN_DENYLIST

try:
    from wordfreq import zipf_frequency as _zipf_frequency  # type: ignore
    HAS_WORDFREQ = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    _zipf_frequency = None  # type: ignore[assignment]
    HAS_WORDFREQ = False

# Strip **peak token** markers from contexts tokenized for rare-word ranking.
_EMPHASIS_MARKER_RE = re.compile(r"\*\*([^*]*)\*\*")

# ASCII letters + internal hyphens/apostrophes for Zipf wordfreq extraction.
_CONTEXT_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")


def strip_emphasis_markers(text: str) -> str:
    """Remove ``**...**`` peak-token markers; keep the wrapped text."""
    return _EMPHASIS_MARKER_RE.sub(r"\1", text or "")


def rare_word_ranking_from_contexts(
    contexts: Sequence[str],
    *,
    top_n: int,
    zipf_cutoff: float,
    min_word_len: int,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """Aggregate words across ``contexts`` and rank them by topical rarity.

    For each word we compute its Zipf frequency in everyday text (via
    ``wordfreq.zipf_frequency``, range ~0.0=never-seen to ~8.0=THE most
    common word). Words with Zipf >= ``zipf_cutoff`` (common English: ``the``,
    ``of``, ``and``, ...) are dropped, so what remains is topical
    vocabulary. Tokens matching a small denylist of model specials (e.g.
    ``unk``, ``bos``, ``eos`` as extracted from decoded ``<unk>``-style
    strings) are dropped before counting. The remaining words are scored by

        score = count_in_contexts * max(zipf_cutoff - zipf, 0.5)

    -- a recurring rare word (high count, low zipf) outranks both a one-shot
    rare word and a common word that happens to repeat. The 0.5 floor stops
    completely unknown OOV strings (which ``wordfreq`` returns as zipf=0)
    from dominating the ranking just by being unknown; recognized rare
    words still get larger rarity boosts.

    Returns up to ``top_n`` dicts of ``{"word", "count", "zipf", "score"}``,
    sorted by score (desc), zipf (asc), count (desc). Returns ``[]`` when
    ``wordfreq`` is not installed or no qualifying word survives the filter.
    """
    if top_n <= 0 or not HAS_WORDFREQ or not contexts:
        return []
    counts: Dict[str, int] = {}
    for ctx in contexts:
        if not ctx:
            continue
        clean = strip_emphasis_markers(ctx).lower()
        for tok in _CONTEXT_WORD_RE.findall(clean):
            if len(tok) < min_word_len or tok in RARE_WORD_SPECIAL_TOKEN_DENYLIST:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        return []
    scored: List[Tuple[float, float, int, str]] = []
    for word, n in counts.items():
        z = float(_zipf_frequency(word, lang))
        if z >= zipf_cutoff:
            continue
        rarity_boost = max(zipf_cutoff - z, 0.5)
        scored.append((n * rarity_boost, -z, n, word))
    if not scored:
        return []
    scored.sort(reverse=True)
    return [
        {"word": w, "count": int(c), "zipf": float(-neg_z), "score": float(s)}
        for s, neg_z, c, w in scored[:top_n]
    ]


def extract_context_strings(contexts: Sequence[Any]) -> List[str]:
    """Pull text out of either ``[{"context": str, ...}, ...]`` or ``[str, ...]``."""
    out: List[str] = []
    for c in contexts or []:
        if isinstance(c, dict):
            t = c.get("context")
            if isinstance(t, str) and t:
                out.append(t)
        elif isinstance(c, str) and c:
            out.append(c)
    return out


def top_contexts_for_latent(
    Y_base: torch.Tensor,
    token_ids: List[int],
    sample_ids: List[int],
    spans: Dict[int, Tuple[int, int]],
    tokenizer: Any,
    latent_idx: int,
    n_contexts: int,
    context_window: int,
) -> List[Dict[str, Any]]:
    """
    For latent ``latent_idx``, pick the n_contexts tokens with the highest
    Y_base[latent_idx, :] activation and return their local windows (each peak
    token wrapped in **...**), mimicking ``top_positive_activation_contexts``.
    """
    row = Y_base[latent_idx].detach().cpu().numpy().astype(np.float64, copy=False)
    n_tokens = row.shape[0]
    if n_tokens == 0:
        return []
    k = max(1, min(n_contexts, n_tokens))
    top_idx = np.argpartition(-row, k - 1)[:k]
    top_idx = top_idx[np.argsort(-row[top_idx])]

    sample_ids_arr = np.asarray(sample_ids)
    token_ids_arr = np.asarray(token_ids, dtype=np.int64)

    out: List[Dict[str, Any]] = []
    seen_keys = set()
    for gi in top_idx:
        gi_int = int(gi)
        sid = int(sample_ids_arr[gi_int])
        ctx = _marked_context_text(
            tokenizer, token_ids_arr, sample_ids_arr, spans,
            gi_int, context_window,
        )
        key = (sid, ctx)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append({
            "activation": float(row[gi_int]),
            "sample_id": sid,
            "context": ctx,
        })
    return out
