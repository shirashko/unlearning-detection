"""
Flat token-array helpers extracted from supervised SNMF tooling.

Maps ``sample_ids`` contiguous spans into global token slices and renders local
peak-marked context strings (SentencePiece-aware) for audits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _sample_id_to_spans(sample_ids_arr: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """Map each sample_id to contiguous [start, end) slice in flat token arrays."""
    n = len(sample_ids_arr)
    if n == 0:
        return {}
    boundaries = np.r_[0, np.flatnonzero(sample_ids_arr[1:] != sample_ids_arr[:-1]) + 1, n]
    spans: Dict[int, Tuple[int, int]] = {}
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        spans[int(sample_ids_arr[a])] = (int(a), int(b))
    return spans


def _token_piece(tokenizer: Any, tid: int) -> str:
    toks = tokenizer.convert_ids_to_tokens([int(tid)])
    return toks[0] if toks else str(tid)


def _sp_piece_to_space(text: str) -> str:
    """SentencePiece / Gemma use U+2581 at subword starts; swap for ASCII space."""
    return text.replace("\u2581", " ")


def _marked_context_text(
    tokenizer: Any,
    all_token_ids: np.ndarray,
    sample_ids_arr: np.ndarray,
    spans: Dict[int, Tuple[int, int]],
    global_idx: int,
    context_window: int,
) -> str:
    """
    Local window (``context_window`` tokens each side, same sample only); pieces from
    ``convert_ids_to_tokens`` concatenated; peak token wrapped in ``**...**``.
    """
    sid = int(sample_ids_arr[global_idx])
    samp_start, samp_end = spans[sid]
    win_lo = max(samp_start, int(global_idx) - context_window)
    win_hi = min(samp_end, int(global_idx) + context_window + 1)
    parts: List[str] = []
    for j in range(win_lo, win_hi):
        if int(sample_ids_arr[j]) != sid:
            continue
        piece = _token_piece(tokenizer, int(all_token_ids[j]))
        parts.append("**" + piece + "**" if j == global_idx else piece)
    return _sp_piece_to_space("".join(parts))
