from __future__ import annotations

from typing import Final, FrozenSet

# Stems for classic HF-style bracket specials; keep logit-lens patterns and rare-word
# denylist aligned on this set.
AUDIT_EXACT_SPECIAL_STEMS: Final[FrozenSet[str]] = frozenset({
    "bos",
    "eos",
    "pad",
    "unk",
    "mask",
    "sep",
    "cls",
})

# Decoded context windows sometimes expose chat / turn delimiters as bare words;
# include for rare-word filtering only (not as explicit "<eot>" patterns for lens ID scan).
_AUDIT_RARE_WORD_EXTRA_STEMS: Final[FrozenSet[str]] = frozenset({
    "eot",
    "sot",
})

RARE_WORD_SPECIAL_TOKEN_DENYLIST: Final[FrozenSet[str]] = (
    AUDIT_EXACT_SPECIAL_STEMS | _AUDIT_RARE_WORD_EXTRA_STEMS
)

LOGIT_LENS_SPECIAL_TOKEN_PATTERNS: Final[tuple[str, ...]] = tuple(
    f"<{s}>" for s in sorted(AUDIT_EXACT_SPECIAL_STEMS)
)

LOGIT_LENS_SPECIAL_TOKEN_PREFIXES: Final[tuple[str, ...]] = (
    "<unused",
    "<reserved",
    "<start_of_turn",
    "<end_of_turn",
    "<|",
    "<extra_id_",
)
