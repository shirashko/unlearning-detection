"""Abstract and concrete strategies for ranking latent features in label-free audits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Type

import numpy as np

# Minimum positive scale for rel_delta denominators: rank uses max(mean_base, eps), not mean_base+eps,
# so near-zero or (under ridge projection) slightly negative base means cannot invert or explode the ratio.
REL_DELTA_EPS: float = 1e-9


class MeanPeakUnlearningMetrics(NamedTuple):
    """Per-latent statistics from mean peak SNMF coefficients (base vs candidate)."""

    rel_delta: np.ndarray
    abs_rel_delta: np.ndarray


def compute_mean_peak_metrics(
    mean_base: np.ndarray,
    mean_candidate: np.ndarray,
    *,
    eps: float = REL_DELTA_EPS,
) -> MeanPeakUnlearningMetrics:
    """Fractional change vs M_base with denominator max(mean_base, eps)."""
    mean_b = np.asarray(mean_base)
    mean_c = np.asarray(mean_candidate)
    raw = mean_b - mean_c
    denom = np.maximum(mean_b, eps)
    rel_delta = raw / denom
    abs_rel_delta = np.abs(rel_delta)
    return MeanPeakUnlearningMetrics(rel_delta, abs_rel_delta)


class BaseFeatureRanker(ABC):
    """Strategy for ranking latent features from base vs candidate mean-peak stats."""

    @property
    @abstractmethod
    def record_field(self) -> str:
        """Key in per-latent JSON records used for ordering (must match exported columns)."""

    @abstractmethod
    def ranking_vector(self, metrics: MeanPeakUnlearningMetrics) -> np.ndarray:
        """Scalar score per latent; higher means more prioritized for the audit top-K."""

    def compute_scores(
        self,
        mean_base: np.ndarray,
        mean_candidate: np.ndarray,
        *,
        eps: float = REL_DELTA_EPS,
    ) -> np.ndarray:
        """Convenience: full metrics tuple then strategy-specific ranking vector."""
        m = compute_mean_peak_metrics(mean_base, mean_candidate, eps=eps)
        return self.ranking_vector(m)


class RelativeDeltaRanker(BaseFeatureRanker):
    """Fractional decay vs M_base: (mean_base - mean_cand) / max(mean_base, eps)."""

    @property
    def record_field(self) -> str:
        return "rel_delta"

    def ranking_vector(self, metrics: MeanPeakUnlearningMetrics) -> np.ndarray:
        return metrics.rel_delta


class AbsoluteRelativeDeltaRanker(BaseFeatureRanker):
    """Magnitude of fractional change: abs(rel_delta)."""

    @property
    def record_field(self) -> str:
        return "abs_rel_delta"

    def ranking_vector(self, metrics: MeanPeakUnlearningMetrics) -> np.ndarray:
        return metrics.abs_rel_delta


class RankerFactory:
    """Maps CLI/config ``rank_by`` strings to ranker instances."""

    _rankers: Dict[str, Type[BaseFeatureRanker]] = {
        "rel_delta": RelativeDeltaRanker,
        "abs_rel_delta": AbsoluteRelativeDeltaRanker,
    }

    @classmethod
    def get_ranker(cls, rank_by: str) -> BaseFeatureRanker:
        try:
            ctor = cls._rankers[rank_by]
        except KeyError as e:
            raise ValueError(f"Unknown ranking strategy requested: {rank_by!r}") from e
        return ctor()

    @classmethod
    def sort_records(
        cls,
        items: list[dict[str, Any]],
        rank_by: str,
        *,
        reverse: bool = True,
    ) -> None:
        """Sort audit record dicts in place by the score field for ``rank_by``."""
        ranker = cls.get_ranker(rank_by)
        key = ranker.record_field
        items.sort(key=lambda r: float(r.get(key, 0.0)), reverse=reverse)


def global_top_features(
    layer_payloads: list[dict[str, Any]],
    rank_by: str,
    top_k_global: int,
) -> list[dict[str, Any]]:
    """Cross-layer top features: flatten per-layer tops, sort by ranker, take head."""
    items: list[dict[str, Any]] = []
    for payload in layer_payloads:
        layer = payload["layer"]
        for rec in payload["top_decreased_latents"]:
            items.append({
                "layer": layer,
                "latent_idx": rec["latent_idx"],
                "rel_delta": rec.get("rel_delta", 0.0),
                "abs_rel_delta": rec.get("abs_rel_delta", 0.0),
                "mean_Y_base": rec["mean_Y_base"],
                "mean_Y_candidate": rec["mean_Y_candidate"],
                "top_contexts": rec.get("top_contexts", []),
                "top_vocab_base": rec.get("top_vocab_base", []),
                "rare_context_words": rec.get("rare_context_words", []),
            })
    RankerFactory.sort_records(items, rank_by, reverse=True)
    return items[:top_k_global]
