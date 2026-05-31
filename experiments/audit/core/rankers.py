"""Abstract and concrete strategies for ranking latent features in label-free audits."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Type

import numpy as np

# Minimum positive scale for rel_delta denominators: rank uses max(mean_base, eps), not mean_base+eps,
# so near-zero or (under ridge projection) slightly negative base means cannot invert or explode the ratio.
REL_DELTA_EPS: float = 1e-9


class LatentUnlearningMetrics(NamedTuple):
    """Per-latent statistics comparing base vs candidate SNMF coefficient profiles."""

    rel_delta: np.ndarray
    abs_rel_delta: np.ndarray
    peak_profile_l2: np.ndarray
    peak_profile_cosine_dist: np.ndarray
    normalized_peak_profile_l2: np.ndarray


# Backward-compatible alias used in type hints across the audit pipeline.
MeanPeakUnlearningMetrics = LatentUnlearningMetrics


def compute_mean_peak_metrics(
    mean_base: np.ndarray,
    mean_candidate: np.ndarray,
    *,
    eps: float = REL_DELTA_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Fractional change vs M_base with denominator max(mean_base, eps)."""
    mean_b = np.asarray(mean_base)
    mean_c = np.asarray(mean_candidate)
    raw = mean_b - mean_c
    denom = np.maximum(mean_b, eps)
    rel_delta = raw / denom
    abs_rel_delta = np.abs(rel_delta)
    return rel_delta, abs_rel_delta


def compute_latent_unlearning_metrics(
    Y_base_max: np.ndarray,
    Y_cand_max: np.ndarray,
    *,
    eps: float = REL_DELTA_EPS,
) -> LatentUnlearningMetrics:
    """
    Compare per-latent base vs candidate statistics on aligned per-prompt peak profiles.

    ``Y_*_max`` have shape ``(n_prompts, K)``. Besides mean-peak ``rel_delta``, also
    computes distances between the full per-prompt activation vectors so redistribution
    across prompts is visible even when the mean peak is unchanged.
    """
    base = np.asarray(Y_base_max, dtype=np.float64)
    cand = np.asarray(Y_cand_max, dtype=np.float64)
    if base.shape != cand.shape:
        raise ValueError(
            f"Y_base_max and Y_cand_max must share shape, got {base.shape} vs {cand.shape}"
        )

    rel_delta, abs_rel_delta = compute_mean_peak_metrics(
        base.mean(axis=0), cand.mean(axis=0), eps=eps,
    )

    diff = base - cand
    peak_profile_l2 = np.linalg.norm(diff, axis=0)

    base_norm = np.linalg.norm(base, axis=0)
    normalized_peak_profile_l2 = peak_profile_l2 / (base_norm + eps)

    cand_norm = np.linalg.norm(cand, axis=0)
    dot = np.sum(base * cand, axis=0)
    cos_sim = dot / (base_norm * cand_norm + eps)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    peak_profile_cosine_dist = 1.0 - cos_sim

    return LatentUnlearningMetrics(
        rel_delta=rel_delta,
        abs_rel_delta=abs_rel_delta,
        peak_profile_l2=peak_profile_l2,
        peak_profile_cosine_dist=peak_profile_cosine_dist,
        normalized_peak_profile_l2=normalized_peak_profile_l2,
    )


class BaseFeatureRanker(ABC):
    """Strategy for ranking latent features from base vs candidate mean-peak stats."""

    @property
    @abstractmethod
    def record_field(self) -> str:
        """Key in per-latent JSON records used for ordering (must match exported columns)."""

    @abstractmethod
    def ranking_vector(self, metrics: LatentUnlearningMetrics) -> np.ndarray:
        """Scalar score per latent; higher means more prioritized for the audit top-K."""

    def compute_scores(
        self,
        Y_base_max: np.ndarray,
        Y_cand_max: np.ndarray,
        *,
        eps: float = REL_DELTA_EPS,
    ) -> np.ndarray:
        """Convenience: full metrics tuple then strategy-specific ranking vector."""
        m = compute_latent_unlearning_metrics(Y_base_max, Y_cand_max, eps=eps)
        return self.ranking_vector(m)


class RelativeDeltaRanker(BaseFeatureRanker):
    """Fractional decay vs M_base: (mean_base - mean_cand) / max(mean_base, eps)."""

    @property
    def record_field(self) -> str:
        return "rel_delta"

    def ranking_vector(self, metrics: LatentUnlearningMetrics) -> np.ndarray:
        return metrics.rel_delta


class AbsoluteRelativeDeltaRanker(BaseFeatureRanker):
    """Magnitude of fractional change: abs(rel_delta)."""

    @property
    def record_field(self) -> str:
        return "abs_rel_delta"

    def ranking_vector(self, metrics: LatentUnlearningMetrics) -> np.ndarray:
        return metrics.abs_rel_delta


class PeakProfileL2Ranker(BaseFeatureRanker):
    """L2 distance between per-prompt peak activation profiles (base vs candidate)."""

    @property
    def record_field(self) -> str:
        return "peak_profile_l2"

    def ranking_vector(self, metrics: LatentUnlearningMetrics) -> np.ndarray:
        return metrics.peak_profile_l2


class PeakProfileCosineDistRanker(BaseFeatureRanker):
    """Cosine distance (1 - cos_sim) between per-prompt peak activation profiles."""

    @property
    def record_field(self) -> str:
        return "peak_profile_cosine_dist"

    def ranking_vector(self, metrics: LatentUnlearningMetrics) -> np.ndarray:
        return metrics.peak_profile_cosine_dist


class NormalizedPeakProfileL2Ranker(BaseFeatureRanker):
    """L2 profile distance normalized by ||Y_base_max[:, i]||_2 + eps (fractional energy shift)."""

    @property
    def record_field(self) -> str:
        return "normalized_peak_profile_l2"

    def ranking_vector(self, metrics: LatentUnlearningMetrics) -> np.ndarray:
        return metrics.normalized_peak_profile_l2


class RankerFactory:
    """Maps CLI/config ``rank_by`` strings to ranker instances."""

    _rankers: Dict[str, Type[BaseFeatureRanker]] = {
        "rel_delta": RelativeDeltaRanker,
        "abs_rel_delta": AbsoluteRelativeDeltaRanker,
        "peak_profile_l2": PeakProfileL2Ranker,
        "peak_profile_cosine_dist": PeakProfileCosineDistRanker,
        "normalized_peak_profile_l2": NormalizedPeakProfileL2Ranker,
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
                "peak_profile_l2": rec.get("peak_profile_l2", 0.0),
                "peak_profile_cosine_dist": rec.get("peak_profile_cosine_dist", 0.0),
                "normalized_peak_profile_l2": rec.get("normalized_peak_profile_l2", 0.0),
                "mean_Y_base": rec["mean_Y_base"],
                "mean_Y_candidate": rec["mean_Y_candidate"],
                "top_contexts": rec.get("top_contexts", []),
                "top_vocab_base": rec.get("top_vocab_base", []),
                "rare_context_words": rec.get("rare_context_words", []),
            })
    RankerFactory.sort_records(items, rank_by, reverse=True)
    return items[:top_k_global]
