"""Core audit primitives: subspace projection, ranking strategies, per-layer orchestration."""

from experiments.audit.core.layer_auditor import LayerAuditor
from experiments.audit.core.projection import (
    SubspaceProjector,
    frob_relative_residual,
    per_prompt_peaks,
    project_onto_basis,
)
from experiments.audit.core.rankers import (
    REL_DELTA_EPS,
    BaseFeatureRanker,
    LatentUnlearningMetrics,
    MeanPeakUnlearningMetrics,
    RankerFactory,
    compute_latent_unlearning_metrics,
    compute_mean_peak_metrics,
    global_top_features,
)

__all__ = [
    "BaseFeatureRanker",
    "LayerAuditor",
    "LatentUnlearningMetrics",
    "MeanPeakUnlearningMetrics",
    "RankerFactory",
    "REL_DELTA_EPS",
    "SubspaceProjector",
    "compute_latent_unlearning_metrics",
    "compute_mean_peak_metrics",
    "frob_relative_residual",
    "global_top_features",
    "per_prompt_peaks",
    "project_onto_basis",
]
