"""Per-layer projection + mean-peak metrics, kept out of the CLI script global scope."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from experiments.audit.config import AuditConfig
from experiments.audit.core.projection import (
    SubspaceProjector,
    per_prompt_peaks,
)
from experiments.audit.core.metric_format import round_audit_metric
from experiments.audit.core.rankers import (
    REL_DELTA_EPS,
    RankerFactory,
    LatentUnlearningMetrics,
    compute_latent_unlearning_metrics,
)


class LayerAuditor:
    """Orchestrates ridge projection, residuals, and latent ranking for one layer."""

    __slots__ = ("cfg", "projector", "ranker")

    def __init__(self, cfg: AuditConfig, projector: SubspaceProjector) -> None:
        self.cfg = cfg
        self.projector = projector
        self.ranker = RankerFactory.get_ranker(cfg.snmf.rank_by)

    def audit_layer(
        self,
        layer_idx: int,
        Z: torch.Tensor,
        A_base: torch.Tensor,
        A_cand: torch.Tensor,
        sample_ids: List[int],
    ) -> Dict[str, Any]:
        """Project, Frobenius residuals, mean-peak metrics, and top-K latent indices."""
        Y_base = self.projector.project_onto_basis(A_base, Z)
        Y_cand = self.projector.project_onto_basis(A_cand, Z)

        res_base = self.projector.compute_frobenius_residual(A_base, Z, Y_base)
        res_cand = self.projector.compute_frobenius_residual(A_cand, Z, Y_cand)

        Y_base_max, sample_ids_list = per_prompt_peaks(Y_base, sample_ids)
        Y_cand_max, _ = per_prompt_peaks(Y_cand, sample_ids)

        mean_base = Y_base_max.mean(axis=0)
        mean_cand = Y_cand_max.mean(axis=0)
        m = compute_latent_unlearning_metrics(Y_base_max, Y_cand_max, eps=REL_DELTA_EPS)
        scores = self.ranker.ranking_vector(m)

        K = int(Z.shape[1])
        n_keep = max(1, min(self.cfg.snmf.top_k_per_layer, K))
        top_indices = np.argpartition(-scores, n_keep - 1)[:n_keep]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return {
            "layer_idx": layer_idx,
            "K": K,
            "Y_base": Y_base,
            "sample_ids_list": sample_ids_list,
            "residuals": {
                "base": res_base,
                "candidate": res_cand,
                "delta": res_cand - res_base,
            },
            "mean_base": mean_base,
            "mean_candidate": mean_cand,
            "metrics": m,
            "scores": scores,
            "top_indices": top_indices,
        }

    def build_layer_payload_numeric(
        self,
        core: Dict[str, Any],
        layer_idx: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Per-latent table, CSV rows, rel_delta stats, and JSON base minus ``top_decreased_latents``."""
        m: LatentUnlearningMetrics = core["metrics"]
        rel_delta, abs_rel_delta = m.rel_delta, m.abs_rel_delta
        peak_profile_l2 = m.peak_profile_l2
        peak_profile_cosine_dist = m.peak_profile_cosine_dist
        normalized_peak_profile_l2 = m.normalized_peak_profile_l2
        mean_base = core["mean_base"]
        mean_cand = core["mean_candidate"]
        K = int(core["K"])
        sample_ids_list: List[int] = core["sample_ids_list"]
        ridge_lambda = float(self.projector.ridge_lambda)
        rank_by = self.cfg.snmf.rank_by

        per_latent: Dict[int, Dict[str, Any]] = {}
        rows: List[Dict[str, Any]] = []
        for i in range(K):
            metrics = {
                "mean_Y_base": round_audit_metric(float(mean_base[i])),
                "mean_Y_candidate": round_audit_metric(float(mean_cand[i])),
                "rel_delta": round_audit_metric(float(rel_delta[i])),
                "abs_rel_delta": round_audit_metric(float(abs_rel_delta[i])),
                "peak_profile_l2": round_audit_metric(float(peak_profile_l2[i])),
                "peak_profile_cosine_dist": round_audit_metric(
                    float(peak_profile_cosine_dist[i]),
                ),
                "normalized_peak_profile_l2": round_audit_metric(
                    float(normalized_peak_profile_l2[i]),
                ),
            }
            per_latent[i] = metrics
            rows.append({"latent_idx": i, "layer": layer_idx, **metrics})

        rel_delta_stats = {
            "mean": round_audit_metric(float(rel_delta.mean())),
            "std": round_audit_metric(float(rel_delta.std())),
            "max": round_audit_metric(float(rel_delta.max())),
            "min": round_audit_metric(float(rel_delta.min())),
            "p99": round_audit_metric(float(np.percentile(rel_delta, 99))),
            "p1": round_audit_metric(float(np.percentile(rel_delta, 1))),
            "epsilon": REL_DELTA_EPS,
        }

        res = core["residuals"]
        partial_payload: Dict[str, Any] = {
            "layer": layer_idx,
            "K": K,
            "n_prompts": int(len(sample_ids_list)),
            "ridge_lambda": ridge_lambda,
            "reconstruction_residual_relative": {
                "base": round_audit_metric(float(res["base"])),
                "candidate": round_audit_metric(float(res["candidate"])),
                "delta": round_audit_metric(float(res["delta"])),
            },
            "rel_delta_stats": rel_delta_stats,
            "rank_by": rank_by,
            "per_latent": per_latent,
        }
        return partial_payload, rows
