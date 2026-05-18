"""Ridge projection onto an SNMF basis and reconstruction residuals."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

from experiments.audit.context_windows import _sample_id_to_spans


@torch.inference_mode()
def project_onto_basis(
    A: torch.Tensor,
    Z: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """
    A : (n_tokens, d_mlp) activations from some model on the audit prompts.
    Z : (d_mlp, K)        SNMF feature basis from M_base (saved as F in snmf_factors.pt).

    Returns Y of shape (K, n_tokens) so that Z @ Y ~= A.T.
    Coefficients are unconstrained (diagnostic projection on a fixed basis, not SNMF refit).
    """
    if A.ndim != 2 or Z.ndim != 2:
        raise ValueError(f"A and Z must be 2D, got A={tuple(A.shape)} Z={tuple(Z.shape)}")
    d_mlp_a, K = Z.shape
    if A.shape[1] != d_mlp_a:
        raise ValueError(
            f"A has d_mlp={A.shape[1]} but Z expects d_mlp={d_mlp_a}; "
            "check --mode and --snmf-dir."
        )

    Z64 = Z.to(dtype=torch.float64)
    A64 = A.to(dtype=torch.float64)

    G = Z64.T @ Z64 + ridge_lambda * torch.eye(K, dtype=torch.float64, device=Z.device)
    rhs = Z64.T @ A64.T
    Y = torch.linalg.solve(G, rhs)
    return Y.to(dtype=torch.float32)


def per_prompt_peaks(
    Y: torch.Tensor, sample_ids: Sequence[int]
) -> Tuple[np.ndarray, List[int]]:
    """Reduce token-level coefficients to per-prompt peaks (max over tokens).

    Y          : (K, n_tokens)
    sample_ids : len n_tokens, parallel to Y's columns.

    Returns (Y_max, sample_ids_list) where Y_max is (n_prompts, K) np.float64 and
    sample_ids_list is the sorted unique sample ids (length n_prompts).
    """
    sample_ids_arr = np.asarray(sample_ids)
    spans = _sample_id_to_spans(sample_ids_arr)
    sample_ids_list = list(spans.keys())
    Y_np = Y.detach().cpu().numpy().astype(np.float64, copy=False)
    K, _n_tokens = Y_np.shape
    n_prompts = len(sample_ids_list)
    Y_max = np.empty((n_prompts, K), dtype=np.float64)
    for i, sid in enumerate(sample_ids_list):
        s, e = spans[sid]
        Y_max[i, :] = Y_np[:, s:e].max(axis=1)

    return Y_max, sample_ids_list


@torch.inference_mode()
def frob_relative_residual(A: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor) -> float:
    """|| A - (Z Y)^T ||_F^2 / || A ||_F^2. Returns a scalar float."""
    A = A.to(dtype=torch.float64)
    Z = Z.to(dtype=torch.float64)
    Y = Y.to(dtype=torch.float64)

    recon = (Z @ Y).T
    diff = A - recon

    num = float((diff * diff).sum().item())
    den = float((A * A).sum().item()) + 1e-12
    return num / den


class SubspaceProjector:
    """Holds ridge λ for repeated SNMF-basis projections (audit subspace)."""

    __slots__ = ("ridge_lambda",)

    def __init__(self, ridge_lambda: float) -> None:
        self.ridge_lambda = ridge_lambda

    def project_onto_basis(self, A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        return project_onto_basis(A, Z, self.ridge_lambda)

    def compute_frobenius_residual(self, A: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor) -> float:
        """Relative Frobenius reconstruction error ||A - ZY^T||_F^2 / ||A||_F^2."""
        return frob_relative_residual(A, Z, Y)
