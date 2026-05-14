"""Logit-lens snapshot of a base model for SNMF-feature interpretation.

A feature direction ``f`` in MLP-intermediate space (``d_mlp``) contributes
``r = W_down @ f`` (``d_model``) to the residual stream at a layer's output.
The logit-lens then projects ``r`` to vocab via
``logits = lm_head( final_norm(r) )``. We snapshot the relevant HF modules /
weights so the original base model can be discarded right after activation
collection without losing the ability to do this projection.

Used by both ``experiments/audit/general_unlearning_audit.py`` and
``experiments/audit/unlearning_audit.py``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


class LogitLens:
    """CPU snapshot of the components needed to logit-lens an SNMF feature.

    Two anti-noise knobs (both on by default; toggle from the CLI):

    * **center_unembed** -- subtract the mean unembedding row from ``E`` before
      projection (``E_c = E - E.mean(0)``). Cheap fix for the well-known
      anisotropy of token embeddings: without it, almost every residual
      direction appears to "promote" the same cluster of rare-frequency
      tokens, because their embeddings all point in a shared high-norm
      direction. See Mu & Viswanath, "All-but-the-Top" (ICLR 2018).
    * **mask_special_tokens** -- set logits for special / unused / reserved
      tokens to ``-inf`` before topk. ``<bos>``, ``<eos>``, ``<pad>``,
      ``<start_of_turn>``, Gemma's ``<unused...>`` slots, etc. tend to have
      unusually large unembedding norms and dominate the topk on essentially
      arbitrary directions.

    Memory: ``lm_head`` is the dominant cost (vocab x d_model). For Gemma-2-2b
    that's ~2.4 GB fp32, manageable on CPU; we move to ``device`` only while
    projecting.
    """

    _SPECIAL_TOKEN_PATTERNS: Tuple[str, ...] = (
        "<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<sep>", "<cls>",
    )
    _SPECIAL_TOKEN_PREFIXES: Tuple[str, ...] = (
        "<unused", "<reserved", "<start_of_turn", "<end_of_turn",
        "<|", "<extra_id_",
    )

    def __init__(
        self,
        base_model: torch.nn.Module,
        layers: Sequence[int],
        *,
        tokenizer: Any = None,
        center_unembed: bool = True,
        mask_special_tokens: bool = True,
    ) -> None:
        self.final_norm = copy.deepcopy(base_model.model.norm).cpu().eval()
        self.lm_head = copy.deepcopy(base_model.lm_head).cpu().eval()
        for p in self.final_norm.parameters():
            p.requires_grad_(False)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)

        self.center_unembed = bool(center_unembed)
        if self.center_unembed:
            with torch.no_grad():
                W = self.lm_head.weight  # (vocab, d_model)
                W.sub_(W.mean(dim=0, keepdim=True))

        self.down_proj: Dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            w = base_model.model.layers[int(layer_idx)].mlp.down_proj.weight
            self.down_proj[int(layer_idx)] = w.detach().cpu().clone()

        self.special_token_ids: List[int] = []
        if mask_special_tokens and tokenizer is not None:
            self.special_token_ids = self._collect_special_token_ids(tokenizer)
        self._device: str = "cpu"

    @classmethod
    def _collect_special_token_ids(cls, tokenizer: Any) -> List[int]:
        ids: set = set()
        for tid in getattr(tokenizer, "all_special_ids", []) or []:
            if isinstance(tid, int) and tid >= 0:
                ids.add(int(tid))
        added = getattr(tokenizer, "added_tokens_decoder", None) or {}
        for tid, tok in added.items():
            text = getattr(tok, "content", None) or str(tok)
            if isinstance(text, str) and (
                text in cls._SPECIAL_TOKEN_PATTERNS
                or text.startswith(cls._SPECIAL_TOKEN_PREFIXES)
            ):
                try:
                    ids.add(int(tid))
                except (TypeError, ValueError):
                    pass
        return sorted(ids)

    def to(self, device: str) -> "LogitLens":
        self.final_norm = self.final_norm.to(device)
        self.lm_head = self.lm_head.to(device)
        self._device = device
        return self

    @torch.no_grad()
    def topk_from_residual(
        self,
        r: torch.Tensor,
        top_k: int,
        tokenizer: Any,
    ) -> List[Dict[str, Any]]:
        """Logit-lens an arbitrary residual direction ``r`` of shape ``(d_model,)``."""
        if r.ndim != 1:
            raise ValueError(f"r must be 1D, got shape {tuple(r.shape)}")
        if top_k <= 0:
            return []
        normed = self.final_norm(r.unsqueeze(0))                # (1, d_model)
        logits = self.lm_head(normed).squeeze(0)                # (vocab,)
        if self.special_token_ids:
            mask_ids = torch.tensor(self.special_token_ids,
                                    device=logits.device, dtype=torch.long)
            logits.index_fill_(0, mask_ids, float("-inf"))
        top_vals, top_ids = torch.topk(logits, top_k)
        ids = top_ids.detach().cpu().tolist()
        vals = top_vals.detach().cpu().tolist()
        return [
            {"token_id": int(tid), "token": tokenizer.decode([int(tid)]), "logit": float(v)}
            for tid, v in zip(ids, vals)
        ]

    @torch.no_grad()
    def feature_residual(
        self,
        F: torch.Tensor,
        layer: int,
        latent_indices: Sequence[int],
        weights: Optional[Sequence[float]] = None,
    ) -> torch.Tensor:
        """Return the residual-stream contribution ``W_down @ Σ_i w_i F[:, i]``.

        ``F`` is expected to be ``(d_mlp, K)`` in CPU memory; we move what we
        need to ``self._device`` and return a 1D tensor of shape ``(d_model,)``
        on that device.
        """
        if not latent_indices:
            raise ValueError("latent_indices must be non-empty")
        device = self._device
        W_down = self.down_proj[int(layer)].to(device)          # (d_model, d_mlp)
        idx = list(int(i) for i in latent_indices)
        cols = F[:, idx].to(device=device, dtype=W_down.dtype)  # (d_mlp, n)
        if weights is None:
            f_sum = cols.sum(dim=1)                             # (d_mlp,)
        else:
            if len(weights) != len(idx):
                raise ValueError(
                    f"weights length {len(weights)} != latent_indices length {len(idx)}"
                )
            w = torch.tensor(list(weights), device=device, dtype=W_down.dtype)
            f_sum = cols @ w                                    # (d_mlp,)
        return W_down @ f_sum                                   # (d_model,)

    @torch.no_grad()
    def project_latents(
        self,
        F: torch.Tensor,
        layer: int,
        latent_indices: Sequence[int],
        top_k: int,
        tokenizer: Any,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Logit-lens each latent in ``F`` (shape ``(d_mlp, K)``) at ``layer``.

        Returns a dict mapping latent index -> list of ``{token_id, token,
        logit}`` dicts of length up to ``top_k``.
        """
        if top_k <= 0 or not latent_indices:
            return {}
        device = self._device
        W_down = self.down_proj[int(layer)].to(device)        # (d_model, d_mlp)
        F_dev = F.to(device=device, dtype=W_down.dtype)        # (d_mlp, K)

        mask_ids: Optional[torch.Tensor] = None
        if self.special_token_ids:
            mask_ids = torch.tensor(self.special_token_ids,
                                    device=device, dtype=torch.long)

        out: Dict[int, List[Dict[str, Any]]] = {}
        for li in latent_indices:
            li_int = int(li)
            f = F_dev[:, li_int]                               # (d_mlp,)
            r = W_down @ f                                      # (d_model,)
            normed = self.final_norm(r.unsqueeze(0))            # (1, d_model)
            logits = self.lm_head(normed).squeeze(0)            # (vocab,)
            if mask_ids is not None:
                logits.index_fill_(0, mask_ids, float("-inf"))
            top_vals, top_ids = torch.topk(logits, top_k)
            ids = top_ids.detach().cpu().tolist()
            vals = top_vals.detach().cpu().tolist()
            out[li_int] = [
                {
                    "token_id": int(tid),
                    "token": tokenizer.decode([int(tid)]),
                    "logit": float(v),
                }
                for tid, v in zip(ids, vals)
            ]
        return out
