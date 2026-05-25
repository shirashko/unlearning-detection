from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

import torch

from experiments.audit.special_tokens import (
    LOGIT_LENS_SPECIAL_TOKEN_PATTERNS,
    LOGIT_LENS_SPECIAL_TOKEN_PREFIXES,
)


class LogitLens:
    """Logit-Lens utility for interpreting decoupled sub-network latents (e.g., sNMF/SAE features).

    Mathematical Pipeline:
        1. Residual Contribution:  r = W_down @ f
        2. Vocabulary Projection:  logits = LM_head( LN_final(r) )

    Anti-Noise Mechanisms:
        * center_unembed (Default: True): Subtracts the mean token embedding row from the unembedding matrix. 
          Mitigates representation anisotropy (token embedding degeneration), suppressing generic, 
          high-norm rare tokens that mask true semantic vectors. See Mu & Viswanath (ICLR 2018).
        * mask_special_tokens (Default: True): Sets logits of control/structural tokens (<bos>, <eos>, 
          Gemma turn tags) to -inf. Prevents outlier unembedding weights from dominating top-k indices.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        layers: Sequence[int],
        *,
        tokenizer: Any = None,
        center_unembed: bool = True,
        mask_special_tokens: bool = True,
    ) -> None:
        # Independent copies on CPU so the original model can be freed.
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
            self.down_proj[int(layer_idx)] = w.detach().cpu().clone() # (d_model, d_mlp).

        self.special_token_ids: List[int] = []
        if mask_special_tokens and tokenizer is not None:
            self.special_token_ids = self._collect_special_token_ids(tokenizer)
        self._device: str = "cpu"

    @classmethod
    def _collect_special_token_ids(cls, tokenizer: Any) -> List[int]:
        ids: set = set()
        # Whatever the tokenizer itself marks as special.
        for tid in getattr(tokenizer, "all_special_ids", []) or []:
            if isinstance(tid, int) and tid >= 0:
                ids.add(int(tid))
        added = getattr(tokenizer, "added_tokens_decoder", None) or {}
        for tid, tok in added.items():
            text = getattr(tok, "content", None) or str(tok)
            if isinstance(text, str) and (
                text in LOGIT_LENS_SPECIAL_TOKEN_PATTERNS
                or text.startswith(LOGIT_LENS_SPECIAL_TOKEN_PREFIXES)
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
        """Logit-lens an arbitrary residual-stream direction ``r`` (shape ``(d_model,)``).

        Applies ``final_norm`` + (possibly centered) ``lm_head`` + special-token mask,
        same path as ``project_latents``, returning top-k
        ``{token_id, token, logit}`` entries.
        """
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
        """Operate on ``F`` (shape ``(d_mlp, K)``) at ``layer`` across a vectorized batch.

        Returns a dict mapping latent index -> list of ``{token_id, token,
        logit}`` dicts of length up to ``top_k``.
        """
        if top_k <= 0 or not latent_indices:
            return {}
            
        device = self._device
        W_down = self.down_proj[int(layer)].to(device)        # (d_model, d_mlp)
        idx = list(int(i) for i in latent_indices)
        
        # Extract all requested latent columns simultaneously
        f_batch = F[:, idx].to(device=device, dtype=W_down.dtype) # (d_mlp, num_latents)
        
        # Project all to residual stream -> shape: (num_latents, d_model)
        r_batch = (W_down @ f_batch).T                         
        
        # Explicitly structure as (Batch=1, Seq=num_latents, Hidden=d_model)
        # This guarantees LayerNorm/RMSNorm processes the hidden dimension exactly as it does in forward passes.
        r_batch_structured = r_batch.unsqueeze(0)              # (1, num_latents, d_model)
        normed = self.final_norm(r_batch_structured).squeeze(0) # (num_latents, d_model)
        
        # Project to vocabulary logits
        logits_batch = self.lm_head(normed)                    # (num_latents, vocab)
        
        # Batched index fill for special tokens
        if self.special_token_ids:
            mask_ids = torch.tensor(self.special_token_ids, device=device, dtype=torch.long)
            logits_batch.index_fill_(1, mask_ids, float("-inf"))
            
        # Batched top-k
        top_vals, top_ids = torch.topk(logits_batch, top_k, dim=1)
        
        # Move everything to CPU at once
        top_ids_cpu = top_ids.cpu().tolist()
        top_vals_cpu = top_vals.cpu().tolist()
        
        out: Dict[int, List[Dict[str, Any]]] = {}
        for b_idx, li_int in enumerate(idx):
            out[li_int] = [
                {
                    "token_id": int(tid),
                    "token": tokenizer.decode([int(tid)]),
                    "logit": float(v),
                }
                for tid, v in zip(top_ids_cpu[b_idx], top_vals_cpu[b_idx])
            ]
        return out