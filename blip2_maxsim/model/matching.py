"""ReTrack-style token MaxSim scoring without any auxiliary branch."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def normalize_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """L2-normalize token features so dot products are cosine similarities."""
    return F.normalize(tokens.float(), dim=-1)


def maxsim_logits(
    query_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    token_mat_weight: torch.Tensor,
    eps_temperature: float = 1.0e-2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable in-batch MaxSim logits.

    Args:
        query_tokens: ``[B, Tq, D]`` composed-query tokens.
        target_tokens: ``[N, Tt, D]`` target-image tokens.
        token_mat_weight: learnable ``[Tq, Tq]`` token aggregation matrix.
        eps_temperature: sharpening temperature used by the original ReTrack
            token aggregation.
    """
    if query_tokens.ndim != 3 or target_tokens.ndim != 3:
        raise ValueError(
            "query_tokens and target_tokens must be 3D, got "
            f"{tuple(query_tokens.shape)} and {tuple(target_tokens.shape)}"
        )
    if query_tokens.shape[-1] != target_tokens.shape[-1]:
        raise ValueError(
            f"feature dim mismatch: {tuple(query_tokens.shape)} vs {tuple(target_tokens.shape)}"
        )

    query = normalize_tokens(query_tokens)
    target = normalize_tokens(target_tokens)
    pair_sim = torch.einsum("bqd,nkd->bnqk", query, target)
    per_token_scores = pair_sim.max(dim=-1).values
    token_weights = torch.softmax(
        torch.matmul(
            torch.softmax(per_token_scores / float(eps_temperature), dim=-1),
            token_mat_weight,
        )
        / float(eps_temperature),
        dim=-1,
    )
    logits = torch.sum(per_token_scores * token_weights, dim=-1)
    return logits, per_token_scores


def maxsim_logits_low_memory(
    query_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    token_mat_weight: torch.Tensor,
    eps_temperature: float = 1.0e-2,
) -> torch.Tensor:
    """Compute MaxSim scores one query at a time for full-bank evaluation."""
    if query_tokens.ndim != 3 or target_tokens.ndim != 3:
        raise ValueError(
            "query_tokens and target_tokens must be 3D, got "
            f"{tuple(query_tokens.shape)} and {tuple(target_tokens.shape)}"
        )
    if query_tokens.shape[-1] != target_tokens.shape[-1]:
        raise ValueError(
            f"feature dim mismatch: {tuple(query_tokens.shape)} vs {tuple(target_tokens.shape)}"
        )

    query = normalize_tokens(query_tokens)
    target_transposed = normalize_tokens(target_tokens).permute(0, 2, 1).contiguous()
    rows = []
    for index in range(query.shape[0]):
        single_query = query[index].unsqueeze(0).unsqueeze(1)
        similarity = torch.matmul(single_query, target_transposed)
        per_token_scores = similarity.max(dim=-1).values
        token_weights = torch.softmax(
            torch.matmul(
                torch.softmax(per_token_scores / float(eps_temperature), dim=-1),
                token_mat_weight,
            )
            / float(eps_temperature),
            dim=-1,
        )
        rows.append(torch.sum(per_token_scores * token_weights, dim=-1))
    return torch.cat(rows, dim=0)


def aligned_maxsim_scores(
    query_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
    token_mat_weight: torch.Tensor,
    eps_temperature: float = 1.0e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each query against its own ordered candidate list.

    Args:
        query_tokens: composed-query tokens with shape ``[B, Tq, D]``.
        candidate_tokens: candidate tokens with shape ``[B, K, Tt, D]``.
        token_mat_weight: shared learnable MaxSim aggregation matrix.
        eps_temperature: sharpening temperature used by the baseline MaxSim.

    Returns:
        A ``[B, K]`` score matrix and the pre-aggregation ``[B, K, Tq]``
        per-query-token maxima.  Unlike :func:`maxsim_logits`, this function
        never forms cross-query candidate pairs.
    """
    if query_tokens.ndim != 3 or candidate_tokens.ndim != 4:
        raise ValueError(
            "query_tokens and candidate_tokens must be 3D/4D, got "
            f"{tuple(query_tokens.shape)} and {tuple(candidate_tokens.shape)}"
        )
    if query_tokens.shape[0] != candidate_tokens.shape[0]:
        raise ValueError("Aligned MaxSim requires matching query batch dimensions")
    if query_tokens.shape[-1] != candidate_tokens.shape[-1]:
        raise ValueError("Aligned MaxSim feature dimensions do not match")

    query = normalize_tokens(query_tokens)
    candidate = normalize_tokens(candidate_tokens)
    pair_sim = torch.einsum("bqd,bnkd->bnqk", query, candidate)
    per_token_scores = pair_sim.max(dim=-1).values
    token_weights = torch.softmax(
        torch.matmul(
            torch.softmax(per_token_scores / float(eps_temperature), dim=-1),
            token_mat_weight,
        )
        / float(eps_temperature),
        dim=-1,
    )
    scores = torch.sum(per_token_scores * token_weights, dim=-1)
    return scores, per_token_scores


# Compatibility aliases make old warmup analysis snippets easy to migrate while
# keeping this package fully self-contained.
retrack_token_logits = maxsim_logits
retrack_token_logits_low_memory = maxsim_logits_low_memory
