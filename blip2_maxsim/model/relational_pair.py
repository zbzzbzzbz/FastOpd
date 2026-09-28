"""Candidate-conditioned relational pair branch for BLIP-2 MaxSim.

The branch intentionally consumes the shared BLIP-2 query/candidate tokens
*before* MaxSim aggregation.  Consequently its loss can update the shared
token encoder, Pair Encoder, and relational head, but has no computational
path to ``compose_token_mat_weight``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from blip2_maxsim.model.maxsim import BLIP2MaxSim


class ScalarMLP(nn.Module):
    """LayerNorm/ReLU scalar head used by the relational branch."""

    def __init__(self, width: int, layers: int = 3) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for _ in range(int(layers)):
            modules.extend((nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU()))
        modules.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value).squeeze(-1)


class PairEncoderBlock(nn.Module):
    """One full-interaction Transformer block over ``[Q32; C32]``."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(sequence)
        sequence = (
            sequence
            + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        )
        return sequence + self.ffn(self.ffn_norm(sequence))


class BLIP2MaxSimRelationalPair(BLIP2MaxSim):
    """BLIP-2 MaxSim with a separate, training-only relational pair head.

    Retrieval continues to use the inherited MaxSim scorer.  Pair labels are
    predicted on a separate path:

      shared Q/C tokens -> Pair Encoder -> Relational Branch -> pair logit.

    No tensors are detached: relational gradients reach the shared Q-Former
    and projections.  The MaxSim aggregation matrix is absent from this path.
    """

    def __init__(
        self,
        *,
        pair_encoder_layers: int = 2,
        pair_attention_heads: int = 8,
        relational_tokens: int = 16,
        pair_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        width = int(self.vision_proj.out_features)
        if width % int(pair_attention_heads):
            raise ValueError(
                "embedding width must be divisible by pair_attention_heads"
            )
        if int(pair_encoder_layers) <= 0:
            raise ValueError("pair_encoder_layers must be positive")
        if int(relational_tokens) <= 0:
            raise ValueError("relational_tokens must be positive")

        self.pair_width = width
        self.pair_attention_heads = int(pair_attention_heads)
        self.pair_head_dim = width // self.pair_attention_heads

        # Query/candidate roles are required; candidate rank embeddings are
        # deliberately absent so the pair head cannot read the input order.
        self.pair_role_embeddings = nn.Parameter(torch.empty(2, width))
        self.pair_encoder_blocks = nn.ModuleList(
            [
                PairEncoderBlock(width, self.pair_attention_heads, pair_dropout)
                for _ in range(int(pair_encoder_layers))
            ]
        )
        self.pair_output_norm = nn.LayerNorm(width)

        self.relational_tokens = nn.Parameter(
            torch.empty(int(relational_tokens), width)
        )
        self.rel_query_projection = nn.Linear(width, width)
        self.rel_query_norm = nn.LayerNorm(width)
        self.rel_kv_projection = nn.Linear(width, width)
        self.rel_kv_norm = nn.LayerNorm(width)
        self.rel_output_projection = nn.Linear(width, width)
        self.relational_head = ScalarMLP(width, layers=3)

        nn.init.normal_(self.pair_role_embeddings, std=0.02)
        nn.init.normal_(self.relational_tokens, std=width**-0.5)

    def pair_condition(
        self,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Condition each candidate independently on the same query.

        Args:
            query_tokens: ``[B, 32, D]``.
            candidate_tokens: ``[B, N, 32, D]``.

        Returns:
            Candidate-conditioned query and query-conditioned candidate tokens,
            both with shape ``[B, N, 32, D]``.
        """
        if query_tokens.ndim != 3 or candidate_tokens.ndim != 4:
            raise ValueError("Pair Encoder expects query [B,T,D], candidates [B,N,T,D]")
        batch, candidates, token_count, width = candidate_tokens.shape
        if tuple(query_tokens.shape) != (batch, token_count, width):
            raise ValueError(
                "query/candidate token shapes are incompatible: "
                f"{tuple(query_tokens.shape)} vs {tuple(candidate_tokens.shape)}"
            )
        if width != self.pair_width:
            raise ValueError(f"expected token width {self.pair_width}, got {width}")

        expanded_query = query_tokens[:, None].expand(-1, candidates, -1, -1)
        expanded_query = expanded_query + self.pair_role_embeddings[0][None, None, None]
        conditioned_input = (
            candidate_tokens + self.pair_role_embeddings[1][None, None, None]
        )
        sequence = torch.cat((expanded_query, conditioned_input), dim=2).reshape(
            batch * candidates, token_count * 2, width
        )
        for block in self.pair_encoder_blocks:
            sequence = block(sequence)
        sequence = self.pair_output_norm(sequence)
        conditioned_query = sequence[:, :token_count].view(
            batch, candidates, token_count, width
        )
        conditioned_candidate = sequence[:, token_count:].view(
            batch, candidates, token_count, width
        )
        return conditioned_query, conditioned_candidate

    def relational_scores(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        """Return one directional preference logit per conditioned pair."""
        if first.ndim != 3 or tuple(second.shape) != tuple(first.shape):
            raise ValueError("relational inputs must have matching [B,T,D] shapes")
        batch, token_count, width = first.shape
        if width != self.pair_width:
            raise ValueError(
                f"expected relational width {self.pair_width}, got {width}"
            )

        pair = torch.cat((first, second), dim=1)
        relation_queries = self.rel_query_norm(
            self.rel_query_projection(self.relational_tokens)
        )
        relation_queries = relation_queries.view(
            1, -1, self.pair_attention_heads, self.pair_head_dim
        ).transpose(1, 2)
        relation_queries = relation_queries.expand(batch, -1, -1, -1)

        key_values = self.rel_kv_norm(self.rel_kv_projection(pair))
        key_values = key_values.view(
            batch,
            token_count * 2,
            self.pair_attention_heads,
            self.pair_head_dim,
        ).transpose(1, 2)
        attention = torch.softmax(
            torch.einsum("bhqd,bhpd->bhqp", relation_queries, key_values)
            / math.sqrt(self.pair_head_dim),
            dim=-1,
        )
        output_first = torch.einsum(
            "bhqp,bhpd->bhqd",
            attention[..., :token_count],
            key_values[..., :token_count, :],
        )
        output_second = torch.einsum(
            "bhqp,bhpd->bhqd",
            attention[..., token_count:],
            key_values[..., token_count:, :],
        )
        difference = (output_first - output_second).transpose(1, 2).contiguous()
        difference = difference.view(batch, self.relational_tokens.shape[0], width)
        pooled = self.rel_output_projection(difference).sum(dim=1)
        return self.relational_head(pooled)

    def relational_pair_logits(
        self,
        query_tokens: torch.Tensor,
        first_candidate_tokens: torch.Tensor,
        second_candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Condition two candidates independently, then compare their halves."""
        if tuple(first_candidate_tokens.shape) != tuple(second_candidate_tokens.shape):
            raise ValueError("first/second candidate token shapes differ")
        candidates = torch.stack(
            (first_candidate_tokens, second_candidate_tokens), dim=1
        )
        _, conditioned_candidates = self.pair_condition(query_tokens, candidates)
        return self.relational_scores(
            conditioned_candidates[:, 0], conditioned_candidates[:, 1]
        )


__all__ = [
    "BLIP2MaxSimRelationalPair",
    "PairEncoderBlock",
    "ScalarMLP",
]
