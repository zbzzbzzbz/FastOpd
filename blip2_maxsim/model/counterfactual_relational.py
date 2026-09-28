"""Two-query/one-candidate relational branch for offline counterfactuals."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from blip2_maxsim.model.relational_pair import BLIP2MaxSimRelationalPair


class BLIP2MaxSimCounterfactualRelational(BLIP2MaxSimRelationalPair):
    """Compare how one candidate changes under original vs counterfactual edits.

    Each query/candidate pair is processed independently by the existing Pair
    Encoder.  The relational head then compares the two conditioned candidate
    halves.  The two edit variants never see one another inside Pair Encoder.
    """

    def encode_reference_image(self, reference_image: torch.Tensor) -> torch.Tensor:
        """Run frozen EVA-ViT once so both edit variants reuse its output."""
        if reference_image.ndim != 4:
            raise ValueError(
                "counterfactual training expects reference images [B,C,H,W]"
            )
        if self.train_vit:
            image_embeds = self.visual_encoder(reference_image)
        else:
            with torch.no_grad():
                image_embeds = self.visual_encoder(reference_image)
        return self.ln_vision(image_embeds)

    def compose_from_image_embeds(
        self,
        image_embeds: torch.Tensor,
        captions,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Fuse cached-within-step reference features with a batch of edits."""
        text_tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.text_input_length,
            return_tensors="pt",
        ).to(device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        image_attention = torch.ones(
            image_embeds.shape[:-1], dtype=torch.long, device=device
        )
        query_attention = torch.ones(
            query_tokens.shape[:-1], dtype=torch.long, device=device
        )
        attention_mask = torch.cat((query_attention, text_tokens.attention_mask), dim=1)
        output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention,
            return_dict=True,
            output_attentions=False,
        )
        hidden = output.last_hidden_state[:, : query_tokens.shape[1], :]
        return F.normalize(self.text_proj(hidden), dim=-1)

    def counterfactual_relational_logits(
        self,
        original_query_tokens: torch.Tensor,
        counterfactual_query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        counterfactual_first: torch.Tensor,
    ) -> torch.Tensor:
        """Score which of two separately conditioned queries better fits C.

        The output is directional: positive means the first relational input is
        preferred.  Randomizing the input order prevents a positional shortcut.
        """
        if original_query_tokens.shape != counterfactual_query_tokens.shape:
            raise ValueError("original/counterfactual query token shapes differ")
        if original_query_tokens.shape != candidate_tokens.shape:
            raise ValueError("query/candidate token shapes differ")
        pair_count = original_query_tokens.shape[0]
        if tuple(counterfactual_first.shape) != (pair_count,):
            raise ValueError("counterfactual_first must have shape [pair_count]")

        queries = torch.stack(
            (original_query_tokens, counterfactual_query_tokens), dim=1
        ).flatten(0, 1)
        repeated_candidates = candidate_tokens[:, None].expand(-1, 2, -1, -1)
        _, conditioned = self.pair_condition(
            queries, repeated_candidates.flatten(0, 1)[:, None]
        )
        conditioned = conditioned[:, 0].reshape(
            pair_count, 2, candidate_tokens.shape[1], candidate_tokens.shape[2]
        )
        original_conditioned = conditioned[:, 0]
        counterfactual_conditioned = conditioned[:, 1]
        selector = counterfactual_first[:, None, None]
        first = torch.where(selector, counterfactual_conditioned, original_conditioned)
        second = torch.where(selector, original_conditioned, counterfactual_conditioned)
        return self.relational_scores(first, second)

    def counterfactual_closed_loop_logits(
        self,
        original_query_tokens: torch.Tensor,
        counterfactual_query_tokens: torch.Tensor,
        counterfactual_candidate_tokens: torch.Tensor,
        gt_candidate_tokens: torch.Tensor,
        counterfactual_first: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the same query order to Ccf and GT and return opposite tasks.

        Returns ``[M, 2]`` logits.  Column zero compares the two queries on the
        counterfactual candidate; column one compares them on the true target.
        The Relational Head itself is unchanged and still receives two
        conditioned candidate sequences of shape ``[M, 32, D]`` per call.
        """
        if counterfactual_candidate_tokens.shape != gt_candidate_tokens.shape:
            raise ValueError("counterfactual/GT candidate token shapes differ")
        counterfactual_candidate_logits = self.counterfactual_relational_logits(
            original_query_tokens,
            counterfactual_query_tokens,
            counterfactual_candidate_tokens,
            counterfactual_first,
        )
        gt_candidate_logits = self.counterfactual_relational_logits(
            original_query_tokens,
            counterfactual_query_tokens,
            gt_candidate_tokens,
            counterfactual_first,
        )
        return torch.stack(
            (counterfactual_candidate_logits, gt_candidate_logits), dim=1
        )


__all__ = ["BLIP2MaxSimCounterfactualRelational"]
