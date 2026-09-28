"""BLIP2 retrieval backbone with a training-only competitive TCB head."""

import torch

from blip2_maxsim.model.maxsim import BLIP2MaxSim
from blip2_maxsim.model.competitive_tcb_head import CompetitiveTCBHead
from blip2_maxsim.model.counterfactual_relational import (
    BLIP2MaxSimCounterfactualRelational,
)


class BLIP2MaxSimCompetitiveTCB(BLIP2MaxSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.competitive_tcb = CompetitiveTCBHead(self.vision_proj.out_features)

    # Reuse token encoding and the two-image forwarding interface, without
    # constructing any of the old Pair Encoder or relational MLP parameters.
    encode_reference_image = BLIP2MaxSimCounterfactualRelational.encode_reference_image
    compose_from_image_embeds = (
        BLIP2MaxSimCounterfactualRelational.compose_from_image_embeds
    )
    counterfactual_closed_loop_logits = (
        BLIP2MaxSimCounterfactualRelational.counterfactual_closed_loop_logits
    )

    def counterfactual_relational_logits(
        self,
        original_query_tokens,
        counterfactual_query_tokens,
        candidate_tokens,
        counterfactual_first,
    ):
        if (
            counterfactual_first.shape != (original_query_tokens.shape[0],)
            or counterfactual_first.dtype != torch.bool
        ):
            raise ValueError("counterfactual_first must be an aligned boolean vector")
        z = self.competitive_tcb(
            original_query_tokens, counterfactual_query_tokens, candidate_tokens
        )
        # The shared loss expects first-input-preferred logits. Because the
        # new head is antisymmetric, reversing text order is exactly this sign.
        return torch.where(counterfactual_first, z, -z)
