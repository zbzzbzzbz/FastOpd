"""Image-led, joint ori/CF attention competition; no text-value classifier."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CompetitiveTCBHead(nn.Module):
    def __init__(self, width: int = 256):
        super().__init__()
        self.width = int(width)
        self.candidate_projection = nn.Linear(width, width, bias=False)
        self.query_projection = nn.Linear(width, width, bias=False)
        self.visual_weight = nn.Linear(width, 1, bias=False)
        # Preserve the warmup token geometry; salience starts as uniform.
        nn.init.eye_(self.candidate_projection.weight)
        nn.init.eye_(self.query_projection.weight)
        nn.init.zeros_(self.visual_weight.weight)

    def forward(self, original, counterfactual, candidate, *, return_details=False):
        if original.ndim != 3 or original.shape != counterfactual.shape:
            raise ValueError("expected matching ori/CF [M,T,D] tensors")
        if (
            candidate.ndim != 3
            or candidate.shape[0] != original.shape[0]
            or candidate.shape[2] != self.width
            or original.shape[2] != self.width
            or original.shape[1] == 0
            or candidate.shape[1] == 0
        ):
            raise ValueError(
                "candidate must have matching batch/width and nonempty tokens"
            )
        # BLIP tokens are unit-normalized. RMS-unit scaling gives standard
        # dot-product-attention magnitude without adding a learned temperature.
        # Use FP32 for the small head, including stable log probabilities.
        with torch.autocast(device_type=original.device.type, enabled=False):
            c = F.normalize(candidate.float(), dim=-1)
            both = F.normalize(
                torch.cat((original, counterfactual), dim=1).float(), dim=-1
            )
            cq = self.candidate_projection(c * math.sqrt(self.width))
            qk = self.query_projection(both * math.sqrt(self.width))
            energy = torch.matmul(cq, qk.transpose(-1, -2)) / math.sqrt(self.width)
            log_attention = F.log_softmax(energy, dim=-1)  # joint ori + CF axis
            split = original.shape[1]
            log_ori = torch.logsumexp(log_attention[..., :split], dim=-1)
            log_cf = torch.logsumexp(log_attention[..., split:], dim=-1)
            support = log_cf - log_ori
            rho = F.softmax(self.visual_weight(c).squeeze(-1), dim=-1)
            canonical_logit = (rho * support).sum(dim=-1)
        if not return_details:
            return canonical_logit
        return canonical_logit, dict(
            attention=log_attention.exp(),
            p_ori=log_ori.exp(),
            p_cf=log_cf.exp(),
            visual_weights=rho,
            token_log_odds=support,
        )


def architecture_metadata(width=256):
    return {
        "name": "candidate_led_joint_attention_competition_v1",
        "width": width,
        "added_parameters": 2 * width * width + width,
        "attention": "candidate queries; jointly normalized ori/CF keys; no AV",
        "projection_bias": False,
        "input_scaling": "L2-normalized tokens multiplied by sqrt(width) before projections",
        "projection_initialization": "identity",
        "visual_pooling": "learned candidate-only softmax weights, initially uniform",
        "output": "sum_i rho_i * (log p_cf_i - log p_ori_i)",
        "head_precision": "float32",
        "pair_encoder_layers": 0,
        "relational_tokens": 0,
        "dropout": 0.0,
        "text_value_or_residual_to_head": False,
    }
