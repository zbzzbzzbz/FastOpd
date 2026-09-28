"""Equal-weight candidate BCE and masked candidate-versus-GT logit margin."""

import math
import torch
import torch.nn.functional as F


def relative_gt_pair_losses(
    selected_logits, selected_targets, pair_is_gt, gt_logits_for_non_gt, *, margin=1.0
):
    """Both forwards must use the SAME text orientation for each pair.

    Targets indicate whether CF is first. Returned z values always mean
    positive = CF preferred. GT-selected pairs keep unscaled candidate BCE;
    other pairs average candidate BCE and hinge. Both logits receive gradients.
    """
    if not math.isfinite(float(margin)) or margin <= 0:
        raise ValueError("relative margin must be finite and positive")
    if selected_logits.ndim != 1 or selected_targets.shape != selected_logits.shape:
        raise ValueError("expected aligned 1D logits and targets")
    if pair_is_gt.shape != selected_logits.shape or pair_is_gt.dtype != torch.bool:
        raise ValueError("expected aligned boolean GT mask")
    if not bool(((selected_targets == 0) | (selected_targets == 1)).all()):
        raise ValueError("targets must encode binary text orientation")
    non_gt = ~pair_is_gt
    if gt_logits_for_non_gt.shape != (int(non_gt.sum()),):
        raise ValueError("GT logits must match non-GT pairs in order")
    targets = selected_targets.float()
    candidate_loss = F.binary_cross_entropy_with_logits(
        selected_logits.float(), targets, reduction="none"
    )
    direction = (2.0 * targets - 1.0)[non_gt]
    z_c = selected_logits.float()[non_gt] * direction
    z_g = gt_logits_for_non_gt.float() * direction
    relative_loss = F.relu(float(margin) - z_c + z_g)
    pair_loss = candidate_loss.clone()
    pair_loss[non_gt] = 0.5 * (candidate_loss[non_gt] + relative_loss)
    return pair_loss, candidate_loss, relative_loss, z_c, z_g


def relative_gt_metrics(relative_loss, z_c, z_g, *, margin=1.0):
    """Separate margin satisfaction from binary candidate accuracy."""
    count = relative_loss.numel()

    def mean(value):
        return float(value.detach().float().mean()) if count else 0.0

    gap = z_c - z_g
    return {
        "relative_pairs": count,
        "relative_loss": mean(relative_loss),
        "relative_margin_satisfied_rate": mean(gap >= margin),
        "relative_active_rate": mean(gap < margin),
        "relative_z_c": mean(z_c),
        "relative_z_g": mean(z_g),
        "relative_gap": mean(gap),
    }
