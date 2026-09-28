"""Full FashionIQ training with Qwen supervision and competitive TCB."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from blip2_maxsim.training import training_loop as base
from blip2_maxsim.training import teacher_training as qwen
from blip2_maxsim.model.matching import aligned_maxsim_scores
from blip2_maxsim.training.tcb_loss import relative_gt_pair_losses, relative_gt_metrics
from blip2_maxsim.training.student_core import append_jsonl, atomic_json, iter_jsonl
from blip2_maxsim.teacher.counterfactual import (
    prompt_version as counterfactual_prompt_version,
    request_counterfactual_chunk,
)
from blip2_maxsim.model.competitive_tcb import BLIP2MaxSimCompetitiveTCB
from blip2_maxsim.model.competitive_tcb_head import architecture_metadata


VERSION = (
    "blip2_maxsim_qwen32_demotion_rpa_competitive_tcb_v4_fixed269_relative_gtcl_v1"
)
TCB_VERSION = "competitive_tcb_joint_query_attention_fixed269_equal_relative_gt_v1"
TCB_WEIGHT = 0.1
TCB_PAIRS_PER_QUERY = 3
TCB_RELATIVE_MARGIN = 1.0

DEMOTION_LOSS_VERSION = "rpa_linear_teacher_weight_x_student_discounted_demotion_v1"
DEMOTION_BETA = 1.0


def demotion_aware_weights(
    scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    beta: float = DEMOTION_BETA,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return normalized [B, K-1] demotion-aware suffix weights."""
    if scores.ndim != 2 or teacher_scores.shape != scores.shape:
        raise ValueError("scores and teacher_scores must be aligned [B, K] tensors")
    candidate_count = int(scores.shape[1])
    if candidate_count < 2:
        raise ValueError("demotion-aware RPA requires at least two candidates")
    if float(beta) < 0.0:
        raise ValueError("demotion beta must be non-negative")

    scores_fp32 = scores.detach().float()
    teacher_fp32 = teacher_scores.detach().float()
    student_order = torch.argsort(scores_fp32, dim=1, descending=True, stable=True)
    student_ranks = torch.empty_like(student_order)
    rank_values = (
        torch.arange(
            1, candidate_count + 1, device=scores.device, dtype=student_order.dtype
        )
        .unsqueeze(0)
        .expand_as(student_order)
    )
    student_ranks.scatter_(1, student_order, rank_values)

    ideal_ranks = torch.arange(
        1, candidate_count + 1, device=scores.device, dtype=torch.float32
    ).unsqueeze(0)
    ideal_discounts = 1.0 / torch.log2(1.0 + ideal_ranks)
    student_discounts = 1.0 / torch.log2(1.0 + student_ranks.float())
    # Positive only when the teacher item is below its ideal teacher position.
    demotion = (ideal_discounts - student_discounts).clamp_min(0.0)

    base_weights = []
    for index in range(candidate_count - 1):
        teacher_gap = (
            teacher_fp32[:, index : index + 1] - teacher_fp32[:, index + 1 :]
        ).abs()
        base_weights.append(teacher_gap.mean(dim=1))
    base_matrix = torch.stack(base_weights, dim=1)
    raw_weights = base_matrix * (1.0 + float(beta) * demotion[:, : candidate_count - 1])
    return (
        raw_weights / (raw_weights.sum(dim=1, keepdim=True) + float(epsilon))
    ).detach()


def demotion_aware_rpa_loss(
    scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float,
    beta: float = DEMOTION_BETA,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Original RPA suffix likelihood with demotion-aware query weights."""
    weights = demotion_aware_weights(scores, teacher_scores, beta=beta, epsilon=epsilon)
    scaled = scores.float() / max(float(temperature), 1.0e-6)
    suffix_losses = torch.stack(
        [
            -F.log_softmax(scaled[:, index:], dim=1)[:, 0]
            for index in range(int(scores.shape[1]) - 1)
        ],
        dim=1,
    )
    return (weights * suffix_losses).sum(dim=1).mean()


def _loss_metadata() -> dict[str, object]:
    return {
        "name": DEMOTION_LOSS_VERSION,
        "base_teacher_weight": "mean_{j>r} abs(t_r-t_j), unchanged from original RPA",
        "demotion": "relu(D(r)-D(p_ir))",
        "discount": "D(p)=1/log2(1+p), with 1-indexed ranks",
        "raw_weight": "w_r * (1 + beta * demotion_ir)",
        "beta": DEMOTION_BETA,
        "rank_gradient": "detached/stop-gradient",
        "query_reduction": "normalize raw weights per query, then batch mean",
        "pair_impact_average": False,
        "gain_transform": False,
        "gt_aware_weight": False,
        "inversion_mask": False,
    }


def build_model(args, device):
    model = BLIP2MaxSimCompetitiveTCB(
        vit_model="eva_clip_g",
        image_size=int(args.image_size),
        train_vit=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        text_input_length=64,
        rank_loss_weight=1.0,
        rank_temperature=float(args.rank_temperature),
        maxsim_temperature=float(args.maxsim_temperature),
    )
    info = model.load_model_state(str(args.base_checkpoint))
    missing = [
        key for key in info["missing_keys"] if not key.startswith("competitive_tcb.")
    ]
    if missing:
        raise RuntimeError(f"Warmup checkpoint missing backbone keys: {missing[:10]}")
    if args.initial_delta is not None:
        delta = torch.load(
            str(args.initial_delta), map_location="cpu", weights_only=False
        )
        state = delta.get("trainable_state", delta.get("model_state"))
        if not isinstance(state, Mapping):
            raise RuntimeError("Initial delta has no parameter dictionary")
        loaded = model.load_state_dict(state, strict=False)
        if loaded.unexpected_keys or any(
            key.startswith("competitive_tcb.") for key in loaded.missing_keys
        ):
            raise RuntimeError(
                "Initial delta is not a compatible competitive-TCB checkpoint"
            )
    count = sum(p.numel() for p in model.competitive_tcb.parameters())
    if count != architecture_metadata()["added_parameters"]:
        raise RuntimeError("Unexpected competitive TCB parameter count")
    print("[CompetitiveTCB] " + json.dumps(architecture_metadata()), flush=True)
    return model.to(device)


def configure_training(args) -> None:
    args.listwise_variant = _loss_metadata()["name"]
    args.listwise_query_normalization = True
    args.demotion_beta = DEMOTION_BETA
    args.tcb_weight = TCB_WEIGHT
    args.tcb_pairs_per_query = TCB_PAIRS_PER_QUERY
    positions = {5: (2, 3, 5), 10: (2, 6, 9), 15: (2, 8, 14)}.get(
        int(args.teacher_list_size)
    )
    if positions is None:
        raise ValueError("competitive TCB requires teacher_list_size 5, 10, or 15")
    args.tcb_fixed_positions_override = list(positions)
    args.tcb_fixed_positions = list(positions)
    args.tcb_include_gt = True
    args.tcb_masked_gt_closed_loop = False
    args.tcb_relative_gt_loop = True
    args.tcb_relative_margin = TCB_RELATIVE_MARGIN
    args.tcb_pair_encoder_layers = 0
    args.tcb_pair_attention_heads = 1
    args.tcb_relational_tokens = 0
    args.tcb_pair_dropout = 0.0
    args.tcb_qwen_max_tokens = 192
    args.tcb_qwen_temperature = 0.2
    args.tcb_qwen_top_p = 0.9
    args.counterfactual_cache_filename = "online_qwen32_counterfactual_v4_cache.jsonl"
    if float(args.list_weight) <= 0:
        raise ValueError("TCB integration requires a positive list weight")
    args.tcb_architecture = architecture_metadata()

    args.fastopd_version = VERSION


def _orientation(query_id: str, candidate_id: str, seed: int) -> bool:
    value = f"{int(seed)}|{query_id}|{candidate_id}".encode()
    return bool(hashlib.sha256(value).digest()[0] & 1)


def compute_live_joint_loss(
    model: BLIP2MaxSimCompetitiveTCB,
    *,
    raw_batch: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    image_loader: Any,
    device: torch.device,
    temperature: float,
    args: Any,
) -> tuple[torch.Tensor, float, int, int]:
    if len(manifests) != len(results) or not manifests:
        raise ValueError("Live joint loss requires aligned non-empty manifests/results")
    category = str(manifests[0]["category"])
    source_indices = torch.tensor(
        [int(row["source_batch_index"]) for row in manifests], dtype=torch.long
    )
    references = (
        raw_batch["ref_img"]
        .index_select(0, source_indices)
        .to(device, non_blocking=True)
    )
    edits = [str(row["edit"]) for row in manifests]
    ranked_ids = [
        [str(item_id) for item_id in result["ranked_candidate_ids"]]
        for result in results
    ]
    list_size = int(args.teacher_list_size)
    if any(len(ids) != list_size or len(set(ids)) != list_size for ids in ranked_ids):
        raise RuntimeError(f"Teacher ranking is not a {list_size}-item permutation")
    flat_ids = [item_id for ids in ranked_ids for item_id in ids]
    candidate_images = image_loader.load(flat_ids, category).to(
        device, non_blocking=True
    )
    from blip2_maxsim.training.memory_probe import detail as memory_detail

    memory_detail(
        "candidate_images_loaded",
        image_shape=list(candidate_images.shape),
        candidate_image_bytes=candidate_images.numel()
        * candidate_images.element_size(),
    )

    empty = torch.empty(0, device=device)
    relative_metrics = relative_gt_metrics(
        empty, empty, empty, margin=args.tcb_relative_margin
    )
    with base.autocast_context(device):
        reference_embeds = model.encode_reference_image(references)
        query_tokens = model.compose_from_image_embeds(
            reference_embeds, edits, device=device
        )
        memory_detail(
            "query_encoded",
            reference_shape=list(references.shape),
            edit_chars=[len(x) for x in edits],
        )
        _, flat_tokens, _ = model.target_fea(candidate_images, device=device)
        memory_detail("candidate_encoded")
        candidate_tokens = flat_tokens.reshape(len(manifests), list_size, 32, 256)
        scores, _ = aligned_maxsim_scores(
            query_tokens,
            candidate_tokens,
            model.compose_token_mat_weight,
            model.maxsim_temperature,
        )
        memory_detail("maxsim_attention_scored")
        teacher_scores = torch.linspace(
            1.0, 0.0, list_size, device=device, dtype=torch.float32
        ).repeat(len(manifests), 1)
        list_loss = base.listwise_rpa_loss(scores, teacher_scores, temperature)

        owners: list[int] = []
        candidate_indices: list[int] = []
        counterfactual_texts: list[str] = []
        orientations: list[bool] = []
        pair_is_gt: list[bool] = []
        pair_teacher_ranks: list[int] = []
        for owner, (manifest, result, ids) in enumerate(
            zip(manifests, results, ranked_ids)
        ):
            counterfactual = result.get("tcb_v4")
            if not isinstance(counterfactual, Mapping):
                continue
            cf_ids = [str(value) for value in counterfactual["candidate_ids"]]
            cf_texts = [str(value) for value in counterfactual["counterfactual_texts"]]
            cf_ranks = [
                int(value) for value in counterfactual["candidate_teacher_ranks"]
            ]
            cf_is_gt = [bool(value) for value in counterfactual["candidate_is_gt"]]
            if len(cf_ids) != TCB_PAIRS_PER_QUERY or len(cf_texts) != len(cf_ids):
                raise RuntimeError("TCB result does not contain three aligned pairs")
            if len(cf_ranks) != len(cf_ids) or len(cf_is_gt) != len(cf_ids):
                raise RuntimeError("TCB rank/GT metadata is not aligned")
            for candidate_id, text, teacher_rank, is_gt in zip(
                cf_ids, cf_texts, cf_ranks, cf_is_gt
            ):
                owners.append(owner)
                candidate_indices.append(ids.index(candidate_id))
                counterfactual_texts.append(text)
                pair_teacher_ranks.append(teacher_rank)
                pair_is_gt.append(is_gt)
                orientations.append(
                    _orientation(
                        str(manifest["query_id"]), candidate_id, int(args.seed)
                    )
                )

        if owners:
            owner_tensor = torch.tensor(owners, dtype=torch.long, device=device)
            candidate_tensor = torch.tensor(
                candidate_indices, dtype=torch.long, device=device
            )
            cf_first = torch.tensor(orientations, dtype=torch.bool, device=device)
            counterfactual_queries = model.compose_from_image_embeds(
                reference_embeds.index_select(0, owner_tensor),
                counterfactual_texts,
                device=device,
            )
            selected_original_queries = query_tokens.index_select(0, owner_tensor)
            logits = model.counterfactual_relational_logits(
                selected_original_queries,
                counterfactual_queries,
                candidate_tokens[owner_tensor, candidate_tensor],
                cf_first,
            ).float()
            targets = cf_first.float()
            gt_mask = torch.tensor(pair_is_gt, dtype=torch.bool, device=device)
            nongt_mask = ~gt_mask
            if args.tcb_relative_gt_loop and bool(nongt_mask.any()):
                nongt_indices = torch.nonzero(nongt_mask, as_tuple=False).flatten()
                gt_candidate_indices = torch.tensor(
                    [
                        ranked_ids[owner].index(str(manifests[owner]["target_id"]))
                        for owner in owners
                    ],
                    dtype=torch.long,
                    device=device,
                )
                gt_closed_loop_logits = model.counterfactual_relational_logits(
                    selected_original_queries.index_select(0, nongt_indices),
                    counterfactual_queries.index_select(0, nongt_indices),
                    candidate_tokens[
                        owner_tensor.index_select(0, nongt_indices),
                        gt_candidate_indices.index_select(0, nongt_indices),
                    ],
                    cf_first.index_select(0, nongt_indices),
                ).float()
                pair_losses, selected_pair_losses, relative_losses, z_c, z_g = (
                    relative_gt_pair_losses(
                        logits,
                        targets,
                        gt_mask,
                        gt_closed_loop_logits,
                        margin=args.tcb_relative_margin,
                    )
                )
                relative_metrics = relative_gt_metrics(
                    relative_losses,
                    z_c,
                    z_g,
                    margin=args.tcb_relative_margin,
                )
            else:
                selected_pair_losses = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
                pair_losses = selected_pair_losses
            per_query_sum = pair_losses.new_zeros(len(manifests))
            per_query_count = pair_losses.new_zeros(len(manifests))
            per_query_sum.index_add_(0, owner_tensor, pair_losses)
            per_query_count.index_add_(0, owner_tensor, torch.ones_like(pair_losses))
            eligible = per_query_count > 0
            tcb_loss = (per_query_sum[eligible] / per_query_count[eligible]).mean()
            selected_correct = ((logits.detach() > 0) == (targets > 0.5)).float()
            tcb_judgments = len(owners)
            tcb_accuracy = float(selected_correct.mean())
            tcb_positive_rate = float(targets.mean())
            tcb_queries = int(eligible.sum())
            tcb_pairs = len(owners)
            gt_pair_count = int(gt_mask.sum())
            nongt_pair_count = int(nongt_mask.sum())
            gt_pair_loss = (
                float(pair_losses[gt_mask].mean().detach()) if gt_pair_count else 0.0
            )
            nongt_pair_loss = (
                float(pair_losses[nongt_mask].mean().detach())
                if nongt_pair_count
                else 0.0
            )
            gt_pair_accuracy = (
                float(selected_correct[gt_mask].mean()) if gt_pair_count else 0.0
            )
            nongt_pair_accuracy = (
                float(selected_correct[nongt_mask].mean()) if nongt_pair_count else 0.0
            )
            selected_candidate_loss = float(selected_pair_losses.mean().detach())
            selected_candidate_accuracy = float(selected_correct.mean())
            gt_rank_counts = {
                str(rank): sum(
                    bool(is_gt) and int(pair_rank) == rank
                    for pair_rank, is_gt in zip(pair_teacher_ranks, pair_is_gt)
                )
                for rank in sorted(set(pair_teacher_ranks))
            }
            gt_rank_counts = {
                rank: count for rank, count in gt_rank_counts.items() if count
            }
        else:
            tcb_loss = candidate_tokens.sum() * 0.0
            tcb_accuracy = 0.0
            tcb_positive_rate = 0.0
            tcb_queries = 0
            tcb_pairs = 0
            tcb_judgments = 0
            gt_pair_count = 0
            nongt_pair_count = 0
            gt_pair_loss = 0.0
            nongt_pair_loss = 0.0
            gt_pair_accuracy = 0.0
            nongt_pair_accuracy = 0.0
            selected_candidate_loss = 0.0
            selected_candidate_accuracy = 0.0
            gt_rank_counts = {}

        # The shared training loop multiplies the returned value by list_weight.
        # This algebra keeps the intended 0.2*List + 0.1*TCB objective exact.
        tcb_active_fraction = tcb_queries / max(len(manifests), 1)
        effective_loss = (
            list_loss
            + (float(args.tcb_weight) / float(args.list_weight))
            * tcb_active_fraction
            * tcb_loss
        )
        memory_detail("tcb_head_and_loss_complete", tcb_pairs=len(owners))

    append_jsonl(
        {
            "time": time.time(),
            "queries": len(manifests),
            "tcb_queries": tcb_queries,
            "tcb_pairs": tcb_pairs,
            "tcb_judgments": tcb_judgments,
            "list_loss": float(list_loss.detach()),
            "tcb_loss": float(tcb_loss.detach()),
            "tcb_active_fraction": tcb_active_fraction,
            "effective_loss_returned_to_shared_loop": float(effective_loss.detach()),
            "tcb_accuracy": tcb_accuracy,
            "tcb_positive_rate": tcb_positive_rate,
            "gt_pairs": gt_pair_count,
            "nongt_pairs": nongt_pair_count,
            "gt_pair_loss": gt_pair_loss,
            "nongt_pair_loss": nongt_pair_loss,
            "gt_pair_accuracy": gt_pair_accuracy,
            "nongt_pair_accuracy": nongt_pair_accuracy,
            "selected_candidate_loss": selected_candidate_loss,
            "selected_candidate_accuracy": selected_candidate_accuracy,
            "relative_gt_loop": True,
            "relative_margin": float(args.tcb_relative_margin),
            "hard_gt_closed_loop": False,
            "accuracy_scope": "selected candidate binary judgments only",
            **relative_metrics,
            "gt_teacher_rank_counts": gt_rank_counts,
        },
        args.output_dir / "tcb_train_metrics.jsonl",
    )
    agreement = float((scores.detach().argmax(dim=1) == 0).float().mean())
    teacher_gt_top1 = sum(
        ids[0] == str(manifest["target_id"])
        for manifest, ids in zip(manifests, ranked_ids)
    )
    return effective_loss, agreement, int(teacher_gt_top1), len(manifests)


def _aggregate_sidecars(output_dir: Path) -> dict[str, Any]:
    request_path = output_dir / "counterfactual_request_events.jsonl"
    train_path = output_dir / "tcb_train_metrics.jsonl"
    request_rows = list(iter_jsonl(request_path)) if request_path.is_file() else []
    train_rows = list(iter_jsonl(train_path)) if train_path.is_file() else []
    total_pairs = sum(int(row["tcb_pairs"]) for row in train_rows)
    total_judgments = sum(
        int(row.get("tcb_judgments", row["tcb_pairs"])) for row in train_rows
    )
    total_queries = sum(int(row["tcb_queries"]) for row in train_rows)
    gt_pairs = sum(int(row.get("gt_pairs", 0)) for row in train_rows)
    nongt_pairs = sum(int(row.get("nongt_pairs", 0)) for row in train_rows)
    gt_rank_counts: dict[str, int] = {}
    for row in train_rows:
        for rank, count in row.get("gt_teacher_rank_counts", {}).items():
            gt_rank_counts[str(rank)] = gt_rank_counts.get(str(rank), 0) + int(count)
    return {
        "request_batches": len(request_rows),
        "requested_queries": sum(int(row["queries"]) for row in request_rows),
        "cache_hits": sum(int(row["cache_hits"]) for row in request_rows),
        "cache_misses": sum(int(row["cache_misses"]) for row in request_rows),
        "generation_attempts": sum(
            int(row["generation_attempts"]) for row in request_rows
        ),
        "generation_failures": sum(int(row["failures"]) for row in request_rows),
        "training_microbatches": len(train_rows),
        "tcb_queries": total_queries,
        "tcb_pairs": total_pairs,
        "tcb_judgments": total_judgments,
        "mean_tcb_loss": (
            sum(float(row["tcb_loss"]) * int(row["tcb_queries"]) for row in train_rows)
            / max(total_queries, 1)
        ),
        "pair_accuracy": (
            sum(
                float(row["tcb_accuracy"])
                * int(row.get("tcb_judgments", row["tcb_pairs"]))
                for row in train_rows
            )
            / max(total_judgments, 1)
        ),
        "positive_rate": (
            sum(
                float(row["tcb_positive_rate"])
                * int(row.get("tcb_judgments", row["tcb_pairs"]))
                for row in train_rows
            )
            / max(total_judgments, 1)
        ),
        "gt_pairs": gt_pairs,
        "nongt_pairs": nongt_pairs,
        "gt_pair_loss": (
            sum(
                float(row.get("gt_pair_loss", 0.0)) * int(row.get("gt_pairs", 0))
                for row in train_rows
            )
            / max(gt_pairs, 1)
        ),
        "nongt_pair_loss": (
            sum(
                float(row.get("nongt_pair_loss", 0.0)) * int(row.get("nongt_pairs", 0))
                for row in train_rows
            )
            / max(nongt_pairs, 1)
        ),
        "gt_pair_accuracy": (
            sum(
                float(row.get("gt_pair_accuracy", 0.0)) * int(row.get("gt_pairs", 0))
                for row in train_rows
            )
            / max(gt_pairs, 1)
        ),
        "nongt_pair_accuracy": (
            sum(
                float(row.get("nongt_pair_accuracy", 0.0))
                * int(row.get("nongt_pairs", 0))
                for row in train_rows
            )
            / max(nongt_pairs, 1)
        ),
        "gt_teacher_rank_counts": gt_rank_counts,
        "relative_gt_loop": True,
        "hard_gt_closed_loop": False,
        "relative_pairs": sum(int(row.get("relative_pairs", 0)) for row in train_rows),
        "selected_candidate_loss": (
            sum(
                float(row.get("selected_candidate_loss", 0)) * int(row["tcb_pairs"])
                for row in train_rows
            )
            / max(total_pairs, 1)
        ),
        "selected_candidate_accuracy": (
            sum(
                float(row.get("selected_candidate_accuracy", 0)) * int(row["tcb_pairs"])
                for row in train_rows
            )
            / max(total_pairs, 1)
        ),
        **{
            key: sum(
                float(row.get(key, 0)) * int(row.get("relative_pairs", 0))
                for row in train_rows
            )
            / max(sum(int(row.get("relative_pairs", 0)) for row in train_rows), 1)
            for key in (
                "relative_loss",
                "relative_margin_satisfied_rate",
                "relative_active_rate",
                "relative_z_c",
                "relative_z_g",
                "relative_gap",
            )
        },
    }


def main() -> None:
    original_qwen_request = qwen._request_qwen_as_online_teacher
    original_decorate = qwen._decorate_summary
    counterfactual_cache: dict[tuple[str, str], dict[str, Any]] | None = None

    def request_with_counterfactuals(manifests, **kwargs):
        nonlocal counterfactual_cache
        args = kwargs["args"]
        client = kwargs["client"]
        client.defer_sleep = True
        try:
            successful, ranking_results, hits, calls, failures = original_qwen_request(
                manifests, **kwargs
            )
            if counterfactual_cache is None:
                cf_path = (
                    args.output_dir / "online_qwen32_counterfactual_v4_cache.jsonl"
                )
                repair = base.repair_truncated_jsonl_tail(cf_path)
                counterfactual_cache = base.load_gpt_cache(cf_path)
                print(
                    "[CounterfactualCache] "
                    + json.dumps(
                        {
                            "path": str(cf_path.resolve()),
                            "rows": len(counterfactual_cache),
                            **repair,
                        }
                    ),
                    flush=True,
                )
            cf_path = args.output_dir / "online_qwen32_counterfactual_v4_cache.jsonl"
            cf_results, cf_stats = request_counterfactual_chunk(
                successful,
                ranking_results,
                args=args,
                cache=counterfactual_cache,
                cache_path=cf_path,
                client=client,
            )
        finally:
            client.defer_sleep = False
            client.sleep()
        augmented = []
        for ranking, counterfactual in zip(ranking_results, cf_results):
            row = dict(ranking)
            if counterfactual is not None:
                row["tcb_v4"] = counterfactual
            augmented.append(row)
        append_jsonl(
            {"time": time.time(), **cf_stats},
            args.output_dir / "counterfactual_request_events.jsonl",
        )
        print("[CounterfactualBatch] " + json.dumps(cf_stats), flush=True)
        return successful, augmented, hits, calls, failures

    def decorate_summary(args) -> None:
        original_decorate(args)
        path = args.output_dir / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["version"] = VERSION
        summary["listwise_objective"] = _loss_metadata()
        summary["loss_weights"]["transformation_comparison"] = float(args.tcb_weight)
        summary["transformation_comparison"] = {
            "version": TCB_VERSION,
            "counterfactual_prompt_version": counterfactual_prompt_version(
                str(args.dataset_name)
            ),
            "counterfactual_generation": (
                "parallel A/B/C branches"
                if args.cmi_three_branch
                else "one joint A/B/C request"
            ),
            "candidate_sampling": (
                f"fixed teacher ranks {list(args.tcb_fixed_positions)}, including GT"
                if args.tcb_fixed_positions is not None and args.tcb_include_gt
                else "deterministic three-way sample without replacement from ranked non-GT candidates"
            ),
            "supervision": "CF wins on selected candidate; for non-GT pairs, z_c - z_g >= margin",
            "gt_closed_loop": False,
            "relative_gt_loop": True,
            "relative_margin": float(args.tcb_relative_margin),
            "relative_pair_objective": "0.5 * (softplus(-z_c) + relu(margin - z_c + z_g))",
            "selected_gt_pair_objective": "softplus(-z_c); relative term masked",
            "text_orientation": "same shuffled orientation for candidate and GT; z=(2*cf_first-1)*logit",
            "relative_gradient": "both candidate and GT logits; no detach",
            "accuracy_scope": "candidate binary accuracy; relative margin satisfaction logged separately",
            "pairs_per_query": TCB_PAIRS_PER_QUERY,
            "pair_reduction": "mean within query, then shared raw-batch reduction",
            "weight": float(args.tcb_weight),
            "relational_gradient_to_shared_qformer": True,
            "relational_gradient_to_maxsim_weight": False,
            "cache": str(
                (
                    args.output_dir / "online_qwen32_counterfactual_v4_cache.jsonl"
                ).resolve()
            ),
            "metrics": _aggregate_sidecars(args.output_dir),
        }
        summary["train"]["effective_preference_loss_note"] = (
            "shared list fields contain List + (TCB_weight/List_weight)*TCB; "
            "the weighted total is exactly 0.8*GT + 0.2*List + 0.1*TCB"
        )
        atomic_json(summary, path)
        print(
            "[TCBSummary] "
            + json.dumps(summary["transformation_comparison"], ensure_ascii=False),
            flush=True,
        )

    def run_training(args) -> None:
        def compute_with_tcb(model, **kwargs):
            return compute_live_joint_loss(model, args=args, **kwargs)

        qwen.run_training(
            args,
            model_builder=build_model,
            listwise_loss=demotion_aware_rpa_loss,
            live_list_loss=compute_with_tcb,
            teacher_request=request_with_counterfactuals,
            summary_hook=decorate_summary,
            version=VERSION,
        )
        path = Path(args.output_dir) / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["tcb_architecture"] = architecture_metadata()
            atomic_json(summary, path)

    qwen.main(training_fn=run_training, configure_args=configure_training)


if __name__ == "__main__":
    main()
