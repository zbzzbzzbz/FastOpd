"""FashionIQ retrieval evaluation for the BLIP-2 + MaxSim baseline."""

from __future__ import annotations

import datetime
import time
from statistics import median
from typing import Dict, List, Sequence, Set

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _round(value: float) -> float:
    return round(float(value), 2)


def recall_from_scores(
    scores: np.ndarray,
    *,
    target_ids: Sequence[str],
    bank_item_ids: Sequence[str],
    reference_ids: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Compute FashionIQ recall after masking each query's reference image."""
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if len(target_ids) != scores.shape[0]:
        raise ValueError("target_ids and query count do not match")
    if len(bank_item_ids) != scores.shape[1]:
        raise ValueError("bank_item_ids and target count do not match")
    if reference_ids is not None and len(reference_ids) != scores.shape[0]:
        raise ValueError("reference_ids and query count do not match")

    bank_index = {str(item_id): index for index, item_id in enumerate(bank_item_ids)}
    ranks: List[int] = []
    for query_index, target_id in enumerate(target_ids):
        target_index = bank_index.get(str(target_id))
        if target_index is None:
            continue
        row = scores[query_index].copy()
        if reference_ids is not None:
            reference_index = bank_index.get(str(reference_ids[query_index]))
            if reference_index is not None:
                row[reference_index] = -1.0e10
        ranking = np.argsort(row)[::-1]
        ranks.append(int(np.where(ranking == target_index)[0][0]) + 1)

    if not ranks:
        return {
            "R1": 0.0,
            "R5": 0.0,
            "R10": 0.0,
            "R50": 0.0,
            "meanR3": 0.0,
            "meanR4": 0.0,
            "MnR": 0.0,
            "MdR": 0.0,
            "evaluated": 0,
        }

    total = float(len(ranks))
    r1 = 100.0 * sum(rank <= 1 for rank in ranks) / total
    r5 = 100.0 * sum(rank <= 5 for rank in ranks) / total
    r10 = 100.0 * sum(rank <= 10 for rank in ranks) / total
    r50 = 100.0 * sum(rank <= 50 for rank in ranks) / total
    return {
        "R1": _round(r1),
        "R5": _round(r5),
        "R10": _round(r10),
        "R50": _round(r50),
        "meanR3": _round((r1 + r5 + r10) / 3.0),
        "meanR4": _round((r1 + r5 + r10 + r50) / 4.0),
        "MnR": _round(sum(ranks) / total),
        "MdR": _round(float(median(ranks))),
        "evaluated": len(ranks),
    }


def summarize_categories(
    per_category: Dict[str, Dict[str, float]]
) -> Dict[str, object]:
    if not per_category:
        return {}
    output: Dict[str, object] = {
        "num_categories": len(per_category),
        "evaluated_total": sum(
            int(item.get("evaluated", 0)) for item in per_category.values()
        ),
    }
    for key in ("R1", "R5", "R10", "R50", "meanR3", "meanR4", "MnR", "MdR"):
        values = [float(item[key]) for item in per_category.values() if key in item]
        if values:
            output[f"macro_{key}"] = _round(sum(values) / len(values))
    output["per_category"] = per_category
    return output


@torch.no_grad()
def encode_queries(model, query_loader, device: torch.device) -> Dict[str, object]:
    model.eval()
    tokens: Dict[str, List[torch.Tensor]] = {}
    target_ids: Dict[str, List[str]] = {}
    reference_ids: Dict[str, List[str]] = {}
    for batch in tqdm(query_loader, desc="encode queries", unit="batch"):
        reference_image = batch["ref_img"].to(device, non_blocking=True)
        _, _, query_tokens = model.compose_feature(
            reference_image,
            list(batch["edit"]),
            device=device,
        )
        query_tokens = F.normalize(query_tokens, dim=-1).cpu()
        for index, category in enumerate(batch["dress_type"]):
            category = str(category)
            tokens.setdefault(category, []).append(query_tokens[index : index + 1])
            target_ids.setdefault(category, []).append(str(batch["target_id"][index]))
            reference_ids.setdefault(category, []).append(
                str(batch["candidate_id"][index])
            )
    return {
        "tokens": {key: torch.cat(value, dim=0) for key, value in tokens.items()},
        "target_ids": target_ids,
        "reference_ids": reference_ids,
    }


@torch.no_grad()
def encode_target_bank(
    model,
    *,
    build_target_loader,
    categories: Sequence[str],
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    tokens: Dict[str, torch.Tensor] = {}
    item_ids: Dict[str, List[str]] = {}
    for category in categories:
        category_tokens: List[torch.Tensor] = []
        category_ids: List[str] = []
        for batch in tqdm(
            build_target_loader(str(category)),
            desc=f"encode bank [{category}]",
            unit="batch",
            leave=False,
        ):
            image = batch["tar_img"].to(device, non_blocking=True)
            _, target_tokens, _ = model.target_fea(image, device=device)
            category_tokens.append(F.normalize(target_tokens, dim=-1).cpu())
            category_ids.extend(str(item) for item in batch["item_id"])
        if category_tokens:
            tokens[str(category)] = torch.cat(category_tokens, dim=0)
            item_ids[str(category)] = category_ids
    return {"tokens": tokens, "item_ids": item_ids}


def _filter_retrack_val_split(
    query_cache: Dict[str, object],
    bank_cache: Dict[str, object],
    categories: Sequence[str],
) -> Dict[str, object]:
    """Match the candidate pool used by the local ReTrack FashionIQ run."""
    filtered_tokens: Dict[str, torch.Tensor] = {}
    filtered_ids: Dict[str, List[str]] = {}
    for category in categories:
        category = str(category)
        bank_tokens = bank_cache["tokens"].get(category)
        bank_ids = bank_cache["item_ids"].get(category, [])
        if bank_tokens is None:
            continue
        allowed: Set[str] = set(query_cache["target_ids"].get(category, []))
        allowed.update(query_cache["reference_ids"].get(category, []))
        keep = [
            index for index, item_id in enumerate(bank_ids) if str(item_id) in allowed
        ]
        if keep:
            indices = torch.tensor(keep, dtype=torch.long)
            filtered_tokens[category] = bank_tokens.index_select(0, indices)
            filtered_ids[category] = [str(bank_ids[index]) for index in keep]
    return {"tokens": filtered_tokens, "item_ids": filtered_ids}


@torch.no_grad()
def evaluate_fashioniq(
    model,
    *,
    query_loader,
    build_target_loader,
    categories: Sequence[str],
    device: torch.device,
    candidate_pool: str = "retrack_val_split",
) -> Dict[str, object]:
    """Encode the validation split and evaluate category-wise MaxSim retrieval."""
    start_time = time.time()
    query_cache = encode_queries(model, query_loader, device)
    bank_cache = encode_target_bank(
        model,
        build_target_loader=build_target_loader,
        categories=categories,
        device=device,
    )
    if candidate_pool == "retrack_val_split":
        bank_cache = _filter_retrack_val_split(query_cache, bank_cache, categories)
    elif candidate_pool != "full_bank":
        raise ValueError(f"Unsupported candidate_pool: {candidate_pool}")

    per_category: Dict[str, Dict[str, float]] = {}
    bank_sizes: Dict[str, int] = {}
    for category in categories:
        category = str(category)
        query_tokens = query_cache["tokens"].get(category)
        target_tokens = bank_cache["tokens"].get(category)
        bank_ids = bank_cache["item_ids"].get(category, [])
        if query_tokens is None or target_tokens is None or not bank_ids:
            continue
        scores = model.fine_grained_sim(query_tokens, target_tokens)
        per_category[category] = recall_from_scores(
            scores,
            target_ids=query_cache["target_ids"][category],
            bank_item_ids=bank_ids,
            reference_ids=query_cache["reference_ids"][category],
        )
        bank_sizes[category] = len(bank_ids)

    output = summarize_categories(per_category)
    output["candidate_pool"] = candidate_pool
    output["bank_size_by_category"] = bank_sizes
    elapsed = time.time() - start_time
    output["eval_time_sec"] = _round(elapsed)
    output["eval_time_str"] = str(datetime.timedelta(seconds=int(elapsed)))
    return output
