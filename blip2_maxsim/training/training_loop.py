"""Coarse-to-fine online preference distillation for FashionIQ.

The warmup full bank is only a high-recall index.  For every current query we
retrieve a coarse Top-K, re-encode a small shortlist with the current student,
and ask GPT to rank a current-student Top-10 containing the GT.  The gradient
pass re-encodes those ten images again, so both query and candidate paths are
updated.  No image/Vision-Transformer feature cache is created.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from blip2_maxsim.data.common import parse_categories, resolve_fashioniq_layout
from blip2_maxsim.data import FashionIQDataModule, FashionIQPairDataset, build_transform
from blip2_maxsim.data.image_resolver import resolve_manifest_image_path
from blip2_maxsim.training.evaluation import evaluate_fashioniq
from blip2_maxsim.teacher.ranking import (
    candidate_set_sha256,
    is_content_inspection_failure,
    labels_for_count,
    parse_ranking,
    rank_one,
    request_fingerprint,
    prompt_version,
)
from blip2_maxsim.model.matching import aligned_maxsim_scores
from blip2_maxsim.training.student_core import (
    append_jsonl,
    atomic_json,
    autocast_context,
    build_model,
    build_optimizer,
    iter_jsonl,
    listwise_rpa_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "blip2_maxsim_stage1_online_opd_coarse100_live20_v1"
DEFAULT_DATASET = Path(os.environ.get("FASTOPD_DATASET_ROOT", "data/FashionIQ"))
DEFAULT_BASE = Path(
    os.environ.get("FASTOPD_BASE_CHECKPOINT", "checkpoints/warmup_end.pt")
)
DEFAULT_TRAIN_BANK = Path(
    os.environ.get("FASTOPD_TRAIN_ANCHOR_DIR", "data/train_warmup_end")
)


@dataclass
class CategorySnapshotBank:
    category: str
    item_ids: list[str]
    tokens: torch.Tensor
    id_to_index: dict[str, int]


class ParallelImageLoader:
    """Load dynamic shortlist images without retaining a persistent image cache."""

    def __init__(self, args: argparse.Namespace, transform, workers: int) -> None:
        self.args = args
        self.transform = transform
        self.executor = ThreadPoolExecutor(max_workers=max(int(workers), 1))

    def _load_one(self, item: tuple[str, str]) -> torch.Tensor:
        image_id, category = item
        path = resolve_manifest_image_path(self.args, {"category": category}, image_id)
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def load(self, image_ids: Sequence[str], category: str) -> torch.Tensor:
        if not image_ids:
            raise ValueError("Cannot load an empty dynamic candidate batch")
        items = [(str(image_id), str(category)) for image_id in image_ids]
        return torch.stack(list(self.executor.map(self._load_one, items)))

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--phase", choices=("validate", "train"), default="validate")
    parser.add_argument(
        "--experiment", default="stage1_online_opd_coarse100_live20_bs64_gt08_list02"
    )
    parser.add_argument("--dataset-name", choices=("fashioniq",), default="fashioniq")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--initial-delta", type=Path, default=None)
    parser.add_argument("--train-anchor-dir", type=Path, default=DEFAULT_TRAIN_BANK)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/full_fashioniq",
    )
    parser.add_argument("--categories", default="dress,shirt,toptee")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=364)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument(
        "--list-micro-batch-size",
        type=int,
        default=32,
        help="Active OPD queries per gradient micro-batch; each has teacher_list_size live images.",
    )
    parser.add_argument(
        "--candidate-encode-batch-size",
        type=int,
        default=64,
        help="Image batch used by the no-gradient live Top-20 reranking pass.",
    )
    parser.add_argument("--coarse-topk", type=int, default=100)
    parser.add_argument("--live-shortlist-size", type=int, default=20)
    parser.add_argument("--teacher-list-size", type=int, default=10)
    parser.add_argument("--teacher-min-gt-rank", type=int, default=2)
    parser.add_argument("--teacher-max-gt-rank", type=int, default=20)
    parser.add_argument(
        "--teacher-max-score-gap",
        type=float,
        default=0.0,
        help="If positive, skip GPT when live Top-1 minus GT exceeds this value.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-loader-workers", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--gt-weight", type=float, default=0.8)
    parser.add_argument("--list-weight", type=float, default=0.2)
    parser.add_argument("--qformer-lr", type=float, default=5e-6)
    parser.add_argument("--head-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--rank-temperature", type=float, default=0.07)
    parser.add_argument("--opd-temperature", type=float, default=0.07)
    parser.add_argument("--maxsim-temperature", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-preflight-eval", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume model/optimizer/progress from the rolling checkpoint when present.",
    )
    parser.add_argument(
        "--checkpoint-percent-interval",
        type=int,
        default=10,
        help="Atomically overwrite one rolling checkpoint at this percent interval.",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=10.0,
        help="Do not start another GPT batch below this free-space threshold.",
    )

    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("FASTOPD_API_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--api-format",
        choices=("responses", "chat_completions"),
        default="responses",
        help="Multimodal teacher API protocol.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable provider-side thinking for compatible Chat Completions models.",
    )
    parser.add_argument("--api-max-tokens", type=int, default=0)
    parser.add_argument("--api-temperature", type=float, default=None)
    parser.add_argument("--api-top-p", type=float, default=None)
    parser.add_argument(
        "--trust-env", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--image-detail", choices=("auto", "low", "high"), default="auto"
    )
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--parse-policy", choices=("strict", "paper_fallback"), default="strict"
    )
    parser.add_argument(
        "--execute-gpt",
        "--execute-teacher",
        dest="execute_gpt",
        action="store_true",
    )
    parser.add_argument(
        "--cache-replay-model",
        default=None,
        help="Prefer cached rankings from this teacher through the configured step.",
    )
    parser.add_argument(
        "--cache-replay-until-step",
        type=int,
        default=0,
        help="Last optimizer step allowed to use --cache-replay-model.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dataset_categories(raw: str | Sequence[str], dataset_name: str) -> list[str]:
    return parse_categories(raw)


def _manifest_shards(anchor_dir: Path, manifest: Mapping[str, Any]) -> list[Path]:
    rows = manifest.get("bank_shards", [])
    paths = (
        [anchor_dir / str(row["file"]) for row in rows]
        if rows
        else sorted(anchor_dir.glob("bank_shard_*.pt"))
    )
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"Incomplete bank shards under {anchor_dir}")
    return paths


def validate_anchor_manifest(
    anchor_dir: Path, *, base_checkpoint: Path
) -> dict[str, Any]:
    manifest_path = anchor_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("split")) != "train":
        raise ValueError(f"Expected train snapshot, found {manifest.get('split')}")
    recorded = Path(str(manifest.get("fm_ckpt_path", ""))).expanduser().resolve()
    if recorded != base_checkpoint.resolve():
        raise ValueError(
            f"Snapshot checkpoint mismatch: manifest={recorded}, requested={base_checkpoint}"
        )
    if str(manifest.get("save_dtype")) != "fp32":
        raise ValueError(f"Expected fp32 snapshot, found {manifest.get('save_dtype')}")
    _manifest_shards(anchor_dir, manifest)
    return manifest


def load_snapshot_banks(
    anchor_dir: Path, *, base_checkpoint: Path, categories: Sequence[str]
) -> tuple[dict[str, CategorySnapshotBank], dict[str, Any]]:
    manifest = validate_anchor_manifest(anchor_dir, base_checkpoint=base_checkpoint)
    wanted = {str(category) for category in categories}
    ids_by_category = {category: [] for category in wanted}
    tokens_by_category: dict[str, list[torch.Tensor]] = {
        category: [] for category in wanted
    }
    for path in tqdm(
        _manifest_shards(anchor_dir, manifest), desc="load train snapshot", unit="shard"
    ):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {"item_ids", "dress_types", "target_tokens"}
        if not required.issubset(payload):
            raise ValueError(f"Malformed snapshot shard {path}")
        item_ids = [str(value) for value in payload["item_ids"]]
        dress_types = [str(value) for value in payload["dress_types"]]
        tokens = payload["target_tokens"]
        if (
            tokens.dtype != torch.float32
            or tokens.ndim != 3
            or tuple(tokens.shape[1:]) != (32, 256)
            or len(item_ids) != len(tokens)
            or len(dress_types) != len(tokens)
            or not bool(torch.isfinite(tokens).all())
        ):
            raise ValueError(f"Unexpected snapshot payload in {path}")
        for category in wanted:
            indices = [i for i, value in enumerate(dress_types) if value == category]
            if indices:
                selector = torch.tensor(indices, dtype=torch.long)
                tokens_by_category[category].append(tokens.index_select(0, selector))
                ids_by_category[category].extend(item_ids[i] for i in indices)
    expected = {
        str(key): int(value)
        for key, value in dict(manifest.get("bank_by_category", {})).items()
    }
    banks = {}
    for category in categories:
        category = str(category)
        item_ids = ids_by_category[category]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError(f"Duplicate {category} IDs in snapshot")
        if len(item_ids) != expected.get(category):
            raise ValueError(
                f"{category} snapshot count {len(item_ids)} != {expected.get(category)}"
            )
        banks[category] = CategorySnapshotBank(
            category=category,
            item_ids=item_ids,
            tokens=torch.cat(tokens_by_category[category]).contiguous(),
            id_to_index={item_id: index for index, item_id in enumerate(item_ids)},
        )
    return banks, manifest


def repair_truncated_jsonl_tail(path: Path) -> dict[str, Any]:
    """Remove only a malformed final JSONL row, preserving it in a sidecar.

    Append-only logs can end with one partial row if the filesystem fills
    during ``flush``.  Corruption anywhere except the physical final row is
    treated as a hard error rather than silently discarded.
    """
    if not path.is_file():
        return {"repaired": False, "valid_rows": 0}
    lines = path.read_bytes().splitlines(keepends=True)
    offset = 0
    valid_rows = 0
    for index, line in enumerate(lines):
        start = offset
        offset += len(line)
        if not line.strip():
            continue
        try:
            json.loads(line)
            valid_rows += 1
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if index != len(lines) - 1:
                raise ValueError(
                    f"Non-tail JSONL corruption at {path}, line {index + 1}"
                ) from error
            backup = path.with_name(f"{path.name}.corrupt_tail_{int(time.time())}.bin")
            backup.write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(start)
                handle.flush()
                os.fsync(handle.fileno())
            return {
                "repaired": True,
                "valid_rows": valid_rows,
                "removed_bytes": len(line),
                "backup": str(backup.resolve()),
            }
    return {"repaired": False, "valid_rows": valid_rows}


def load_gpt_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cache = {}
    if path.is_file():
        for row in iter_jsonl(path):
            cache[(str(row["query_id"]), str(row["fingerprint"]))] = row
    return cache


def is_skippable_teacher_failure(result: Mapping[str, Any]) -> bool:
    """Only query-specific ranking refusals may drop the List-wise term.

    Transport, authentication, quota, and model-availability errors are run-level
    failures.  Treating those as individual teacher abstentions would silently
    turn OPD training into GT-only training.
    """
    error = str(result.get("error", ""))
    return error.startswith(
        "ValueError: Incomplete permutation:"
    ) or is_content_inspection_failure(error)


def request_gpt(
    manifests: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    cache: dict[tuple[str, str], dict[str, Any]],
    cache_path: Path,
    preferred_cache_model: str | None = None,
) -> tuple[
    list[Mapping[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    list[dict[str, Any]],
]:
    ordered: list[dict[str, Any] | None] = [None] * len(manifests)
    pending: list[tuple[int, Mapping[str, Any], str]] = []
    failures: list[dict[str, Any]] = []
    fatal_failures: list[dict[str, Any]] = []
    cache_hits = 0
    for index, manifest in enumerate(manifests):
        fingerprint = request_fingerprint(manifest, args)
        cached = None
        if preferred_cache_model:
            cache_args = copy.copy(args)
            cache_args.model = str(preferred_cache_model)
            replay_fingerprint = request_fingerprint(manifest, cache_args)
            cached = cache.get((str(manifest["query_id"]), replay_fingerprint))
        if cached is None:
            cached = cache.get((str(manifest["query_id"]), fingerprint))
        if cached and cached.get("candidate_set_sha256") == candidate_set_sha256(
            manifest
        ):
            cache_hits += 1
            if cached.get("status") == "ok":
                ordered[index] = cached
            elif is_skippable_teacher_failure(cached):
                failures.append(cached)
            else:
                # A cached infrastructure failure must be retried rather than
                # permanently suppressing the online teacher.
                cache_hits -= 1
                pending.append((index, manifest, fingerprint))
        else:
            pending.append((index, manifest, fingerprint))
    if pending:
        with ThreadPoolExecutor(
            max_workers=min(int(args.workers), len(pending))
        ) as executor:
            futures = {}
            for index, manifest, fingerprint in pending:
                resolver = (
                    lambda item_id, manifest=manifest: resolve_manifest_image_path(
                        args, manifest, item_id
                    )
                )
                future = executor.submit(
                    rank_one, manifest, args, image_path_for_id=resolver
                )
                futures[future] = (index, fingerprint)
            for future in as_completed(futures):
                index, fingerprint = futures[future]
                result = future.result()
                append_jsonl(result, cache_path)
                cache[(str(result["query_id"]), fingerprint)] = result
                if result.get("status") == "ok":
                    ordered[index] = result
                elif is_skippable_teacher_failure(result):
                    failures.append(result)
                else:
                    fatal_failures.append(result)
    if fatal_failures:
        first = fatal_failures[0]
        raise RuntimeError(
            f"GPT infrastructure failure for {first.get('query_id')}: "
            f"{first.get('error')}; failed_queries={len(fatal_failures)}"
        )
    successful_manifests = [
        manifest for manifest, result in zip(manifests, ordered) if result is not None
    ]
    successful_results = [result for result in ordered if result is not None]
    if len(successful_manifests) + len(failures) != len(manifests):
        raise RuntimeError("Missing online GPT result")
    return (
        successful_manifests,
        successful_results,
        cache_hits,
        len(pending),
        failures,
    )


@torch.no_grad()
def full_snapshot_scores(
    query_tokens: torch.Tensor,
    *,
    bank_tokens_transposed: torch.Tensor,
    token_mat_weight: torch.Tensor,
    eps_temperature: float,
) -> torch.Tensor:
    """Use detached snapshot tokens only to retrieve coarse candidate IDs."""
    query = F.normalize(query_tokens.detach().float(), dim=-1)
    weight = token_mat_weight.detach().float()
    rows = []
    for index in range(query.shape[0]):
        similarity = torch.matmul(
            query[index].unsqueeze(0).unsqueeze(1), bank_tokens_transposed
        )
        per_token = similarity.max(dim=-1).values
        token_weights = torch.softmax(
            torch.matmul(
                torch.softmax(per_token / float(eps_temperature), dim=-1), weight
            )
            / float(eps_temperature),
            dim=-1,
        )
        rows.append(torch.sum(per_token * token_weights, dim=-1))
    return torch.cat(rows, dim=0)


def build_coarse_shortlists(
    batch: Mapping[str, Any],
    *,
    scores: torch.Tensor,
    bank: CategorySnapshotBank,
    coarse_topk: int,
    live_shortlist_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(scores.shape[0]):
        row_scores = scores[index].clone()
        reference_id = str(batch["candidate_id"][index])
        target_id = str(batch["target_id"][index])
        reference_index = bank.id_to_index.get(reference_id)
        if reference_index is not None:
            row_scores[reference_index] = -torch.inf
        target_index = bank.id_to_index.get(target_id)
        if target_index is None:
            raise KeyError(f"GT {target_id} is absent from {bank.category} snapshot")
        target_score = row_scores[target_index]
        snapshot_gt_rank = int((row_scores > target_score).sum().item()) + 1
        k = min(int(coarse_topk), row_scores.numel())
        top = torch.topk(row_scores, k=k, largest=True, sorted=True)
        coarse_indices = [int(value) for value in top.indices.tolist()]
        coarse_ids = [bank.item_ids[value] for value in coarse_indices]
        live_ids = coarse_ids[: int(live_shortlist_size)]
        if target_id not in live_ids:
            live_ids[-1] = target_id
        if len(live_ids) != len(set(live_ids)):
            raise RuntimeError(
                "Duplicate ID created while forcing GT into live shortlist"
            )
        rows.append(
            {
                "source_batch_index": index,
                "query_id": f"train:{bank.category}:{int(batch['pair_id'][index])}",
                "category": bank.category,
                "ref_id": reference_id,
                "target_id": target_id,
                "edit": str(batch["edit"][index]),
                "snapshot_gt_rank": snapshot_gt_rank,
                "target_in_coarse_topk": target_id in coarse_ids,
                "live_candidate_ids": live_ids,
            }
        )
    return rows


@torch.no_grad()
def encode_dynamic_candidates(
    model,
    *,
    image_ids: Sequence[str],
    category: str,
    image_loader: ParallelImageLoader,
    encode_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    model.eval()
    for start in range(0, len(image_ids), int(encode_batch_size)):
        ids = image_ids[start : start + int(encode_batch_size)]
        images = image_loader.load(ids, category).to(device, non_blocking=True)
        with autocast_context(device):
            _, tokens, _ = model.target_fea(images, device=device)
        parts.append(F.normalize(tokens.float(), dim=-1))
        del images, tokens
    return torch.cat(parts, dim=0)


@torch.no_grad()
def live_rerank(
    model,
    *,
    query_tokens: torch.Tensor,
    shortlists: Sequence[Mapping[str, Any]],
    image_loader: ParallelImageLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not shortlists:
        return [], {}
    category = str(shortlists[0]["category"])
    unique_ids = list(
        dict.fromkeys(
            str(item_id) for row in shortlists for item_id in row["live_candidate_ids"]
        )
    )
    unique_tokens = encode_dynamic_candidates(
        model,
        image_ids=unique_ids,
        category=category,
        image_loader=image_loader,
        encode_batch_size=int(args.candidate_encode_batch_size),
        device=device,
    )
    id_to_live_index = {item_id: index for index, item_id in enumerate(unique_ids)}
    indices = torch.tensor(
        [
            [id_to_live_index[str(item_id)] for item_id in row["live_candidate_ids"]]
            for row in shortlists
        ],
        dtype=torch.long,
        device=device,
    )
    candidate_tokens = unique_tokens.index_select(0, indices.flatten()).reshape(
        len(shortlists), int(args.live_shortlist_size), 32, 256
    )
    current_scores, _ = aligned_maxsim_scores(
        query_tokens,
        candidate_tokens,
        model.compose_token_mat_weight,
        model.maxsim_temperature,
    )
    manifests = []
    counters = {
        "live_gt_top1_skipped": 0,
        "live_gt_rank_gate_skipped": 0,
        "live_score_gap_skipped": 0,
        "target_in_coarse_topk": 0,
        "live_unique_candidates": len(unique_ids),
        "live_gt_ranks": [],
    }
    list_size = int(args.teacher_list_size)
    for row_index, source in enumerate(shortlists):
        candidate_ids = [str(value) for value in source["live_candidate_ids"]]
        scores = current_scores[row_index].float()
        order = scores.argsort(descending=True).tolist()
        target_id = str(source["target_id"])
        gt_position = candidate_ids.index(target_id)
        gt_rank = order.index(gt_position) + 1
        score_gap = float(scores[order[0]] - scores[gt_position])
        counters["target_in_coarse_topk"] += int(source["target_in_coarse_topk"])
        counters["live_gt_ranks"].append(gt_rank)
        if gt_rank == 1:
            counters["live_gt_top1_skipped"] += 1
            continue
        if (
            not int(args.teacher_min_gt_rank)
            <= gt_rank
            <= int(args.teacher_max_gt_rank)
        ):
            counters["live_gt_rank_gate_skipped"] += 1
            continue
        if float(args.teacher_max_score_gap) > 0 and score_gap > float(
            args.teacher_max_score_gap
        ):
            counters["live_score_gap_skipped"] += 1
            continue

        chosen_positions = [position for position in order if position != gt_position][
            : list_size - 1
        ]
        chosen_positions.append(gt_position)
        chosen_positions.sort(
            key=lambda position: float(scores[position]), reverse=True
        )
        if len(chosen_positions) != list_size:
            raise RuntimeError(
                f"Could not construct GT plus {list_size - 1} hard negatives"
            )
        candidates = [
            {
                "candidate_id": candidate_ids[position],
                "student_rank": rank,
                "student_score": float(scores[position]),
                "is_gt": candidate_ids[position] == target_id,
            }
            for rank, position in enumerate(chosen_positions, start=1)
        ]
        manifests.append(
            {
                "version": VERSION,
                "dataset_name": str(getattr(args, "dataset_name", "fashioniq")),
                "query_id": str(source["query_id"]),
                "category": category,
                "source_batch_index": int(source["source_batch_index"]),
                "ref_id": str(source["ref_id"]),
                "target_id": target_id,
                "edit": str(source["edit"]),
                "candidates": candidates,
                "student_top1_is_gt": False,
                "student_gt_rank_in_live_shortlist": gt_rank,
                "student_gt_rank_in_teacher_list": next(
                    rank
                    for rank, item in enumerate(candidates, start=1)
                    if item["is_gt"]
                ),
                "snapshot_gt_rank": int(source["snapshot_gt_rank"]),
                "live_top1_minus_gt_score": score_gap,
            }
        )
    del unique_tokens, candidate_tokens, current_scores
    return manifests, counters


def compute_gt_loss(model, batch: Mapping[str, Any], device: torch.device):
    reference = batch["ref_img"].to(device, non_blocking=True)
    target = batch["tar_img"].to(device, non_blocking=True)
    with autocast_context(device):
        _, _, query_tokens = model.compose_feature(
            reference, list(batch["edit"]), device=device
        )
        _, target_tokens, _ = model.target_fea(target, device=device)
        loss, logits = model.compute_rank_loss(query_tokens, target_tokens)
    labels = torch.arange(logits.shape[0], device=device)
    accuracy = float((logits.detach().argmax(dim=1) == labels).float().mean())
    return loss, accuracy


def compute_live_list_loss(
    model,
    *,
    raw_batch: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    image_loader: ParallelImageLoader,
    device: torch.device,
    temperature: float,
) -> tuple[torch.Tensor, float, int, int]:
    if len(manifests) != len(results) or not manifests:
        raise ValueError("Live list loss requires aligned non-empty manifests/results")
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
    list_size = len(ranked_ids[0])
    if any(len(ids) != list_size or len(set(ids)) != list_size for ids in ranked_ids):
        raise RuntimeError(f"Teacher result is not a {list_size}-item permutation")
    flat_ids = [item_id for ids in ranked_ids for item_id in ids]
    candidate_images = image_loader.load(flat_ids, category).to(
        device, non_blocking=True
    )
    with autocast_context(device):
        _, _, query_tokens = model.compose_feature(references, edits, device=device)
        _, flat_tokens, _ = model.target_fea(candidate_images, device=device)
        candidate_tokens = flat_tokens.reshape(len(manifests), list_size, 32, 256)
        scores, _ = aligned_maxsim_scores(
            query_tokens,
            candidate_tokens,
            model.compose_token_mat_weight,
            model.maxsim_temperature,
        )
        teacher_scores = torch.linspace(
            1.0, 0.0, list_size, device=device, dtype=torch.float32
        ).repeat(len(manifests), 1)
        loss = listwise_rpa_loss(scores, teacher_scores, temperature)
    agreement = float((scores.detach().argmax(dim=1) == 0).float().mean())
    teacher_gt_top1 = 0
    for manifest, ids in zip(manifests, ranked_ids):
        teacher_gt_top1 += int(ids[0] == str(manifest["target_id"]))
    return loss, agreement, teacher_gt_top1, len(manifests)


def save_delta(
    path: Path, model, args: argparse.Namespace, summary: Mapping[str, Any]
) -> None:
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }
    torch.save(
        {
            "format_version": VERSION,
            "base_checkpoint": str(args.base_checkpoint),
            "initial_delta": str(args.initial_delta) if args.initial_delta else None,
            "snapshot_bank": str(args.train_anchor_dir),
            "trainable_state": state,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "dataset_root_resolved_images"
            },
            "metrics": dict(summary),
        },
        path,
    )


def evaluate_current_full_bank(model, args: argparse.Namespace) -> dict[str, Any]:
    """Always re-encode the complete validation bank with the current model."""
    data = FashionIQDataModule(
        dataset_root=str(args.dataset_root),
        categories=parse_categories(args.categories),
        train_split="train",
        eval_split="val",
        batch_size=int(args.train_batch_size),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        pin_memory=True,
        image_size=int(args.image_size),
        max_train_rows=1,
        max_eval_rows=int(args.max_eval_rows),
    )
    result = evaluate_fashioniq(
        model,
        query_loader=data.eval_query_dataloader(),
        build_target_loader=data.build_eval_target_loader,
        categories=data.categories,
        device=torch.device(args.device),
        candidate_pool="full_bank",
    )
    if result.get("candidate_pool") != "full_bank":
        raise RuntimeError("Evaluation unexpectedly used a reduced candidate pool")
    return result


def _trainable_state_cpu(model) -> dict[str, torch.Tensor]:
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Keep the previous rolling checkpoint valid until the new file is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resume_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_name": str(args.dataset_name),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "initial_delta": (
            str(args.initial_delta.resolve()) if args.initial_delta else None
        ),
        "snapshot_bank": str(args.train_anchor_dir.resolve()),
        "categories": parse_dataset_categories(args.categories, args.dataset_name),
        "train_batch_size": int(args.train_batch_size),
        "coarse_topk": int(args.coarse_topk),
        "live_shortlist_size": int(args.live_shortlist_size),
        "teacher_list_size": int(args.teacher_list_size),
        "gt_weight": float(args.gt_weight),
        "list_weight": float(args.list_weight),
        "seed": int(args.seed),
    }


def save_rolling_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scaler,
    args: argparse.Namespace,
    completed_steps: int,
    total_steps: int,
    milestone_percent: int,
    counters: Mapping[str, Any],
    sums: Mapping[str, Any],
    live_gt_rank_hist: Mapping[str, int],
    snapshot_gt_rank_hist: Mapping[str, int],
    elapsed_training_sec: float,
) -> None:
    _atomic_torch_save(
        {
            "format_version": VERSION,
            "kind": "rolling_training_checkpoint",
            "resume_signature": _resume_signature(args),
            "completed_steps": int(completed_steps),
            "total_steps": int(total_steps),
            "milestone_percent": int(milestone_percent),
            "trainable_state": _trainable_state_cpu(model),
            "optimizer_state": optimizer.state_dict(),
            "grad_scaler_state": scaler.state_dict(),
            "rng_state": _rng_state(),
            "counters": dict(counters),
            "sums": dict(sums),
            "live_gt_rank_hist": dict(live_gt_rank_hist),
            "snapshot_gt_rank_hist": dict(snapshot_gt_rank_hist),
            "elapsed_training_sec": float(elapsed_training_sec),
        },
        path,
    )


def load_rolling_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scaler,
    args: argparse.Namespace,
    total_steps: int,
    device: torch.device,
) -> dict[str, Any] | None:
    if not bool(args.resume) or not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != VERSION:
        raise ValueError(f"Rolling checkpoint version mismatch: {path}")
    if payload.get("resume_signature") != _resume_signature(args):
        raise ValueError("Rolling checkpoint configuration does not match this run")
    if int(payload.get("total_steps", -1)) != int(total_steps):
        raise ValueError("Rolling checkpoint total-step count changed")
    loaded = model.load_state_dict(payload["trainable_state"], strict=False)
    if loaded.unexpected_keys:
        raise RuntimeError(
            f"Unexpected rolling-checkpoint model keys: {loaded.unexpected_keys[:10]}"
        )
    optimizer.load_state_dict(payload["optimizer_state"])
    _optimizer_state_to_device(optimizer, device)
    scaler.load_state_dict(payload["grad_scaler_state"])
    _restore_rng_state(payload["rng_state"])
    return payload


def _free_disk_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def run_epoch(
    model,
    optimizer,
    args: argparse.Namespace,
    device: torch.device,
    banks: Mapping[str, CategorySnapshotBank],
    probe=None,
) -> dict[str, Any]:
    if not args.execute_gpt:
        raise RuntimeError("Online OPD training requires --execute-gpt")
    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set")

    cache_path = args.output_dir / str(
        getattr(args, "teacher_cache_filename", "online_gpt_cache.jsonl")
    )
    event_path = args.output_dir / "online_batch_events.jsonl"
    rolling_path = args.output_dir / "rolling_checkpoint.pt"
    cache_repair = repair_truncated_jsonl_tail(cache_path)
    if cache_repair["repaired"]:
        print("[CacheTailRepair] " + json.dumps(cache_repair), flush=True)
    cache = load_gpt_cache(cache_path)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    categories = parse_dataset_categories(args.categories, args.dataset_name)
    transform = build_transform(int(args.image_size))
    datasets = {
        category: FashionIQPairDataset(
            dataset_root=args.dataset_root,
            split="train",
            categories=[category],
            transform=transform,
            max_rows=0,
        )
        for category in categories
    }
    collate_fn = None
    category_steps = {
        category: len(dataset) // int(args.train_batch_size)
        for category, dataset in datasets.items()
    }
    total_steps = sum(category_steps.values())
    if int(args.max_train_batches) > 0:
        total_steps = min(total_steps, int(args.max_train_batches))
    interval = int(args.checkpoint_percent_interval)
    milestone_by_step = (
        {
            int(math.ceil(total_steps * percent / 100.0)): percent
            for percent in range(interval, 101, interval)
        }
        if not args.skip_checkpoint
        else {}
    )
    image_loader = ParallelImageLoader(args, transform, int(args.image_loader_workers))
    sums = {
        "gt_loss": 0.0,
        "active_list_loss": 0.0,
        "batch_normalized_list_loss": 0.0,
        "weighted_gt": 0.0,
        "weighted_list": 0.0,
        "weighted_total": 0.0,
        "inbatch_gt_accuracy": 0.0,
        "student_teacher_top1_agreement": 0.0,
        "live_top1_rate": 0.0,
    }
    counters = {
        "optimizer_steps": 0,
        "raw_queries": 0,
        "gpt_queries": 0,
        "gpt_attempted_queries": 0,
        "gpt_teacher_failures": 0,
        "gpt_api_calls": 0,
        "gpt_cache_hits": 0,
        "live_gt_top1_skipped": 0,
        "live_gt_rank_gate_skipped": 0,
        "live_score_gap_skipped": 0,
        "target_in_coarse_topk": 0,
        "teacher_gt_top1": 0,
        "teacher_gt_present": 0,
        "live_candidate_encodings_for_mining": 0,
        "live_candidate_encodings_for_loss": 0,
    }
    live_gt_rank_hist: dict[str, int] = {}
    snapshot_gt_rank_hist: dict[str, int] = {}
    session_started = time.time()
    prior_elapsed_sec = 0.0
    resume_payload = load_rolling_checkpoint(
        rolling_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        total_steps=total_steps,
        device=device,
    )
    if resume_payload is not None:
        counters = dict(resume_payload["counters"])
        counters.setdefault("gpt_attempted_queries", counters.get("gpt_queries", 0))
        counters.setdefault("gpt_teacher_failures", 0)
        sums = dict(resume_payload["sums"])
        live_gt_rank_hist = dict(resume_payload["live_gt_rank_hist"])
        snapshot_gt_rank_hist = dict(resume_payload["snapshot_gt_rank_hist"])
        prior_elapsed_sec = float(resume_payload.get("elapsed_training_sec", 0.0))
        resume_completed_steps = int(resume_payload["completed_steps"])
        last_checkpoint_percent = int(resume_payload["milestone_percent"])
        if int(counters["optimizer_steps"]) != resume_completed_steps:
            raise ValueError("Rolling checkpoint progress/counter mismatch")
        print(
            "[Resume] "
            + json.dumps(
                {
                    "path": str(rolling_path.resolve()),
                    "completed_steps": resume_completed_steps,
                    "total_steps": total_steps,
                    "milestone_percent": last_checkpoint_percent,
                    "cached_gpt_rows": len(cache),
                }
            ),
            flush=True,
        )
    else:
        resume_completed_steps = 0
        last_checkpoint_percent = 0
    attempt_path = args.output_dir / "run_attempts.jsonl"
    recorded_attempt_ids = (
        [
            int(row["attempt_id"])
            for row in iter_jsonl(attempt_path)
            if str(row.get("event")) == "start" and row.get("attempt_id") is not None
        ]
        if attempt_path.is_file()
        else []
    )
    prior_attempt_id = max(recorded_attempt_ids or [1 if event_path.is_file() else 0])
    attempt_id = prior_attempt_id + 1
    append_jsonl(
        {
            "event": "start",
            "attempt_id": attempt_id,
            "time": time.time(),
            "resume_completed_steps": resume_completed_steps,
            "total_steps": total_steps,
            "cached_gpt_rows": len(cache),
            "cache_tail_repair": cache_repair,
        },
        attempt_path,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stop = False
    stream_position = 0
    try:
        for epoch in range(1, int(args.epochs) + 1):
            for category_index, category in enumerate(categories):
                cpu_bank = banks[category]
                gpu_tokens = cpu_bank.tokens.to(device, non_blocking=False)
                if probe is not None:
                    probe.snapshot(
                        "snapshot_bank_tokens_on_gpu",
                        bank_tensor_bytes=gpu_tokens.numel()
                        * gpu_tokens.element_size(),
                    )
                gpu_bank = CategorySnapshotBank(
                    category=category,
                    item_ids=cpu_bank.item_ids,
                    tokens=gpu_tokens,
                    id_to_index=cpu_bank.id_to_index,
                )
                bank_transposed = (
                    F.normalize(gpu_tokens.float(), dim=-1)
                    .permute(0, 2, 1)
                    .contiguous()
                )
                if probe is not None:
                    probe.snapshot(
                        "snapshot_bank_index_ready",
                        bank_tensor_bytes=gpu_tokens.numel()
                        * gpu_tokens.element_size(),
                        bank_index_bytes=bank_transposed.numel()
                        * bank_transposed.element_size(),
                    )
                dataset = datasets[category]
                generator = torch.Generator().manual_seed(
                    int(args.seed) + epoch * 100 + category_index
                )
                loader = DataLoader(
                    dataset,
                    batch_size=int(args.train_batch_size),
                    shuffle=True,
                    drop_last=True,
                    num_workers=int(args.num_workers),
                    pin_memory=True,
                    persistent_workers=int(args.num_workers) > 0,
                    generator=generator,
                    collate_fn=collate_fn,
                )
                progress = tqdm(
                    loader, desc=f"coarse-to-fine online [{category}]", unit="batch"
                )
                for raw_batch in progress:
                    if stream_position >= total_steps:
                        stop = True
                        break
                    if stream_position < resume_completed_steps:
                        stream_position += 1
                        continue
                    stream_position += 1
                    free_disk_gib = _free_disk_gib(args.output_dir)
                    if free_disk_gib < float(args.min_free_disk_gb):
                        raise RuntimeError(
                            f"Free disk {free_disk_gib:.2f} GiB is below the "
                            f"{float(args.min_free_disk_gb):.2f} GiB safety threshold; "
                            "refusing to spend GPT tokens on another batch"
                        )
                    raw_count = int(raw_batch["ref_img"].shape[0])
                    step_started = time.time()

                    # Mining is current-query/current-shortlist but graph-free.
                    if probe is not None:
                        probe.begin("mining_query_forward", raw_batch_size=raw_count)
                    model.eval()
                    with torch.no_grad(), autocast_context(device):
                        mining_reference = raw_batch["ref_img"].to(
                            device, non_blocking=True
                        )
                        _, _, mining_queries = model.compose_feature(
                            mining_reference, list(raw_batch["edit"]), device=device
                        )
                    if probe is not None:
                        probe.end("mining_query_forward")
                        probe.begin("coarse_retrieval")
                    coarse_started = time.time()
                    snapshot_scores = full_snapshot_scores(
                        mining_queries,
                        bank_tokens_transposed=bank_transposed,
                        token_mat_weight=model.compose_token_mat_weight,
                        eps_temperature=float(args.maxsim_temperature),
                    )
                    shortlists = build_coarse_shortlists(
                        raw_batch,
                        scores=snapshot_scores,
                        bank=gpu_bank,
                        coarse_topk=int(args.coarse_topk),
                        live_shortlist_size=int(args.live_shortlist_size),
                    )
                    coarse_seconds = time.time() - coarse_started
                    if probe is not None:
                        probe.end("coarse_retrieval")
                        probe.begin("live_rerank_no_grad")
                    live_started = time.time()
                    manifests, live_stats = live_rerank(
                        model,
                        query_tokens=mining_queries,
                        shortlists=shortlists,
                        image_loader=image_loader,
                        args=args,
                        device=device,
                    )
                    live_seconds = time.time() - live_started
                    if probe is not None:
                        probe.end(
                            "live_rerank_no_grad",
                            unique_candidates=int(live_stats["live_unique_candidates"]),
                        )
                    for row in shortlists:
                        rank = int(row["snapshot_gt_rank"])
                        key = str(rank if rank <= 100 else ">100")
                        snapshot_gt_rank_hist[key] = (
                            snapshot_gt_rank_hist.get(key, 0) + 1
                        )
                    for rank in live_stats["live_gt_ranks"]:
                        key = str(rank)
                        live_gt_rank_hist[key] = live_gt_rank_hist.get(key, 0) + 1
                    del mining_reference, mining_queries, snapshot_scores
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if probe is not None:
                        probe.snapshot("before_teacher_cache_lookup")

                    # No Student graph is kept alive while external requests run.
                    gpt_started = time.time()
                    requested_teacher_count = len(manifests)
                    next_optimizer_step = int(counters["optimizer_steps"]) + 1
                    preferred_cache_model = (
                        str(args.cache_replay_model)
                        if args.cache_replay_model
                        and next_optimizer_step <= int(args.cache_replay_until_step)
                        else None
                    )
                    if manifests:
                        (
                            successful_manifests,
                            results,
                            cache_hits,
                            api_calls,
                            teacher_failures,
                        ) = request_gpt(
                            manifests,
                            args=args,
                            cache=cache,
                            cache_path=cache_path,
                            preferred_cache_model=preferred_cache_model,
                        )
                        manifests = list(successful_manifests)
                    else:
                        results, cache_hits, api_calls, teacher_failures = [], 0, 0, []
                    gpt_seconds = time.time() - gpt_started

                    optimizer.zero_grad(set_to_none=True)
                    model.train()
                    if probe is not None:
                        probe.begin("gt_forward")
                    gt_loss, gt_accuracy = compute_gt_loss(model, raw_batch, device)
                    weighted_gt = float(args.gt_weight) * gt_loss
                    if probe is not None:
                        probe.end("gt_forward")
                        probe.begin("gt_backward")
                    scaler.scale(weighted_gt).backward()
                    if probe is not None:
                        probe.end("gt_backward", model=model)

                    teacher_count = len(manifests)
                    active_list_sum = 0.0
                    batch_normalized_list = 0.0
                    weighted_list_value = 0.0
                    agreement_sum = 0.0
                    teacher_gt_top1 = 0
                    teacher_gt_present = 0
                    list_started = time.time()
                    micro_size = int(args.list_micro_batch_size)
                    for start in range(0, teacher_count, micro_size):
                        end = min(start + micro_size, teacher_count)
                        scope = f"list_micro_{start // micro_size + 1}"
                        manifest_chunk = manifests[start:end]
                        result_chunk = results[start:end]
                        if probe is not None:
                            probe.current_stage = scope
                            probe.begin(
                                f"{scope}_forward",
                                query_count=len(manifest_chunk),
                                candidate_images=len(manifest_chunk)
                                * int(args.teacher_list_size),
                            )
                        list_loss, agreement, chunk_gt_top1, chunk_gt_present = (
                            compute_live_list_loss(
                                model,
                                raw_batch=raw_batch,
                                manifests=manifest_chunk,
                                results=result_chunk,
                                image_loader=image_loader,
                                device=device,
                                temperature=float(args.opd_temperature),
                            )
                        )
                        active_fraction = len(manifest_chunk) / max(teacher_count, 1)
                        raw_batch_fraction = len(manifest_chunk) / raw_count
                        weighted_list = (
                            float(args.list_weight) * raw_batch_fraction * list_loss
                        )
                        if probe is not None:
                            probe.end(f"{scope}_forward")
                            probe.current_stage = None
                            probe.begin(f"{scope}_backward")
                        scaler.scale(weighted_list).backward()
                        if probe is not None:
                            probe.end(f"{scope}_backward", model=model)
                        active_list_sum += active_fraction * float(list_loss.detach())
                        batch_normalized_list += raw_batch_fraction * float(
                            list_loss.detach()
                        )
                        weighted_list_value += float(weighted_list.detach())
                        agreement_sum += active_fraction * agreement
                        teacher_gt_top1 += chunk_gt_top1
                        teacher_gt_present += chunk_gt_present
                    list_seconds = time.time() - list_started

                    if probe is not None:
                        probe.begin("optimizer_step", model=model, optimizer=optimizer)
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(args.grad_clip_norm)
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    if probe is not None:
                        probe.end("optimizer_step", model=model, optimizer=optimizer)
                        optimizer.zero_grad(set_to_none=True)
                        probe.snapshot(
                            "zero_grad_idle", model=model, optimizer=optimizer
                        )
                        torch.cuda.empty_cache()
                        probe.snapshot(
                            "empty_cache_idle", model=model, optimizer=optimizer
                        )

                    counters["optimizer_steps"] += 1
                    counters["raw_queries"] += raw_count
                    counters["gpt_queries"] += teacher_count
                    counters["gpt_attempted_queries"] += requested_teacher_count
                    counters["gpt_teacher_failures"] += len(teacher_failures)
                    counters["gpt_api_calls"] += api_calls
                    counters["gpt_cache_hits"] += cache_hits
                    for key in (
                        "live_gt_top1_skipped",
                        "live_gt_rank_gate_skipped",
                        "live_score_gap_skipped",
                        "target_in_coarse_topk",
                    ):
                        counters[key] += int(live_stats[key])
                    counters["teacher_gt_top1"] += teacher_gt_top1
                    counters["teacher_gt_present"] += teacher_gt_present
                    counters["live_candidate_encodings_for_mining"] += int(
                        live_stats["live_unique_candidates"]
                    )
                    counters["live_candidate_encodings_for_loss"] += teacher_count * 10

                    weighted_gt_value = float(weighted_gt.detach())
                    weighted_total_value = weighted_gt_value + weighted_list_value
                    live_top1_rate = (
                        float(live_stats["live_gt_top1_skipped"]) / raw_count
                    )
                    sums["gt_loss"] += float(gt_loss.detach())
                    sums["active_list_loss"] += active_list_sum
                    sums["batch_normalized_list_loss"] += batch_normalized_list
                    sums["weighted_gt"] += weighted_gt_value
                    sums["weighted_list"] += weighted_list_value
                    sums["weighted_total"] += weighted_total_value
                    sums["inbatch_gt_accuracy"] += gt_accuracy
                    sums["student_teacher_top1_agreement"] += agreement_sum
                    sums["live_top1_rate"] += live_top1_rate

                    event = {
                        "version": VERSION,
                        "attempt_id": attempt_id,
                        "resumed_from_step": resume_completed_steps,
                        "epoch": epoch,
                        "category": category,
                        "optimizer_step": counters["optimizer_steps"],
                        "total_optimizer_steps": total_steps,
                        "progress_percent": (
                            100.0 * counters["optimizer_steps"] / total_steps
                        ),
                        "free_disk_gib_before_batch": free_disk_gib,
                        "raw_queries": raw_count,
                        "gpt_queries": teacher_count,
                        "gpt_attempted_queries": requested_teacher_count,
                        "gpt_teacher_failures": len(teacher_failures),
                        "gpt_teacher_failure_query_ids": [
                            str(result.get("query_id")) for result in teacher_failures
                        ],
                        "teacher_request_model": str(args.model),
                        "preferred_cache_model": preferred_cache_model,
                        "teacher_result_models": {
                            model_name: sum(
                                str(result.get("model")) == model_name
                                for result in results
                            )
                            for model_name in sorted(
                                {str(result.get("model")) for result in results}
                            )
                        },
                        "gpt_query_rate": teacher_count / raw_count,
                        "gpt_api_calls": api_calls,
                        "gpt_cache_hits": cache_hits,
                        "snapshot_bank_size": len(gpu_bank.item_ids),
                        "coarse_topk": int(args.coarse_topk),
                        "live_shortlist_size": int(args.live_shortlist_size),
                        "target_in_coarse_topk": int(
                            live_stats["target_in_coarse_topk"]
                        ),
                        "live_gt_top1_skipped": int(live_stats["live_gt_top1_skipped"]),
                        "live_gt_rank_gate_skipped": int(
                            live_stats["live_gt_rank_gate_skipped"]
                        ),
                        "live_score_gap_skipped": int(
                            live_stats["live_score_gap_skipped"]
                        ),
                        "live_unique_candidates": int(
                            live_stats["live_unique_candidates"]
                        ),
                        "gt_loss": float(gt_loss.detach()),
                        "active_list_loss": active_list_sum,
                        "batch_normalized_list_loss": batch_normalized_list,
                        "weighted_gt": weighted_gt_value,
                        "weighted_list": weighted_list_value,
                        "weighted_total": weighted_total_value,
                        "inbatch_gt_accuracy": gt_accuracy,
                        "student_teacher_top1_agreement": agreement_sum,
                        "teacher_gt_top1": teacher_gt_top1,
                        "teacher_gt_present": teacher_gt_present,
                        "combined_parameter_grad_norm": float(grad_norm),
                        "coarse_mining_seconds": coarse_seconds,
                        "live_rerank_seconds": live_seconds,
                        "gpt_seconds": gpt_seconds,
                        "gradient_list_seconds": list_seconds,
                        "step_seconds": time.time() - step_started,
                        "list_reduction": "sum(active per-query losses) / raw_batch_size",
                        "candidate_embedding_policy": (
                            "snapshot IDs for coarse retrieval; current image embeddings "
                            "for live reranking and gradient loss"
                        ),
                    }
                    append_jsonl(event, event_path)
                    current_step = int(counters["optimizer_steps"])
                    milestone_percent = milestone_by_step.get(current_step)
                    if (
                        milestone_percent is not None
                        and milestone_percent > last_checkpoint_percent
                    ):
                        elapsed_training_sec = (
                            prior_elapsed_sec + time.time() - session_started
                        )
                        save_rolling_checkpoint(
                            rolling_path,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            args=args,
                            completed_steps=current_step,
                            total_steps=total_steps,
                            milestone_percent=milestone_percent,
                            counters=counters,
                            sums=sums,
                            live_gt_rank_hist=live_gt_rank_hist,
                            snapshot_gt_rank_hist=snapshot_gt_rank_hist,
                            elapsed_training_sec=elapsed_training_sec,
                        )
                        last_checkpoint_percent = milestone_percent
                        print(
                            "[RollingCheckpoint] "
                            + json.dumps(
                                {
                                    "path": str(rolling_path.resolve()),
                                    "completed_steps": current_step,
                                    "total_steps": total_steps,
                                    "milestone_percent": milestone_percent,
                                    "elapsed_training_sec": elapsed_training_sec,
                                    "free_disk_gib": _free_disk_gib(args.output_dir),
                                }
                            ),
                            flush=True,
                        )
                    if (
                        counters["optimizer_steps"] == 1
                        or counters["optimizer_steps"] % max(int(args.log_every), 1)
                        == 0
                    ):
                        print(
                            "[OnlineBatch] " + json.dumps(event, ensure_ascii=False),
                            flush=True,
                        )
                    progress.set_postfix(
                        step=counters["optimizer_steps"],
                        gpt=teacher_count,
                        liveR1=f"{100.0 * live_top1_rate:.1f}",
                        gt=f"{float(gt_loss.detach()):.3f}",
                        list=f"{batch_normalized_list:.3f}",
                    )
                    del gt_loss, weighted_gt, grad_norm, manifests, results, shortlists
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if probe is not None:
                        probe.snapshot(
                            "post_step_cleanup_idle", model=model, optimizer=optimizer
                        )
                        if int(args.max_train_batches) == 1:
                            from blip2_maxsim.training.memory_probe import (
                                offload_optimizer_state,
                            )

                            offloaded_bytes = offload_optimizer_state(optimizer)
                            torch.cuda.empty_cache()
                            probe.snapshot(
                                "optimizer_offloaded_idle",
                                model=model,
                                optimizer=optimizer,
                                offloaded_bytes=offloaded_bytes,
                            )
                del gpu_bank, gpu_tokens, bank_transposed
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if probe is not None:
                    probe.snapshot(
                        "snapshot_bank_released_idle", model=model, optimizer=optimizer
                    )
                if stop:
                    break
            if stop:
                break
    finally:
        image_loader.close()

    steps = max(counters["optimizer_steps"], 1)
    summary = {
        **counters,
        **{key: value / steps for key, value in sums.items()},
        "gpt_query_rate": counters["gpt_queries"] / max(counters["raw_queries"], 1),
        "target_in_coarse_topk_rate": counters["target_in_coarse_topk"]
        / max(counters["raw_queries"], 1),
        "teacher_gt_top1_rate": counters["teacher_gt_top1"]
        / max(counters["teacher_gt_present"], 1),
        "snapshot_gt_rank_histogram": snapshot_gt_rank_hist,
        "live_gt_rank_histogram": live_gt_rank_hist,
        "epoch_time_sec": prior_elapsed_sec + time.time() - session_started,
        "total_optimizer_steps": total_steps,
        "resumed_from_step": resume_completed_steps,
        "attempt_id": attempt_id,
        "rolling_checkpoint": str(rolling_path.resolve()),
        "last_checkpoint_percent": last_checkpoint_percent,
        "cache_tail_repair": cache_repair,
        "train_batch_size": int(args.train_batch_size),
        "list_micro_batch_size": int(args.list_micro_batch_size),
        "coarse_topk": int(args.coarse_topk),
        "live_shortlist_size": int(args.live_shortlist_size),
        "teacher_list_size": int(args.teacher_list_size),
        "list_reduction": "raw_batch_mean_with_inactive_queries_as_zero",
        "candidate_embeddings_updated": True,
        "persistent_image_or_vit_cache": False,
        "pair_loss": 0.0,
    }
    if device.type == "cuda":
        summary["peak_allocated_gib"] = (
            probe.global_peak_allocated
            if probe is not None
            else torch.cuda.max_memory_allocated(device)
        ) / 1024**3
        summary["peak_reserved_gib"] = (
            probe.global_peak_reserved
            if probe is not None
            else torch.cuda.max_memory_reserved(device)
        ) / 1024**3
    append_jsonl(
        {
            "event": "training_complete",
            "attempt_id": attempt_id,
            "time": time.time(),
            "completed_steps": int(counters["optimizer_steps"]),
            "total_steps": total_steps,
            "rolling_checkpoint": str(rolling_path.resolve()),
        },
        attempt_path,
    )
    return summary


def validate_configuration(args: argparse.Namespace) -> None:
    manifest = validate_anchor_manifest(
        args.train_anchor_dir, base_checkpoint=args.base_checkpoint
    )
    smoke_ranking = " > ".join(
        f"[{label}]" for label in labels_for_count(int(args.teacher_list_size))
    )
    parsed, complete, missing = parse_ranking(
        smoke_ranking, candidate_count=int(args.teacher_list_size)
    )
    payload = {
        "version": VERSION,
        "dataset_name": str(args.dataset_name),
        "base_checkpoint": str(args.base_checkpoint),
        "initial_delta": str(args.initial_delta) if args.initial_delta else None,
        "snapshot_bank": str(args.train_anchor_dir),
        "snapshot_bank_by_category": manifest.get("bank_by_category"),
        "candidate_flow": [
            f"snapshot full-bank Top-{int(args.coarse_topk)} IDs",
            f"current-student live Top-{int(args.live_shortlist_size)} reranking",
            f"GT plus {int(args.teacher_list_size) - 1} current hard negatives",
            "online GPT permutation",
            f"current-student teacher-ranked Top-{int(args.teacher_list_size)} candidate re-encoding with gradients",
        ],
        "candidate_embeddings_updated": True,
        "persistent_image_or_vit_cache": False,
        "train_batch_size": int(args.train_batch_size),
        "list_micro_batch_size": int(args.list_micro_batch_size),
        "list_reduction": "divide active OPD loss sum by original BS=64",
        "loss_weights": {"gt": args.gt_weight, "list": args.list_weight, "pair": 0.0},
        "teacher_gate": {
            "skip_live_gt_top1": True,
            "min_gt_rank": int(args.teacher_min_gt_rank),
            "max_gt_rank": int(args.teacher_max_gt_rank),
            "max_top1_minus_gt_score": float(args.teacher_max_score_gap),
        },
        "gpt": {
            "model": args.model,
            "concurrency": int(args.workers),
            "api_format": str(args.api_format),
            "enable_thinking": bool(args.enable_thinking),
            "prompt_version": prompt_version(
                str(args.dataset_name), int(args.teacher_list_size)
            ),
            "cache_replay_model": args.cache_replay_model,
            "cache_replay_until_step": int(args.cache_replay_until_step),
            "parser_smoke": {
                "ranking": parsed,
                "complete": complete,
                "missing": missing,
            },
        },
        "fault_tolerance": {
            "resume": bool(args.resume),
            "rolling_checkpoint": str(
                (args.output_dir / "rolling_checkpoint.pt").resolve()
            ),
            "checkpoint_percent_interval": int(args.checkpoint_percent_interval),
            "atomic_overwrite": True,
            "gpt_cache": "append-only per completed teacher response",
            "min_free_disk_gib_before_gpt_batch": float(args.min_free_disk_gb),
        },
        "validation": "fresh current-model embeddings over full FashionIQ val bank only",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_training(args: argparse.Namespace) -> None:
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        {
            "version": VERSION,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "dataset_root_resolved_images"
            },
        },
        args.output_dir / "run_config.json",
    )
    device = torch.device(args.device)
    probe = None
    if os.environ.get("FASTOPD_STUDENT_MEMORY_PROBE") == "1":
        if device.type != "cuda":
            raise RuntimeError("Student memory probe requires CUDA")
        from blip2_maxsim.training.memory_probe import StudentMemoryProbe
        from blip2_maxsim.training.memory_probe import set_active_probe

        probe = StudentMemoryProbe(
            args.output_dir / "student_memory_events.jsonl", device
        )
        set_active_probe(probe)
    model = build_model(args, device)
    if probe is not None:
        probe.snapshot("weight_loaded", model=model)
    optimizer = build_optimizer(model, args)
    if probe is not None:
        probe.snapshot("optimizer_created", model=model, optimizer=optimizer)

    preflight = {}
    if not args.skip_preflight_eval:
        preflight_path = args.output_dir / "preflight_full_bank_validation.json"
        if preflight_path.is_file():
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            if preflight.get("candidate_pool") != "full_bank":
                raise ValueError(
                    f"Existing preflight result is not full-bank: {preflight_path}"
                )
            print(
                "[PreflightFullBank] reused "
                + json.dumps(preflight, ensure_ascii=False),
                flush=True,
            )
        else:
            preflight = evaluate_current_full_bank(model, args)
            atomic_json(preflight, preflight_path)
            print(
                "[PreflightFullBank] " + json.dumps(preflight, ensure_ascii=False),
                flush=True,
            )

    banks, bank_manifest = load_snapshot_banks(
        args.train_anchor_dir,
        base_checkpoint=args.base_checkpoint,
        categories=parse_dataset_categories(args.categories, args.dataset_name),
    )
    if probe is not None:
        probe.snapshot("snapshot_bank_loaded_cpu", model=model, optimizer=optimizer)
    train_summary = run_epoch(model, optimizer, args, device, banks, probe=probe)
    del banks
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Persist the trained state before the comparatively long full-bank
    # validation.  A validation/data error must not discard a completed OPD
    # epoch.  The final summary is still written only after evaluation.
    checkpoint = None
    if not args.skip_checkpoint:
        checkpoint = args.output_dir / "epoch_001_delta.pt"
        save_delta(
            checkpoint,
            model,
            args,
            {
                "version": VERSION,
                "experiment": args.experiment,
                "train": train_summary,
                "validation_pending": not bool(args.skip_eval),
            },
        )
        print(f"[Checkpoint] saved before validation: {checkpoint}", flush=True)

    validation = {}
    if not args.skip_eval:
        validation = evaluate_current_full_bank(model, args)
        print(
            "[FullBankValidation] " + json.dumps(validation, ensure_ascii=False),
            flush=True,
        )
    summary: dict[str, Any] = {
        "version": VERSION,
        "dataset_name": str(args.dataset_name),
        "experiment": args.experiment,
        "strict_online": True,
        "coarse_to_fine": True,
        "loss_weights": {
            "gt_maxsim": float(args.gt_weight),
            "gpt_listwise": float(args.list_weight),
            "pairwise": 0.0,
        },
        "snapshot_manifest": {
            "path": str((args.train_anchor_dir / "manifest.json").resolve()),
            "checkpoint": bank_manifest.get("fm_ckpt_path"),
            "bank_by_category": bank_manifest.get("bank_by_category"),
        },
        "candidate_policy": {
            "coarse": "fixed warmup full-bank IDs only",
            "live_rerank": "current model re-encodes Top-20 images",
            "gradient": f"current model re-encodes teacher-ranked Top-{int(args.teacher_list_size)} images",
            "persistent_image_or_vit_cache": False,
        },
        "list_reduction": "active OPD query losses divided by original query BS=64",
        "preflight_validation": preflight,
        "train": train_summary,
        "validation": validation,
    }
    if checkpoint is not None:
        summary["checkpoint"] = str(checkpoint.resolve())
    atomic_json(summary, args.output_dir / "summary.json")
    print("[Summary] " + json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    for name in ("dataset_root", "base_checkpoint", "train_anchor_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.initial_delta is not None:
        args.initial_delta = args.initial_delta.expanduser().resolve()
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    if args.initial_delta is not None and not args.initial_delta.is_file():
        raise FileNotFoundError(args.initial_delta)
    if int(args.epochs) != 1:
        raise ValueError("This first experiment is restricted to one epoch")
    if int(args.train_batch_size) != 64:
        raise ValueError("This experiment requires raw GT batch size 64")
    if int(args.teacher_list_size) not in (5, 10, 15):
        raise ValueError("teacher_list_size must be one of 5, 10, or 15")
    if (
        not int(args.teacher_list_size)
        <= int(args.live_shortlist_size)
        <= int(args.coarse_topk)
    ):
        raise ValueError(
            "Require teacher_list_size <= live_shortlist_size <= coarse_topk"
        )
    if not (
        2
        <= int(args.teacher_min_gt_rank)
        <= int(args.teacher_max_gt_rank)
        <= int(args.live_shortlist_size)
    ):
        raise ValueError(
            "Teacher GT-rank gate must lie inside [2, live_shortlist_size]"
        )
    if not math.isclose(
        float(args.gt_weight) + float(args.list_weight), 1.0, abs_tol=1e-8
    ):
        raise ValueError("GT and List weights must sum to one")
    if (
        int(args.checkpoint_percent_interval) <= 0
        or 100 % int(args.checkpoint_percent_interval) != 0
    ):
        raise ValueError(
            "Checkpoint percent interval must be a positive divisor of 100"
        )
    if float(args.min_free_disk_gb) <= 0:
        raise ValueError("Minimum free-disk threshold must be positive")
    if bool(args.cache_replay_model) != (int(args.cache_replay_until_step) > 0):
        raise ValueError(
            "Cache replay requires both --cache-replay-model and a positive "
            "--cache-replay-until-step"
        )
    positive = (
        args.list_micro_batch_size,
        args.candidate_encode_batch_size,
        args.num_workers,
        args.image_loader_workers,
        args.workers,
        args.max_attempts,
    )
    if min(int(value) for value in positive) < 1:
        raise ValueError("Batch, worker, and retry values must be positive")
    set_seed(int(args.seed))
    layout = resolve_fashioniq_layout(args.dataset_root)
    args.dataset_root_resolved_images = layout.images_dir
    validate_configuration(args)
    if args.phase == "validate":
        return
    run_training(args)


if __name__ == "__main__":
    main()
