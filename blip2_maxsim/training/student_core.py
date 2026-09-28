"""Stage-1 online preference distillation with student-mined GPT Top-10 ranks."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from blip2_maxsim.data.common import parse_categories
from blip2_maxsim.data import (
    FashionIQDataModule,
    FashionIQQueryDataset,
    FashionIQTargetDataset,
    build_transform,
    resolve_fashioniq_image_path,
)
from blip2_maxsim.training.evaluation import evaluate_fashioniq
from blip2_maxsim.teacher.ranking import (
    PROMPT_VERSION,
    candidate_set_sha256,
    parse_ranking,
    rank_one,
    request_fingerprint,
)
from blip2_maxsim.model.matching import aligned_maxsim_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "blip2_maxsim_stage1_online_opd_aligned_bs64_listmb32_v1"
DEFAULT_DATASET = Path(os.environ.get("FASTOPD_DATASET_ROOT", "data/FashionIQ"))
DEFAULT_BASE = Path(
    os.environ.get("FASTOPD_BASE_CHECKPOINT", "checkpoints/warmup_end.pt")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--phase",
        choices=("validate", "mine", "teacher", "train", "all"),
        default="validate",
    )
    parser.add_argument(
        "--experiment",
        default="stage1_online_opd_aligned_bs64_listmb32_gt08_list02",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--initial-delta",
        type=Path,
        default=None,
        help=(
            "Optional trainable-state delta applied after the warmup checkpoint. "
            "The default is intentionally empty so online OPD is a fair Stage-1 "
            "alternative to offline distillation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/stage1_online_opd_aligned_bs64_listmb32_v1",
    )
    parser.add_argument("--categories", default="dress,shirt,toptee")
    parser.add_argument("--image-size", type=int, default=364)
    parser.add_argument("--mine-batch-size", type=int, default=16)
    parser.add_argument("--bank-batch-size", type=int, default=64)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help=(
            "Shared logical query batch for GT and List-wise; GT uses it physically, "
            "while List-wise covers it with list_micro_batch_size chunks."
        ),
    )
    parser.add_argument(
        "--list-micro-batch-size",
        type=int,
        default=32,
        help=(
            "Physical query batch for List-wise candidate encoding. Gradients are "
            "accumulated across all micro-batches before one optimizer step."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-mining-queries", type=int, default=0)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--max-logical-steps", type=int, default=0)
    parser.add_argument("--gt-weight", type=float, default=0.8)
    parser.add_argument("--list-weight", type=float, default=0.2)
    parser.add_argument("--qformer-lr", type=float, default=5e-6)
    parser.add_argument("--head-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--rank-temperature", type=float, default=0.07)
    parser.add_argument("--opd-temperature", type=float, default=0.07)
    parser.add_argument("--maxsim-temperature", type=float, default=1e-2)
    parser.add_argument(
        "--candidate-pool",
        choices=("retrack_val_split", "full_bank"),
        default="retrack_val_split",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")

    # These defaults intentionally match the completed offline GPT listwise run.
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("FASTOPD_API_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--trust-env", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--image-detail", choices=("auto", "low", "high"), default="auto"
    )
    parser.add_argument("--max-image-size", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--gpt-limit", type=int, default=0)
    parser.add_argument(
        "--parse-policy", choices=("strict", "paper_fallback"), default="strict"
    )
    parser.add_argument(
        "--execute-gpt",
        action="store_true",
        help="Required before any paid/external GPT request is sent.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def append_jsonl(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def stat_fingerprint(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return contextlib.nullcontext()


def build_model(args: argparse.Namespace, device: torch.device):
    from blip2_maxsim.model.maxsim import BLIP2MaxSim

    model = BLIP2MaxSim(
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
    base_info = model.load_model_state(str(args.base_checkpoint))
    initial_delta_keys = 0
    if args.initial_delta is not None:
        delta = torch.load(str(args.initial_delta), map_location="cpu")
        if not isinstance(delta, dict):
            raise RuntimeError("Initial delta checkpoint must be a dictionary")
        state = delta.get("trainable_state", delta.get("model_state"))
        if not isinstance(state, Mapping):
            raise RuntimeError("Initial delta has no trainable_state/model_state")
        loaded = model.load_state_dict(state, strict=False)
        if loaded.unexpected_keys:
            raise RuntimeError(
                f"Unexpected initial-delta keys: {loaded.unexpected_keys[:10]}"
            )
        initial_delta_keys = len(state)
    model = model.to(device)
    print(
        "[Checkpoint] "
        + json.dumps(
            {
                "base": str(args.base_checkpoint),
                "initial_delta": (
                    str(args.initial_delta) if args.initial_delta else None
                ),
                "base_missing": len(base_info["missing_keys"]),
                "base_unexpected": len(base_info["unexpected_keys"]),
                "initial_delta_keys": initial_delta_keys,
            }
        ),
        flush=True,
    )
    return model


def build_optimizer(model, args: argparse.Namespace):
    qformer = []
    heads = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (qformer if name.startswith("Qformer.") else heads).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": qformer, "lr": float(args.qformer_lr), "group_name": "qformer"},
            {"params": heads, "lr": float(args.head_lr), "group_name": "heads"},
        ],
        weight_decay=float(args.weight_decay),
    )
    return optimizer


def mining_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "version": VERSION,
        "base": stat_fingerprint(args.base_checkpoint),
        "initial_delta": (
            stat_fingerprint(args.initial_delta)
            if args.initial_delta is not None
            else None
        ),
        "categories": parse_categories(args.categories),
        "image_size": int(args.image_size),
        "maxsim_temperature": float(args.maxsim_temperature),
        "topk": 10,
        "split": "train",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def load_latest(
    path: Path, *, fingerprint: str | None = None
) -> Dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output = {}
    for row in iter_jsonl(path):
        if fingerprint is not None and row.get("mining_fingerprint") != fingerprint:
            raise ValueError(f"Mining cache settings mismatch: {path}")
        output[str(row["query_id"])] = row
    return output


@torch.inference_mode()
def encode_target_bank(
    model,
    dataset: FashionIQTargetDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
    )
    parts = []
    model.eval()
    for batch in tqdm(loader, desc=f"online bank [{dataset.dress_type}]", unit="batch"):
        images = batch["tar_img"].to(device, non_blocking=True)
        with autocast_context(device):
            _, tokens, _ = model.target_fea(images, device=device)
        parts.append(F.normalize(tokens.float(), dim=-1).cpu().half())
    return torch.cat(parts).to(device, dtype=torch.float32)


@torch.inference_mode()
def mine_student_top10(model, args: argparse.Namespace) -> Dict[str, object]:
    """Mine Top-10 with the current student; do not persist any feature bank."""
    cache_path = args.output_dir / "student_top10_cache.jsonl"
    output_path = args.output_dir / "student_top10.jsonl"
    fingerprint = mining_fingerprint(args)
    cached = load_latest(cache_path, fingerprint=fingerprint)
    transform = build_transform(int(args.image_size))
    categories = parse_categories(args.categories)
    global_position = 0
    expected_query_ids = []
    model.eval()
    for category in categories:
        query_dataset = FashionIQQueryDataset(
            dataset_root=args.dataset_root,
            split="train",
            categories=[category],
            transform=transform,
            max_rows=0,
        )
        bank_dataset = FashionIQTargetDataset(
            dataset_root=args.dataset_root,
            split="train",
            dress_type=category,
            transform=transform,
        )
        bank_ids = [str(value) for value in bank_dataset.item_ids]
        bank_lookup = {item_id: index for index, item_id in enumerate(bank_ids)}
        if len(bank_lookup) != len(bank_ids):
            raise RuntimeError(f"Duplicate train-bank IDs for {category}")
        bank_tokens = encode_target_bank(
            model,
            bank_dataset,
            batch_size=int(args.bank_batch_size),
            workers=int(args.num_workers),
            device=torch.device(args.device),
        )
        loader = DataLoader(
            query_dataset,
            batch_size=int(args.mine_batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
        )
        for batch in tqdm(loader, desc=f"online queries [{category}]", unit="batch"):
            keep = []
            for local_index, pair_id in enumerate(batch["pair_id"].tolist()):
                query_id = f"train:{category}:{int(pair_id)}"
                if args.max_mining_queries > 0 and global_position >= int(
                    args.max_mining_queries
                ):
                    break
                global_position += 1
                expected_query_ids.append(query_id)
                if query_id not in cached:
                    keep.append(local_index)
            if args.max_mining_queries > 0 and global_position > int(
                args.max_mining_queries
            ):
                break
            if not keep:
                if args.max_mining_queries > 0 and global_position >= int(
                    args.max_mining_queries
                ):
                    break
                continue
            keep_tensor = torch.tensor(keep, dtype=torch.long)
            images = (
                batch["ref_img"]
                .index_select(0, keep_tensor)
                .to(args.device, non_blocking=True)
            )
            edits = [str(batch["edit"][index]) for index in keep]
            with autocast_context(torch.device(args.device)):
                _, _, query_tokens = model.compose_feature(
                    images, edits, device=args.device
                )
            # Autocast returns fp16 query tokens while the transient bank is fp32.
            # MaxSim's matrix multiplication requires both operands to share dtype.
            scores = model.fine_grained_sim(query_tokens.float(), bank_tokens)
            for result_row, source_index in enumerate(keep):
                reference_id = str(batch["candidate_id"][source_index])
                target_id = str(batch["target_id"][source_index])
                reference_index = bank_lookup.get(reference_id)
                if reference_index is not None:
                    scores[result_row, reference_index] = -1.0e10
                score_row = torch.from_numpy(scores[result_row])
                top = torch.topk(score_row, k=10)
                candidates = [
                    {
                        "candidate_id": bank_ids[int(index)],
                        "student_rank": rank,
                        "student_score": float(score),
                        "is_gt": bank_ids[int(index)] == target_id,
                    }
                    for rank, (score, index) in enumerate(
                        zip(top.values.tolist(), top.indices.tolist()), start=1
                    )
                ]
                pair_id = int(batch["pair_id"][source_index])
                query_id = f"train:{category}:{pair_id}"
                row = {
                    "version": VERSION,
                    "mining_fingerprint": fingerprint,
                    "query_id": query_id,
                    "category": category,
                    "ref_id": reference_id,
                    "target_id": target_id,
                    "edit": str(batch["edit"][source_index]),
                    "candidates": candidates,
                    "student_top1_is_gt": bool(candidates[0]["is_gt"]),
                    "student_gt_rank": next(
                        (item["student_rank"] for item in candidates if item["is_gt"]),
                        None,
                    ),
                }
                append_jsonl(row, cache_path)
                cached[query_id] = row
        del bank_tokens
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        if args.max_mining_queries > 0 and global_position >= int(
            args.max_mining_queries
        ):
            break

    rows = [cached[query_id] for query_id in expected_query_ids]
    atomic_jsonl(rows, output_path)
    summary = {
        "version": VERSION,
        "complete": len(rows) == len(expected_query_ids),
        "mining_fingerprint": fingerprint,
        "queries": len(rows),
        "gt_top1_skipped": sum(bool(row["student_top1_is_gt"]) for row in rows),
        "gpt_required": sum(not bool(row["student_top1_is_gt"]) for row in rows),
        "student_gt_in_top10": sum(row["student_gt_rank"] is not None for row in rows),
        "feature_cache_persisted": False,
        "student_top10_file": str(output_path),
    }
    summary["skip_rate"] = summary["gt_top1_skipped"] / max(summary["queries"], 1)
    atomic_json(summary, args.output_dir / "mining_summary.json")
    print("[MiningSummary] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run_gpt_teacher(args: argparse.Namespace) -> bool:
    mined_path = args.output_dir / "student_top10.jsonl"
    if not mined_path.is_file():
        raise FileNotFoundError(f"Run --phase mine first: {mined_path}")
    all_rows = list(iter_jsonl(mined_path))
    teacher_rows = [row for row in all_rows if not bool(row["student_top1_is_gt"])]
    cache_path = args.output_dir / "gpt_rankings_cache.jsonl"
    cached = load_latest(cache_path)

    def valid_cached(row: Mapping[str, Any]) -> bool:
        query_id = str(row["query_id"])
        item = cached.get(query_id)
        return bool(
            item
            and item.get("status") == "ok"
            and item.get("fingerprint") == request_fingerprint(row, args)
            and item.get("candidate_set_sha256") == candidate_set_sha256(row)
        )

    pending = [row for row in teacher_rows if not valid_cached(row)]
    plan = {
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "student_queries": len(all_rows),
        "gt_top1_skipped": len(all_rows) - len(teacher_rows),
        "gpt_required": len(teacher_rows),
        "gpt_cached": len(teacher_rows) - len(pending),
        "gpt_pending": len(pending),
        "model": args.model,
        "base_url": args.base_url,
        "execute_gpt": bool(args.execute_gpt),
    }
    atomic_json(plan, args.output_dir / "teacher_plan.json")
    print("[TeacherPlan] " + json.dumps(plan, ensure_ascii=False), flush=True)
    if not args.execute_gpt:
        print(
            "No GPT request sent. Add --execute-gpt to run the teacher phase.",
            flush=True,
        )
        return False
    if not os.environ.get(args.api_key_env):
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set")
    if args.gpt_limit > 0:
        pending = pending[: int(args.gpt_limit)]
    if pending:
        with cache_path.open("a", encoding="utf-8") as cache_handle:
            with ThreadPoolExecutor(
                max_workers=min(int(args.workers), len(pending))
            ) as executor:
                futures = {}
                for row in pending:
                    category = str(row["category"])
                    image_resolver = (
                        lambda item_id, category=category: resolve_fashioniq_image_path(
                            args.dataset_root_resolved_images, item_id, category
                        )
                    )
                    future = executor.submit(
                        rank_one, row, args, image_path_for_id=image_resolver
                    )
                    futures[future] = str(row["query_id"])
                for done, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    cache_handle.write(
                        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    cache_handle.flush()
                    cached[str(result["query_id"])] = result
                    print(
                        f"[GPT {done}/{len(pending)}] {result['query_id']}: {result['status']}",
                        flush=True,
                    )

    successes = []
    for row in teacher_rows:
        if valid_cached(row):
            successes.append(cached[str(row["query_id"])])
    success_lookup = {str(row["query_id"]): row for row in successes}
    ranked_candidates = []
    for manifest in teacher_rows:
        result = success_lookup.get(str(manifest["query_id"]))
        if result is None:
            continue
        original_rank = {
            str(item["candidate_id"]): int(item["student_rank"])
            for item in manifest["candidates"]
        }
        for gpt_rank, candidate_id in enumerate(
            result["ranked_candidate_ids"], start=1
        ):
            ranked_candidates.append(
                {
                    "query_id": str(manifest["query_id"]),
                    "category": str(manifest["category"]),
                    "ref_id": str(manifest["ref_id"]),
                    "target_id": str(manifest["target_id"]),
                    "edit": str(manifest["edit"]),
                    "candidate_id": str(candidate_id),
                    "student_rank": original_rank[str(candidate_id)],
                    "gpt_direct_rank": gpt_rank,
                    "is_gt": str(candidate_id) == str(manifest["target_id"]),
                }
            )
    atomic_jsonl(ranked_candidates, args.output_dir / "ranked_candidates.jsonl")
    summary = {
        **plan,
        "execute_gpt": True,
        "completed_queries": len(successes),
        "missing_queries": len(teacher_rows) - len(successes),
        "complete": len(successes) == len(teacher_rows),
        "ranked_candidate_rows": len(ranked_candidates),
    }
    atomic_json(summary, args.output_dir / "teacher_summary.json")
    print("[TeacherSummary] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return bool(summary["complete"])


class OnlineRankedDataset(Dataset):
    """Online student Top-10 lists after GPT permutation."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        teacher_file: Path,
        transform,
        categories: Sequence[str],
        max_rows: int,
    ) -> None:
        from blip2_maxsim.data.common import resolve_fashioniq_layout

        self.layout = resolve_fashioniq_layout(dataset_root)
        self.transform = transform
        allowed = set(categories)
        grouped: Dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in iter_jsonl(teacher_file):
            if str(row["category"]) in allowed:
                grouped[str(row["query_id"])].append(row)
        self.rows = []
        for query_id in sorted(grouped):
            candidates = sorted(
                grouped[query_id], key=lambda row: int(row["gpt_direct_rank"])
            )
            if len(candidates) != 10 or [
                int(x["gpt_direct_rank"]) for x in candidates
            ] != list(range(1, 11)):
                raise RuntimeError(f"Incomplete GPT permutation: {query_id}")
            first = candidates[0]
            self.rows.append(
                {
                    "query_id": query_id,
                    "category": str(first["category"]),
                    "ref_id": str(first["ref_id"]),
                    "target_id": str(first["target_id"]),
                    "edit": str(first["edit"]),
                    "candidate_ids": [str(x["candidate_id"]) for x in candidates],
                }
            )
        if max_rows > 0:
            self.rows = self.rows[: int(max_rows)]

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, item_id: str, category: str) -> torch.Tensor:
        path = resolve_fashioniq_image_path(self.layout.images_dir, item_id, category)
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        category = str(row["category"])
        candidate_ids = list(row["candidate_ids"])
        return {
            "ref_img": self._image(str(row["ref_id"]), category),
            "tar_img": self._image(str(row["target_id"]), category),
            "candidate_imgs": torch.stack(
                [self._image(item_id, category) for item_id in candidate_ids]
            ),
            "edit": str(row["edit"]),
            "teacher_scores": torch.linspace(1.0, 0.0, 10),
            "query_id": str(row["query_id"]),
        }


def listwise_rpa_loss(
    scores: torch.Tensor, teacher_scores: torch.Tensor, temperature: float
) -> torch.Tensor:
    """MAPLE-style margin-weighted suffix ListNet, without any pair loss."""
    scaled = scores.float() / max(float(temperature), 1.0e-6)
    teacher = teacher_scores.float()
    candidates = scaled.shape[1]
    numerator = scaled.new_zeros(())
    denominator = scaled.new_zeros(())
    for index in range(candidates - 1):
        weight = (teacher[:, index : index + 1] - teacher[:, index + 1 :]).mean(dim=1)
        negative_log_probability = -F.log_softmax(scaled[:, index:], dim=1)[:, 0]
        numerator = numerator + (weight * negative_log_probability).sum()
        denominator = denominator + weight.sum()
    return numerator / denominator.clamp_min(1e-8)


def compute_gt_loss(
    model,
    batch: Dict[str, object],
    device: torch.device,
):
    reference_images = batch["ref_img"].to(device, non_blocking=True)
    target_images = batch["tar_img"].to(device, non_blocking=True)
    edits = list(batch["edit"])
    _, _, query_tokens = model.compose_feature(reference_images, edits, device=device)
    _, target_tokens, _ = model.target_fea(target_images, device=device)
    gt_loss, gt_logits = model.compute_rank_loss(query_tokens, target_tokens)
    batch_size = int(reference_images.shape[0])
    gt_accuracy = float(
        (gt_logits.detach().argmax(dim=1) == torch.arange(batch_size, device=device))
        .float()
        .mean()
    )
    return gt_loss, gt_accuracy


def compute_list_loss(
    model,
    batch: Dict[str, object],
    device: torch.device,
    temperature: float,
):
    """Compute List-wise loss for one aligned query micro-batch."""
    reference_images = batch["ref_img"].to(device, non_blocking=True)
    candidate_images = batch["candidate_imgs"].to(device, non_blocking=True)
    teacher_scores = batch["teacher_scores"].to(device, non_blocking=True)
    edits = list(batch["edit"])
    batch_size, candidates = candidate_images.shape[:2]
    _, _, query_tokens = model.compose_feature(reference_images, edits, device=device)
    _, flat_tokens, _ = model.target_fea(candidate_images.flatten(0, 1), device=device)
    candidate_tokens = flat_tokens.reshape(
        batch_size, candidates, flat_tokens.shape[1], flat_tokens.shape[2]
    )
    scores, _ = aligned_maxsim_scores(
        query_tokens,
        candidate_tokens,
        model.compose_token_mat_weight,
        model.maxsim_temperature,
    )
    list_loss = listwise_rpa_loss(scores, teacher_scores, temperature)
    teacher_top1_accuracy = float((scores.detach().argmax(dim=1) == 0).float().mean())
    return list_loss, teacher_top1_accuracy


def slice_batch(batch: Dict[str, object], start: int, end: int) -> Dict[str, object]:
    sliced: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            sliced[key] = value[start:end]
        elif isinstance(value, (list, tuple)):
            sliced[key] = value[start:end]
        else:
            raise TypeError(f"Cannot slice batch field {key!r}: {type(value).__name__}")
    return sliced


def train_one_epoch(model, train_loader, optimizer, args, device: torch.device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logical_steps = len(train_loader)
    if args.max_logical_steps > 0:
        logical_steps = min(logical_steps, int(args.max_logical_steps))
    sums = defaultdict(float)
    started = time.time()
    seen_steps = 0
    seen_queries = 0
    for step, batch in enumerate(
        tqdm(train_loader, total=logical_steps, desc="stage1 online OPD", unit="step")
    ):
        if step >= logical_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        batch_queries = int(batch["ref_img"].shape[0])
        with autocast_context(device):
            gt_loss, gt_accuracy = compute_gt_loss(model, batch, device)
            weighted_gt = float(args.gt_weight) * gt_loss
        scaler.scale(weighted_gt).backward()

        list_loss_value = 0.0
        teacher_top1_accuracy = 0.0
        list_micro_batches = 0
        micro_size = int(args.list_micro_batch_size)
        for start in range(0, batch_queries, micro_size):
            end = min(start + micro_size, batch_queries)
            micro_batch = slice_batch(batch, start, end)
            fraction = (end - start) / batch_queries
            with autocast_context(device):
                list_loss, micro_top1 = compute_list_loss(
                    model,
                    micro_batch,
                    device,
                    float(args.opd_temperature),
                )
                weighted_list = float(args.list_weight) * fraction * list_loss
            scaler.scale(weighted_list).backward()
            list_loss_value += fraction * float(list_loss.detach())
            teacher_top1_accuracy += fraction * micro_top1
            list_micro_batches += 1

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(args.grad_clip_norm)
        )
        scaler.step(optimizer)
        scaler.update()
        seen_steps += 1
        seen_queries += batch_queries
        sums["gt_loss"] += float(gt_loss.detach())
        sums["list_loss"] += list_loss_value
        sums["gt_accuracy"] += gt_accuracy
        sums["teacher_top1_accuracy"] += teacher_top1_accuracy
        weighted_total = (
            float(args.gt_weight) * float(gt_loss.detach())
            + float(args.list_weight) * list_loss_value
        )
        sums["weighted_total"] += weighted_total
        if step == 0 or (step + 1) % max(int(args.log_every), 1) == 0:
            print(
                "[TrainMetric] "
                + json.dumps(
                    {
                        "step": step + 1,
                        "logical_steps": logical_steps,
                        "aligned_query_batch": batch_queries,
                        "gt_loss": float(gt_loss.detach()),
                        "list_loss": list_loss_value,
                        "gt_accuracy": gt_accuracy,
                        "teacher_top1_accuracy": teacher_top1_accuracy,
                        "list_micro_batch_size": micro_size,
                        "list_micro_batches": list_micro_batches,
                        "grad_norm": float(grad_norm),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    output = {key: value / max(seen_steps, 1) for key, value in sums.items()}
    output.update(
        {
            "logical_steps": seen_steps,
            "aligned_queries_seen": seen_queries,
            "training_queries": len(train_loader.dataset),
            "train_batch_size": int(args.train_batch_size),
            "list_micro_batch_size": int(args.list_micro_batch_size),
            "list_micro_batches_per_step": math.ceil(
                int(args.train_batch_size) / int(args.list_micro_batch_size)
            ),
            "epoch_time_sec": time.time() - started,
            "pair_loss": 0.0,
        }
    )
    if device.type == "cuda":
        output["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        output["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
    return output


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


def run_training(args: argparse.Namespace) -> None:
    teacher_summary_path = args.output_dir / "teacher_summary.json"
    ranked_path = args.output_dir / "ranked_candidates.jsonl"
    if not teacher_summary_path.is_file() or not ranked_path.is_file():
        raise FileNotFoundError("Complete the GPT teacher phase before training")
    teacher_summary = json.loads(teacher_summary_path.read_text())
    if not teacher_summary.get("complete"):
        raise RuntimeError(
            "GPT teacher cache is incomplete; refusing partial OPD training"
        )
    metrics_path = args.output_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed/partial training: {metrics_path}"
        )
    device = torch.device(args.device)
    model = build_model(args, device)
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
    opd_dataset = OnlineRankedDataset(
        dataset_root=args.dataset_root,
        teacher_file=ranked_path,
        transform=data.train_transform,
        categories=data.categories,
        max_rows=int(args.max_train_rows),
    )
    if not opd_dataset:
        raise RuntimeError("No non-GT@1 GPT-ranked queries are available")
    train_loader = DataLoader(
        opd_dataset,
        batch_size=int(args.train_batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
    )
    if not train_loader:
        raise RuntimeError(
            "Fewer aligned OPD queries than train_batch_size; cannot form one full batch"
        )
    optimizer = build_optimizer(model, args)
    train_metrics = train_one_epoch(model, train_loader, optimizer, args, device)
    append_jsonl(train_metrics, metrics_path)
    validation = {}
    if not args.skip_eval:
        validation = evaluate_fashioniq(
            model,
            query_loader=data.eval_query_dataloader(),
            build_target_loader=data.build_eval_target_loader,
            categories=data.categories,
            device=device,
            candidate_pool=args.candidate_pool,
        )
        print("[Validation] " + json.dumps(validation, ensure_ascii=False), flush=True)
    summary = {
        "version": VERSION,
        "experiment": args.experiment,
        "loss_weights": {
            "gt_maxsim": float(args.gt_weight),
            "gpt_listwise": float(args.list_weight),
            "pairwise": 0.0,
        },
        "gt_top1_queries_skipped_entirely": int(teacher_summary["gt_top1_skipped"]),
        "available_gpt_ranked_queries": int(teacher_summary["completed_queries"]),
        "trained_gpt_ranked_queries": len(opd_dataset),
        "query_alignment": "same query batch for GT MaxSim and List-wise",
        "train_batch_size": int(args.train_batch_size),
        "list_micro_batch_size": int(args.list_micro_batch_size),
        "optimizer_steps_per_logical_batch": 1,
        "train": train_metrics,
        "validation": validation,
        "checkpoint": (
            None
            if args.skip_checkpoint
            else str((args.output_dir / "epoch_001_delta.pt").resolve())
        ),
    }
    if not args.skip_checkpoint:
        save_delta(args.output_dir / "epoch_001_delta.pt", model, args, summary)
    atomic_json(summary, args.output_dir / "summary.json")
    print("[Summary] " + json.dumps(summary, ensure_ascii=False), flush=True)


def validate_configuration(args: argparse.Namespace) -> None:
    from blip2_maxsim.data.common import resolve_fashioniq_layout

    layout = resolve_fashioniq_layout(args.dataset_root)
    parsed, complete, missing = parse_ranking(
        "[A] > [B] > [C] > [D] > [E] > [F] > [G] > [H] > [I] > [J]"
    )
    payload = {
        "version": VERSION,
        "phase": "validate",
        "dataset_root": str(args.dataset_root),
        "images_dir": str(layout.images_dir),
        "base_checkpoint_exists": args.base_checkpoint.is_file(),
        "initial_delta": str(args.initial_delta) if args.initial_delta else None,
        "initial_delta_exists": (
            args.initial_delta.is_file() if args.initial_delta is not None else None
        ),
        "initialization": (
            "warmup checkpoint only"
            if args.initial_delta is None
            else "warmup checkpoint plus optional delta"
        ),
        "categories": parse_categories(args.categories),
        "loss": {"gt": args.gt_weight, "listwise": args.list_weight, "pairwise": 0.0},
        "train_batch_size": int(args.train_batch_size),
        "query_alignment": (
            f"same {int(args.train_batch_size)}-query batch for GT MaxSim and List-wise"
        ),
        "list_micro_batch_size": int(args.list_micro_batch_size),
        "list_micro_batches_per_step": math.ceil(
            int(args.train_batch_size) / int(args.list_micro_batch_size)
        ),
        "skip_rule": "student_top1_is_gt -> skip the query entirely",
        "gpt": {
            "model": args.model,
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "prompt_version": PROMPT_VERSION,
            "execute_gpt": bool(args.execute_gpt),
            "parser_smoke": {
                "ranking": parsed,
                "complete": complete,
                "missing": missing,
            },
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.base_checkpoint = args.base_checkpoint.expanduser().resolve()
    if args.initial_delta is not None:
        args.initial_delta = args.initial_delta.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(f"Missing base checkpoint: {args.base_checkpoint}")
    if args.initial_delta is not None and not args.initial_delta.is_file():
        raise FileNotFoundError(f"Missing optional initial delta: {args.initial_delta}")
    if not math.isclose(
        float(args.gt_weight) + float(args.list_weight), 1.0, abs_tol=1e-8
    ):
        raise ValueError("GT and List-wise weights must sum to one")
    if (
        min(
            args.workers,
            args.max_attempts,
            args.mine_batch_size,
            args.bank_batch_size,
            args.train_batch_size,
            args.list_micro_batch_size,
        )
        < 1
    ):
        raise ValueError("worker, attempt, and mining batch values must be positive")
    if int(args.train_batch_size) % int(args.list_micro_batch_size) != 0:
        raise ValueError("train_batch_size must be divisible by list_micro_batch_size")
    set_seed(int(args.seed))
    from blip2_maxsim.data.common import resolve_fashioniq_layout

    layout = resolve_fashioniq_layout(args.dataset_root)
    args.dataset_root_resolved_images = layout.images_dir
    if args.phase == "validate":
        validate_configuration(args)
        return
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
    if args.phase in ("mine", "all"):
        device = torch.device(args.device)
        model = build_model(args, device)
        mine_student_top10(model, args)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.phase == "mine":
            return
    teacher_complete = True
    if args.phase in ("teacher", "all"):
        teacher_complete = run_gpt_teacher(args)
        if args.phase == "teacher" or not teacher_complete:
            return
    if args.phase in ("train", "all"):
        run_training(args)


if __name__ == "__main__":
    main()
