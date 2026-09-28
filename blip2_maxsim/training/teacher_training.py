"""Coarse-to-fine online OPD with a local Qwen3-VL-32B teacher.

This is an isolated teacher adapter around the current GPT coarse-to-fine
implementation.  The Student candidate flow and losses stay identical; only
the online permutation teacher is replaced.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from blip2_maxsim.training import training_loop as base
from blip2_maxsim.training.student_core import atomic_json
from blip2_maxsim.teacher.teacher_client import (
    PROMPT_VERSION,
    Qwen32WorkerClient,
    request_fingerprint as qwen_request_fingerprint,
    request_qwen_chunk,
)
from blip2_maxsim.teacher.ranking import (
    labels_for_count,
    prompt_version as ranking_prompt_version,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = (
    "blip2_maxsim_qwen32_demotion_rpa_competitive_tcb_v4_fixed269_relative_gtcl_v1"
)
DEFAULT_QWEN_MODEL = Path(
    os.environ.get("FASTOPD_QWEN_MODEL", "models/Qwen3-VL-32B-Instruct-FP8")
)
DEFAULT_QWEN_PYTHON = Path(os.environ.get("FASTOPD_QWEN_PYTHON", "python"))
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/full_fashioniq"


def _has_option(arguments: Sequence[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in arguments)


def parse_args() -> argparse.Namespace:
    """Add Qwen controls while inheriting all current Student OPD controls."""
    qwen = argparse.ArgumentParser(add_help=False)
    qwen.add_argument("--execute-qwen", action="store_true")
    qwen.add_argument("--qwen-python", type=Path, default=DEFAULT_QWEN_PYTHON)
    qwen.add_argument("--qwen-model-path", type=Path, default=DEFAULT_QWEN_MODEL)
    qwen.add_argument("--qwen-gpu", type=int, default=1)
    qwen.add_argument("--student-physical-gpu", type=int, default=1)
    qwen.add_argument("--qwen-max-model-len", type=int, default=3072)
    qwen.add_argument("--qwen-max-num-batched-tokens", type=int, default=2048)
    qwen.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.73)
    qwen.add_argument("--qwen-cpu-offload-gb", type=float, default=0.0)
    qwen.add_argument("--qwen-max-num-seqs", type=int, default=1)
    qwen.add_argument("--qwen-image-min-pixels", type=int, default=128 * 28 * 28)
    qwen.add_argument("--qwen-image-max-pixels", type=int, default=512 * 28 * 28)
    qwen.add_argument(
        "--qwen-sleep-between-requests",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    qwen.add_argument("--qwen-max-tokens", type=int, default=128)
    qwen.add_argument("--qwen-temperature", type=float, default=0.0)
    qwen.add_argument("--qwen-top-p", type=float, default=1.0)
    qwen.add_argument("--qwen-inference-batch-size", type=int, default=1)
    qwen.add_argument("--qwen-cmi-inference-batch-size", type=int, default=24)
    qwen.add_argument(
        "--cmi-three-branch", action=argparse.BooleanOptionalAction, default=False
    )
    qwen.add_argument("--ranking-spec-k", type=int, default=0)
    qwen.add_argument("--qwen-startup-timeout", type=float, default=900.0)
    qwen.add_argument(
        "--qwen-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    qwen_args, remaining = qwen.parse_known_args()

    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = original

    if not _has_option(remaining, "--experiment"):
        args.experiment = "full_fashioniq_spec_k3_cmi_branch"
    if not _has_option(remaining, "--output-dir"):
        args.output_dir = DEFAULT_OUTPUT
    for key, value in vars(qwen_args).items():
        setattr(args, key, value)
    if not 0 <= int(args.ranking_spec_k) <= 10:
        raise ValueError("--ranking-spec-k must be between 0 and 10")
    if int(args.qwen_cmi_inference_batch_size) <= 0:
        raise ValueError("--qwen-cmi-inference-batch-size must be positive")

    # The shared loop keeps compatibility field names such as gpt_queries.
    # Only this adapter process sees these assignments.
    args.execute_gpt = bool(args.execute_qwen)
    args.model = str(args.qwen_model_path)
    args.api_key_env = "QWEN32_LOCAL_TEACHER_READY"
    args.cache_replay_model = None
    args.cache_replay_until_step = 0
    args.teacher_cache_filename = "online_qwen32_cache.jsonl"
    return args


def _request_qwen_as_online_teacher(
    manifests: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    cache: dict[tuple[str, str], dict[str, Any]],
    cache_path: Path,
    preferred_cache_model: str | None = None,
    client: Qwen32WorkerClient,
) -> tuple[
    list[Mapping[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    list[dict[str, Any]],
]:
    del preferred_cache_model
    keys = [
        (str(manifest["query_id"]), qwen_request_fingerprint(manifest, args))
        for manifest in manifests
    ]
    cache_hits = sum(
        bool(cache.get(key) and cache[key].get("status") == "ok") for key in keys
    )
    calls = len(manifests) - cache_hits
    try:
        results, cache_hits, calls = request_qwen_chunk(
            manifests,
            args=args,
            cache=cache,
            cache_path=cache_path,
            client=client,
        )
        return list(manifests), results, cache_hits, calls, []
    except RuntimeError as request_error:
        # Match the GPT control: malformed/incomplete permutations are retried
        # by the ranker, then only those queries lose their List-wise term.
        # Worker, CUDA, model, and transport failures remain fatal.
        successful_manifests: list[Mapping[str, Any]] = []
        successful_results: list[dict[str, Any]] = []
        skippable_failures: list[dict[str, Any]] = []
        for manifest, key in zip(manifests, keys):
            result = cache.get(key)
            if result is None:
                raise request_error
            if result.get("status") == "ok":
                successful_manifests.append(manifest)
                successful_results.append(result)
            elif base.is_skippable_teacher_failure(result):
                skippable_failures.append(result)
            else:
                raise request_error
        if len(successful_manifests) + len(skippable_failures) != len(manifests):
            raise request_error
        return (
            successful_manifests,
            successful_results,
            cache_hits,
            calls,
            skippable_failures,
        )


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> int:
    """Move Adam state across the same-GPU Qwen/Student handoff boundary."""
    moved_bytes = 0
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if not torch.is_tensor(value) or value.device == device:
                continue
            moved_bytes += value.numel() * value.element_size()
            state[key] = value.detach().to(device=device, non_blocking=False)
    return moved_bytes


def validate_configuration(args: argparse.Namespace) -> None:
    manifest = base.validate_anchor_manifest(
        args.train_anchor_dir, base_checkpoint=args.base_checkpoint
    )
    smoke_ranking = " > ".join(
        f"[{label}]" for label in labels_for_count(int(args.teacher_list_size))
    )
    parsed, complete, missing = base.parse_ranking(
        smoke_ranking, candidate_count=int(args.teacher_list_size)
    )
    payload = {
        "version": getattr(args, "fastopd_version", VERSION),
        "dataset_name": str(args.dataset_name),
        "base_checkpoint": str(args.base_checkpoint),
        "initial_delta": str(args.initial_delta) if args.initial_delta else None,
        "snapshot_bank": str(args.train_anchor_dir),
        "snapshot_bank_by_category": manifest.get("bank_by_category"),
        "categories": base.parse_dataset_categories(args.categories, args.dataset_name),
        "candidate_flow": [
            f"snapshot full-bank Top-{int(args.coarse_topk)} IDs",
            f"current-student live Top-{int(args.live_shortlist_size)} reranking",
            f"GT plus {int(args.teacher_list_size) - 1} current hard negatives",
            "online Qwen3-VL-32B permutation",
            f"current-student teacher-ranked Top-{int(args.teacher_list_size)} candidate re-encoding with gradients",
        ],
        "candidate_embeddings_updated": True,
        "persistent_image_or_vit_cache": False,
        "train_batch_size": int(args.train_batch_size),
        "list_micro_batch_size": int(args.list_micro_batch_size),
        "list_reduction": "divide active OPD loss sum by original BS=64",
        "loss_weights": {
            "gt": args.gt_weight,
            "list": args.list_weight,
            "transformation_comparison": float(getattr(args, "tcb_weight", 0.0)),
        },
        "teacher_gate": {
            "skip_live_gt_top1": True,
            "min_gt_rank": int(args.teacher_min_gt_rank),
            "max_gt_rank": int(args.teacher_max_gt_rank),
            "max_top1_minus_gt_score": float(args.teacher_max_score_gap),
        },
        "teacher": {
            "backend": "qwen3_vl_32b_vllm_worker",
            "model": str(args.qwen_model_path),
            "python": str(args.qwen_python),
            "physical_gpu": int(args.qwen_gpu),
            "student_physical_gpu": int(args.student_physical_gpu),
            "prompt_version": ranking_prompt_version(
                str(args.dataset_name), int(args.teacher_list_size)
            ),
            "visual_inputs_per_query": 1 + int(args.teacher_list_size),
            "max_model_len": int(args.qwen_max_model_len),
            "max_num_batched_tokens": int(args.qwen_max_num_batched_tokens),
            "inference_batch_size": int(args.qwen_inference_batch_size),
            "cmi_inference_batch_size": int(args.qwen_cmi_inference_batch_size),
            "ranking_spec_k": int(args.ranking_spec_k),
            "cmi_three_branch": bool(args.cmi_three_branch),
            "gpu_memory_utilization": float(args.qwen_gpu_memory_utilization),
            "cpu_offload_gb": float(args.qwen_cpu_offload_gb),
            "sleep_level_1_between_requests": bool(args.qwen_sleep_between_requests),
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
            "qwen_cache": str(
                (args.output_dir / args.teacher_cache_filename).resolve()
            ),
            "cache_policy": "append-only per completed Qwen response; no GPT cache reuse",
            "min_free_disk_gib_before_teacher_batch": float(args.min_free_disk_gb),
        },
        "validation": "fresh current-model embeddings over full FashionIQ val bank only",
        "execute_qwen": bool(args.execute_qwen),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _decorate_summary(args: argparse.Namespace) -> None:
    path = args.output_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["version"] = VERSION
    summary["teacher"] = {
        "backend": "qwen3_vl_32b_vllm_worker",
        "model": str(args.qwen_model_path),
        "physical_gpu": int(args.qwen_gpu),
        "prompt_version": ranking_prompt_version(
            str(args.dataset_name), int(args.teacher_list_size)
        ),
        "memory_mode": "vLLM sleep level 1 between logical requests",
    }
    summary["student_physical_gpu"] = int(args.student_physical_gpu)
    summary["loss_weights"]["qwen32_listwise"] = summary["loss_weights"].pop(
        "gpt_listwise"
    )
    summary["candidate_policy"][
        "gradient"
    ] = f"current model re-encodes Qwen32-ranked Top-{int(args.teacher_list_size)} images"
    train = summary["train"]
    train["qwen_queries"] = train.get("gpt_queries", 0)
    train["qwen_calls"] = train.get("gpt_api_calls", 0)
    train["qwen_cache_hits"] = train.get("gpt_cache_hits", 0)
    train["qwen_query_rate"] = train.get("gpt_query_rate", 0.0)
    train["teacher_backend"] = "qwen3_vl_32b_vllm_worker"
    atomic_json(summary, path)
    print("[QwenSummary] " + json.dumps(summary, ensure_ascii=False), flush=True)


def run_training(
    args: argparse.Namespace,
    *,
    model_builder=None,
    listwise_loss=None,
    live_list_loss=None,
    teacher_request=None,
    summary_hook=None,
    version: str | None = None,
) -> None:
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    teacher = Qwen32WorkerClient(args)
    original_request = base.request_gpt
    original_build_optimizer = base.build_optimizer
    original_build_model = base.build_model
    original_listwise_loss = base.listwise_rpa_loss
    original_live_list_loss = base.compute_live_list_loss
    original_version = base.VERSION
    original_prompt_version = base.PROMPT_VERSION
    previous_ready_value = os.environ.get(args.api_key_env)
    optimizer_holder: dict[str, Any] = {}
    handoff_count = 0

    def build_optimizer_and_remember(*builder_args, **builder_kwargs):
        optimizer = original_build_optimizer(*builder_args, **builder_kwargs)
        optimizer_holder["optimizer"] = optimizer
        return optimizer

    def request_after_releasing_student_gradients(manifests, **kwargs):
        nonlocal handoff_count
        # The shared GPT loop normally clears gradients after its external
        # request. Qwen wakes on the same GPU, so this process-local adapter
        # clears gradients and offloads persistent Adam state before wake-up.
        optimizer = optimizer_holder.get("optimizer")
        if optimizer is None:
            raise RuntimeError(
                "Student optimizer was not initialized before Qwen request"
            )
        optimizer.zero_grad(set_to_none=True)
        offloaded_bytes = _move_optimizer_state(optimizer, torch.device("cpu"))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return (teacher_request or _request_qwen_as_online_teacher)(
                manifests, client=teacher, **kwargs
            )
        finally:
            restored_bytes = _move_optimizer_state(optimizer, torch.device(args.device))
            handoff_count += 1
            if handoff_count <= 3 or handoff_count % 10 == 0:
                print(
                    "[QwenMemoryHandoff] "
                    + json.dumps(
                        {
                            "handoff": handoff_count,
                            "optimizer_state_gib": offloaded_bytes / 1024**3,
                            "restored_gib": restored_bytes / 1024**3,
                        }
                    ),
                    flush=True,
                )

    try:
        # Qwen loads first and then sleeps.  The Student is constructed only
        # after Qwen has released its weights from GPU memory.
        teacher.start()
        base.VERSION = version or VERSION
        base.PROMPT_VERSION = PROMPT_VERSION
        base.build_optimizer = build_optimizer_and_remember
        if model_builder is not None:
            base.build_model = model_builder
        if listwise_loss is not None:
            base.listwise_rpa_loss = listwise_loss
        if live_list_loss is not None:
            base.compute_live_list_loss = live_list_loss
        base.request_gpt = request_after_releasing_student_gradients
        os.environ[args.api_key_env] = "1"
        base.run_training(args)
        (summary_hook or _decorate_summary)(args)
    finally:
        base.request_gpt = original_request
        base.build_optimizer = original_build_optimizer
        base.build_model = original_build_model
        base.listwise_rpa_loss = original_listwise_loss
        base.compute_live_list_loss = original_live_list_loss
        base.VERSION = original_version
        base.PROMPT_VERSION = original_prompt_version
        if previous_ready_value is None:
            os.environ.pop(args.api_key_env, None)
        else:
            os.environ[args.api_key_env] = previous_ready_value
        teacher.close()


def main(training_fn=None, configure_args=None) -> None:
    args = parse_args()
    for name in (
        "dataset_root",
        "base_checkpoint",
        "train_anchor_dir",
        "output_dir",
        "qwen_python",
        "qwen_model_path",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.initial_delta is not None:
        args.initial_delta = args.initial_delta.expanduser().resolve()

    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    if args.initial_delta is not None and not args.initial_delta.is_file():
        raise FileNotFoundError(args.initial_delta)
    if not args.qwen_python.is_file():
        raise FileNotFoundError(args.qwen_python)
    if not args.qwen_model_path.is_dir():
        raise FileNotFoundError(args.qwen_model_path)
    if int(args.epochs) != 1:
        raise ValueError("This experiment is restricted to one epoch")
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
    positive = (
        args.list_micro_batch_size,
        args.candidate_encode_batch_size,
        args.num_workers,
        args.image_loader_workers,
        args.max_attempts,
        args.qwen_inference_batch_size,
        args.qwen_max_model_len,
        args.qwen_max_num_seqs,
        args.qwen_max_num_batched_tokens,
        args.qwen_max_tokens,
    )
    if min(int(value) for value in positive) < 1:
        raise ValueError("Batch, worker, and retry values must be positive")
    if not 0.0 < float(args.qwen_gpu_memory_utilization) <= 1.0:
        raise ValueError("qwen-gpu-memory-utilization must be in (0, 1]")
    if float(args.qwen_cpu_offload_gb) < 0.0:
        raise ValueError("qwen-cpu-offload-gb must be non-negative")
    if args.qwen_sleep_between_requests and not math.isclose(
        float(args.qwen_cpu_offload_gb), 0.0, abs_tol=1e-12
    ):
        raise ValueError("sleep/wake mode requires qwen-cpu-offload-gb=0")
    physical_gpus = (int(args.student_physical_gpu), int(args.qwen_gpu))
    if physical_gpus[0] != physical_gpus[1]:
        raise ValueError(
            "Shared-GPU training requires Student and Qwen on the same physical GPU"
        )
    if args.phase == "train" and os.environ.get("CUDA_VISIBLE_DEVICES") != str(
        args.student_physical_gpu
    ):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must equal the selected Student physical GPU"
        )

    base.set_seed(int(args.seed))
    layout = base.resolve_fashioniq_layout(args.dataset_root)
    args.dataset_root_resolved_images = layout.images_dir
    if configure_args is not None:
        configure_args(args)
    validate_configuration(args)
    if args.phase == "validate":
        return
    if not args.execute_qwen:
        raise RuntimeError("Qwen32 online training requires --execute-qwen")
    (training_fn or run_training)(args)


if __name__ == "__main__":
    main()
