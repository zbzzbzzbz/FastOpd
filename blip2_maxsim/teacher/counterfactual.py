"""Qwen32 joint-three counterfactual generation for Transformation Comparison."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from blip2_maxsim.data.image_resolver import resolve_manifest_image_path
from blip2_maxsim.training.student_core import append_jsonl


VERSION = "qwen32_joint3_counterfactual_v4"
PROMPT_VERSION = "fashioniq_delta_original_edit_joint3_v4_safe"
LABELS = ("A", "B", "C")


def word_budget(original_edit: str) -> int:
    count = len(re.findall(r"\b[\w'-]+\b", str(original_edit)))
    if count <= 8:
        return 12
    if count <= 15:
        return 18
    return 24


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text)))


def instruction_word_budget(original_edit: str, dataset_name: str) -> int:
    """Return the actual acceptance ceiling for one generated instruction."""
    return word_budget(original_edit)


def prompt_version(dataset_name: str) -> str:
    return PROMPT_VERSION


def build_prompt(original_edit: str, dataset_name: str = "fashioniq") -> str:
    budget = instruction_word_budget(original_edit, dataset_name)
    return (
        "The reference image is the original garment.\n"
        "Candidates A, B, and C are three possible target images.\n\n"
        "For each candidate independently, write the modification instruction that changes "
        "the reference garment into that candidate garment.\n\n"
        f"Original modification text: {original_edit}\n"
        f"Hard maximum length for each instruction: {budget} words. Never exceed this limit.\n\n"
        "The original modification text is only (1) a grammar-and-length template and (2) a list "
        "of visual attributes worth checking. It is not guaranteed to describe any candidate. "
        "Never copy an attribute value from it unless the labeled candidate visibly supports that value.\n"
        "For A, B, and C separately, compare only that labeled candidate with the reference image. "
        "Write the smallest set of visible clothing changes needed to reach that candidate. Mention "
        "changed clothing attributes only, omit unchanged attributes, and do not write a full product "
        "description. Preserve a short opening such as 'is', 'has', or 'change to' when the original "
        "modification text uses one.\n"
        "Within the word limit, prefer the most informative clearly visible changes over generic ones. "
        "For example, include a visible neckline, sleeve, pattern, graphic, stripe direction, pocket, "
        "or shape change instead of stopping at a shared color change. If several details are visible, "
        "keep only the most informative ones needed to stay within the hard word limit.\n"
        "Do not copy an instruction mechanically across candidates. Identical instructions are allowed "
        "when the visible transformations are genuinely the same. Never invent or add an uncertain "
        "detail merely to make the outputs different.\n"
        "If a detail is uncertain, omit it. Do not infer front/back placement, exact printed text, "
        "material, or lighting-dependent color unless clearly visible. Ignore the person, pose, "
        "body, background, camera angle, crop, lighting, and image quality. Describe only the garment "
        "being retrieved; ignore other garments and styling accessories worn with it, including ties, "
        "trousers, skirts, belts, and jewelry.\n"
        "Do not use 'maintaining', 'compared with the original', or explanatory/product-description "
        "prose. Each result must be exactly one instruction fragment or sentence.\n\n"
        "Formatting examples only; do not copy their clothing attributes:\n"
        "- Original modification text: 'is darker'. Candidate's visible change: black color and "
        "larger white logo. Output: 'is black and has a larger white logo'\n"
        "- Original modification text: 'has longer sleeves and two stripes'. Candidate's visible "
        "change: short sleeves and green floral print. Output: 'has shorter sleeves and a green "
        "floral print'\n"
        "- Original modification text: 'has text on it and has a larger logo'. Candidate's visible "
        "change: a larger skull-and-text graphic, while placement and exact text are uncertain. "
        "Output: 'has a larger skull and text graphic'\n\n"
        "Keep A, B, and C independent. Do not compare the candidates with one another, combine their "
        "attributes, or transfer a detail from one candidate to another.\n\n"
        "Return only this JSON object, with no explanation or Markdown:\n"
        '{"A":"...","B":"...","C":"..."}'
    )


def _normalize_text(value: object) -> str:
    return " ".join(str(value).strip().strip('"').strip("'").split())


def parse_completion(raw: str, *, budget: int) -> dict[str, str]:
    cleaned = str(raw).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("completion does not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict) or set(value) != set(LABELS):
        keys = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"expected exactly A/B/C keys, got {keys}")
    parsed = {label: _normalize_text(value[label]) for label in LABELS}
    if any(not text for text in parsed.values()):
        raise ValueError("one or more counterfactuals are empty")
    return parsed


def _manifest_word_budget(manifest: Mapping[str, Any], args: Any) -> int:
    return instruction_word_budget(
        str(manifest["edit"]),
        str(manifest.get("dataset_name", getattr(args, "dataset_name", "fashioniq"))),
    )


def select_candidates(
    manifest: Mapping[str, Any],
    ranking_result: Mapping[str, Any],
    *,
    seed: int,
    fixed_positions: Sequence[int] | None = None,
    include_gt: bool = False,
) -> list[str]:
    target_id = str(manifest["target_id"])
    ranked = [
        str(candidate_id) for candidate_id in ranking_result["ranked_candidate_ids"]
    ]
    if fixed_positions is not None:
        positions = [int(value) for value in fixed_positions]
        if max(positions, default=0) > len(ranked):
            fallback_positions = {
                5: (2, 3, 5),
                10: (2, 6, 9),
                15: (2, 8, 14),
            }.get(len(ranked))
            if fallback_positions is None:
                raise ValueError(
                    "fixed counterfactual position lies outside the ranking"
                )
            positions = list(fallback_positions)
        if len(positions) != len(LABELS) or len(set(positions)) != len(positions):
            raise ValueError(
                "fixed counterfactual positions must contain three unique ranks"
            )
        if min(positions) < 1 or max(positions) > len(ranked):
            raise ValueError("fixed counterfactual position lies outside the ranking")
        selected = [ranked[position - 1] for position in positions]
        if not include_gt and target_id in selected:
            raise ValueError("fixed positions selected GT while include_gt is disabled")
        return selected
    eligible = [candidate_id for candidate_id in ranked if candidate_id != target_id]
    if len(eligible) < len(LABELS):
        raise ValueError("fewer than three non-GT ranked candidates")
    material = json.dumps(
        {
            "seed": int(seed),
            "query_id": str(manifest["query_id"]),
            "eligible": eligible,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(local_seed).sample(eligible, len(LABELS))


def request_fingerprint(
    manifest: Mapping[str, Any], candidate_ids: Sequence[str], args: Any
) -> str:
    active_prompt_version = prompt_version(
        str(manifest.get("dataset_name", getattr(args, "dataset_name", "fashioniq")))
    )
    payload = {
        "version": VERSION,
        "prompt_version": active_prompt_version,
        "query_id": str(manifest["query_id"]),
        "ref_id": str(manifest["ref_id"]),
        "edit": str(manifest["edit"]),
        "candidate_ids": [str(value) for value in candidate_ids],
        "teacher_list_size": int(getattr(args, "teacher_list_size", 10)),
        "model": str(args.qwen_model_path),
        "max_tokens": int(args.tcb_qwen_max_tokens),
        "temperature": float(args.tcb_qwen_temperature),
        "top_p": float(args.tcb_qwen_top_p),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _worker_request(
    manifest: Mapping[str, Any], candidate_ids: Sequence[str], args: Any
) -> dict[str, Any]:
    return {
        "query_id": str(manifest["query_id"]),
        "prompt": build_prompt(
            str(manifest["edit"]), str(manifest.get("dataset_name", "fashioniq"))
        ),
        "prompt_position": "after_images",
        "reference_path": str(
            resolve_manifest_image_path(args, manifest, str(manifest["ref_id"]))
        ),
        "candidate_labels": list(LABELS),
        "candidate_paths": [
            str(resolve_manifest_image_path(args, manifest, str(candidate_id)))
            for candidate_id in candidate_ids
        ],
    }


def request_counterfactual_chunk(
    manifests: Sequence[Mapping[str, Any]],
    ranking_results: Sequence[Mapping[str, Any]],
    *,
    args: Any,
    cache: dict[tuple[str, str], dict[str, Any]],
    cache_path: Path,
    client: Any,
) -> tuple[list[dict[str, Any] | None], dict[str, int]]:
    """Generate one joint A/B/C response per query, retrying malformed rows."""
    if len(manifests) != len(ranking_results):
        raise ValueError("counterfactual manifests and rankings are not aligned")
    specs = []
    ordered: list[dict[str, Any] | None] = [None] * len(manifests)
    cache_hits = 0
    for index, (manifest, ranking) in enumerate(zip(manifests, ranking_results)):
        ranked = [str(value) for value in ranking["ranked_candidate_ids"]]
        fixed_positions = getattr(args, "tcb_fixed_positions", None)
        candidate_ids = select_candidates(
            manifest,
            ranking,
            seed=int(args.seed),
            fixed_positions=fixed_positions,
            include_gt=bool(getattr(args, "tcb_include_gt", False)),
        )
        fingerprint = request_fingerprint(manifest, candidate_ids, args)
        cached = cache.get((str(manifest["query_id"]), fingerprint))
        if cached and cached.get("status") == "ok":
            cached = dict(cached)
            cached.setdefault(
                "candidate_teacher_ranks",
                [ranked.index(candidate_id) + 1 for candidate_id in candidate_ids],
            )
            cached.setdefault(
                "candidate_is_gt",
                [
                    candidate_id == str(manifest["target_id"])
                    for candidate_id in candidate_ids
                ],
            )
            ordered[index] = cached
            cache_hits += 1
        else:
            specs.append((index, manifest, candidate_ids, fingerprint))

    unresolved = list(specs)
    started = {index: time.monotonic() for index, *_ in specs}
    last_error: dict[int, str] = {}
    last_raw: dict[int, str] = {}
    generation_attempts = 0
    for attempt in range(1, int(args.max_attempts) + 1):
        if not unresolved:
            break
        requests = [
            _worker_request(manifest, candidate_ids, args)
            for _, manifest, candidate_ids, _ in unresolved
        ]
        generation_attempts += len(requests)
        try:
            raw_outputs = client.generate(
                requests,
                int(args.seed) + 10000 + attempt - 1,
                max_tokens=int(args.tcb_qwen_max_tokens),
                temperature=float(args.tcb_qwen_temperature),
                top_p=float(args.tcb_qwen_top_p),
                stage="counterfactual",
            )
        except Exception:
            if attempt == int(args.max_attempts):
                raise
            time.sleep(float(args.retry_delay) * (2 ** (attempt - 1)))
            continue
        retry = []
        for spec, raw in zip(unresolved, raw_outputs):
            index, manifest, candidate_ids, fingerprint = spec
            active_prompt_version = prompt_version(
                str(
                    manifest.get(
                        "dataset_name", getattr(args, "dataset_name", "fashioniq")
                    )
                )
            )
            last_raw[index] = str(raw)
            try:
                parsed = parse_completion(
                    str(raw), budget=_manifest_word_budget(manifest, args)
                )
            except Exception as error:
                last_error[index] = f"{type(error).__name__}: {error}"
                retry.append(spec)
                continue
            result = {
                "status": "ok",
                "version": VERSION,
                "prompt_version": active_prompt_version,
                "teacher_backend": "qwen3_vl_32b_vllm",
                "fingerprint": fingerprint,
                "query_id": str(manifest["query_id"]),
                "category": str(manifest["category"]),
                "ref_id": str(manifest["ref_id"]),
                "edit": str(manifest["edit"]),
                "target_id": str(manifest["target_id"]),
                "candidate_ids": list(candidate_ids),
                "candidate_teacher_ranks": [
                    [
                        str(value)
                        for value in ranking_results[index]["ranked_candidate_ids"]
                    ].index(candidate_id)
                    + 1
                    for candidate_id in candidate_ids
                ],
                "candidate_is_gt": [
                    candidate_id == str(manifest["target_id"])
                    for candidate_id in candidate_ids
                ],
                "label_to_candidate_id": dict(zip(LABELS, candidate_ids)),
                "counterfactual_texts": [parsed[label] for label in LABELS],
                "word_counts": [word_count(parsed[label]) for label in LABELS],
                "word_budget": _manifest_word_budget(manifest, args),
                "over_budget_labels": [
                    label
                    for label in LABELS
                    if word_count(parsed[label]) > _manifest_word_budget(manifest, args)
                ],
                "raw_response": str(raw),
                "attempts": attempt,
                "latency_seconds": time.monotonic() - started[index],
            }
            append_jsonl(result, cache_path)
            cache[(str(result["query_id"]), fingerprint)] = result
            ordered[index] = result
        unresolved = retry
        if unresolved and attempt < int(args.max_attempts):
            time.sleep(float(args.retry_delay) * (2 ** (attempt - 1)))

    for index, manifest, candidate_ids, fingerprint in unresolved:
        active_prompt_version = prompt_version(
            str(
                manifest.get("dataset_name", getattr(args, "dataset_name", "fashioniq"))
            )
        )
        result = {
            "status": "error",
            "version": VERSION,
            "prompt_version": active_prompt_version,
            "teacher_backend": "qwen3_vl_32b_vllm",
            "fingerprint": fingerprint,
            "query_id": str(manifest["query_id"]),
            "category": str(manifest["category"]),
            "ref_id": str(manifest["ref_id"]),
            "edit": str(manifest["edit"]),
            "target_id": str(manifest["target_id"]),
            "candidate_ids": list(candidate_ids),
            "raw_response": last_raw.get(index, ""),
            "attempts": int(args.max_attempts),
            "latency_seconds": time.monotonic() - started[index],
            "error": last_error.get(index, "unknown counterfactual parse failure"),
        }
        append_jsonl(result, cache_path)
        cache[(str(result["query_id"]), fingerprint)] = result

    return ordered, {
        "queries": len(manifests),
        "cache_hits": cache_hits,
        "cache_misses": len(specs),
        "generation_attempts": generation_attempts,
        "successes": sum(result is not None for result in ordered),
        "failures": sum(result is None for result in ordered),
    }


__all__ = [
    "PROMPT_VERSION",
    "VERSION",
    "build_prompt",
    "prompt_version",
    "parse_completion",
    "request_counterfactual_chunk",
    "request_fingerprint",
    "select_candidates",
    "instruction_word_budget",
    "word_budget",
    "word_count",
]
