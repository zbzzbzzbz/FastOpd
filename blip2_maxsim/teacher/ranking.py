"""Standalone GPT listwise image ranker for online Stage-1 OPD."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "stage1_online_opd_gpt_ranker_v1"
PROMPT_VERSION = "fashioniq_reference_edit_AJ_full_permutation_v1"
LABEL_ALPHABET = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LABELS = LABEL_ALPHABET[:10]
_THREAD_LOCAL = threading.local()


def labels_for_count(count: int) -> tuple[str, ...]:
    count = int(count)
    if count < 1 or count > len(LABEL_ALPHABET):
        raise ValueError(
            f"candidate count must be in [1, {len(LABEL_ALPHABET)}], got {count}"
        )
    return LABEL_ALPHABET[:count]


def is_content_inspection_failure(error: Any) -> bool:
    """Return whether the API rejected this query's image content.

    This is deliberately restricted to the provider's explicit inspection
    error code so transport, quota, authentication, and availability failures
    remain fatal at the training-run level.
    """
    return "data_inspection_failed" in str(error).lower()


def prompt_version(dataset_name: str, candidate_count: int = 10) -> str:
    base = PROMPT_VERSION
    return (
        base
        if int(candidate_count) == 10
        else base.replace("AJ", f"A{labels_for_count(candidate_count)[-1]}")
    )


def build_prompt(
    edit: str, dataset_name: str = "fashioniq", candidate_count: int = 10
) -> str:
    """Use the exact direct-listwise prompt from the offline GPT experiment."""
    labels = labels_for_count(candidate_count)
    first_label, last_label = labels[0], labels[-1]
    label_text = ", ".join(f"[{label}]" for label in labels)
    permutation = " > ".join(f"[{label}]" for label in labels)
    return (
        f"You are an expert in composed image retrieval. The task is to rank {candidate_count} candidate target "
        "images for a reference image and a requested edit.\n\n"
        f"Requested edit: {edit}\n\n"
        f"The reference image is shown first and is not a candidate. Candidate images [{first_label}] through "
        f"[{last_label}] follow. The candidate identifiers are {label_text}. Rank every candidate from most to least relevant. A strong candidate satisfies "
        "the requested visual changes while preserving important reference-image attributes that "
        "the edit does not ask to change. Ignore candidate position, identifier letter, person, "
        "pose, crop, background, lighting, and image quality. Compare the candidates jointly.\n\n"
        "Output only one complete permutation in exactly this form: "
        f"{permutation}. "
        "Use each identifier exactly once. Do not output scores or an explanation."
    )


def parse_ranking(
    text: str, policy: str = "strict", candidate_count: int = 10
) -> tuple[list[str], bool, list[str]]:
    expected = labels_for_count(candidate_count)
    labels = re.findall(r"\[\s*([A-Za-z])\s*\]", text)
    if not labels:
        labels = re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", text)
    unique: list[str] = []
    for label in (item.upper() for item in labels):
        if label not in unique:
            unique.append(label)
    unique = [label for label in unique if label in expected]
    missing = [label for label in expected if label not in unique]
    complete = len(unique) == len(expected) and not missing
    if complete:
        return unique, True, []
    if policy == "paper_fallback" and unique:
        return unique + missing, False, missing
    raise ValueError(
        f"Incomplete permutation: parsed={unique}, missing={missing}, raw={text!r}"
    )


def image_data_url(path: Path, max_size: int, quality: int) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_content(
    manifest: Mapping[str, Any],
    *,
    image_path_for_id,
    max_image_size: int,
    jpeg_quality: int,
    image_detail: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    candidates = sorted(
        manifest["candidates"], key=lambda item: int(item["student_rank"])
    )
    candidate_count = len(candidates)
    labels = labels_for_count(candidate_count)
    candidate_ids = [str(item["candidate_id"]) for item in candidates]
    if len(set(candidate_ids)) != candidate_count:
        raise ValueError(f"Duplicate candidates for {manifest['query_id']}")
    label_to_id = dict(zip(labels, candidate_ids))
    content: list[dict[str, str]] = [
        {
            "type": "input_text",
            "text": build_prompt(
                str(manifest["edit"]),
                str(manifest.get("dataset_name", "fashioniq")),
                candidate_count,
            ),
        },
        {"type": "input_text", "text": "Reference image:"},
        {
            "type": "input_image",
            "image_url": image_data_url(
                image_path_for_id(str(manifest["ref_id"])),
                max_image_size,
                jpeg_quality,
            ),
            "detail": image_detail,
        },
    ]
    for label in labels:
        content.extend(
            [
                {"type": "input_text", "text": f"Candidate [{label}]:"},
                {
                    "type": "input_image",
                    "image_url": image_data_url(
                        image_path_for_id(label_to_id[label]),
                        max_image_size,
                        jpeg_quality,
                    ),
                    "detail": image_detail,
                },
            ]
        )
    return content, label_to_id


def _get_http_session(trust_env: bool):
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        import requests

        session = requests.Session()
        session.trust_env = bool(trust_env)
        _THREAD_LOCAL.session = session
    return session


def _extract_output_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    pieces: list[str] = []
    for output in payload.get("output", []):
        if not isinstance(output, Mapping):
            continue
        for content in output.get("content", []):
            if not isinstance(content, Mapping):
                continue
            if content.get("type") in ("output_text", "text") and isinstance(
                content.get("text"), str
            ):
                pieces.append(str(content["text"]))
    text = "\n".join(pieces).strip()
    if not text:
        raise ValueError(
            f"Responses payload contains no output text: {sorted(payload)}"
        )
    return text


def _decode_response_payload(response: Any) -> Mapping[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, Mapping):
        return payload
    completed: Mapping[str, Any] | None = None
    deltas: list[str] = []
    stream_error: Any = None
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line.removeprefix("data:").strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("type")
        if event_type == "response.output_text.delta" and isinstance(
            event.get("delta"), str
        ):
            deltas.append(str(event["delta"]))
        elif event_type == "response.completed" and isinstance(
            event.get("response"), Mapping
        ):
            completed = event["response"]
        elif event_type in ("error", "response.failed"):
            stream_error = event.get("error", event)
    if completed is not None:
        if deltas and not completed.get("output"):
            completed = dict(completed)
            completed["output_text"] = "".join(deltas)
        return completed
    if stream_error is not None:
        raise RuntimeError(f"Responses stream failed: {str(stream_error)[:1000]}")
    if deltas:
        return {"output_text": "".join(deltas)}
    raise RuntimeError(
        f"HTTP {response.status_code} returned neither JSON nor complete SSE: "
        f"{response.text[:500]!r}"
    )


def create_response(
    *,
    base_url: str,
    api_key_env: str,
    model: str,
    content: Sequence[Mapping[str, str]],
    timeout: float,
    trust_env: bool,
    api_format: str = "responses",
    enable_thinking: bool = False,
    max_tokens: int = 0,
    temperature: float | None = None,
    top_p: float | None = None,
) -> tuple[str, str | None, dict[str, Any] | None]:
    api_format = str(api_format)
    if api_format not in {"responses", "chat_completions"}:
        raise ValueError(f"Unsupported API format: {api_format}")
    if api_format == "chat_completions":
        chat_content: list[dict[str, Any]] = []
        for item in content:
            if item.get("type") == "input_text":
                chat_content.append({"type": "text", "text": str(item["text"])})
            elif item.get("type") == "input_image":
                image_url: dict[str, Any] = {"url": str(item["image_url"])}
                if item.get("detail") is not None:
                    image_url["detail"] = str(item["detail"])
                chat_content.append({"type": "image_url", "image_url": image_url})
            else:
                raise ValueError(f"Unsupported multimodal content item: {item}")
        request_body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": chat_content}],
            "enable_thinking": bool(enable_thinking),
        }
        if int(max_tokens) > 0:
            request_body["max_tokens"] = int(max_tokens)
        if temperature is not None:
            request_body["temperature"] = float(temperature)
        if top_p is not None:
            request_body["top_p"] = float(top_p)
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
    else:
        request_body = {
            "model": model,
            "input": [{"role": "user", "content": list(content)}],
            "store": False,
        }
        if int(max_tokens) > 0:
            request_body["max_output_tokens"] = int(max_tokens)
        if temperature is not None:
            request_body["temperature"] = float(temperature)
        if top_p is not None:
            request_body["top_p"] = float(top_p)
        endpoint = f"{base_url.rstrip('/')}/responses"
    response = _get_http_session(trust_env).post(
        endpoint,
        headers={
            "Authorization": f"Bearer {os.environ[api_key_env]}",
            "Content-Type": "application/json",
        },
        json=request_body,
        timeout=float(timeout),
    )
    payload = _decode_response_payload(response)
    if response.status_code >= 400:
        raise RuntimeError(
            f"HTTP {response.status_code}: {str(payload.get('error', payload))[:1000]}"
        )
    if api_format == "chat_completions":
        choices = payload.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        output = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(output, str) or not output.strip():
            raise ValueError("Chat Completions payload contains no message text")
        output_text = output.strip()
    else:
        output_text = _extract_output_text(payload)
    usage = payload.get("usage")
    return (
        output_text,
        str(payload["id"]) if payload.get("id") is not None else None,
        dict(usage) if isinstance(usage, Mapping) else None,
    )


def candidate_set_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {
        "query_id": str(manifest["query_id"]),
        "ref_id": str(manifest["ref_id"]),
        "edit": str(manifest["edit"]),
        "candidate_ids": [
            str(item["candidate_id"])
            for item in sorted(
                manifest["candidates"], key=lambda item: int(item["student_rank"])
            )
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def request_fingerprint(manifest: Mapping[str, Any], args: Any) -> str:
    candidate_count = len(manifest.get("candidates", ()))
    endpoint_hash = hashlib.sha256(str(args.base_url).encode()).hexdigest()
    active_prompt_version = prompt_version(
        str(manifest.get("dataset_name", getattr(args, "dataset_name", "fashioniq"))),
        candidate_count,
    )
    payload = {
        "version": VERSION,
        "prompt_version": active_prompt_version,
        "candidate_set_sha256": candidate_set_sha256(manifest),
        "candidate_count": candidate_count,
        "model": str(args.model),
        "image_detail": str(args.image_detail),
        "max_image_size": int(args.max_image_size),
        "jpeg_quality": int(args.jpeg_quality),
        "parse_policy": str(args.parse_policy),
        "trust_env": bool(args.trust_env),
        "endpoint_sha256": endpoint_hash,
    }
    api_format = str(getattr(args, "api_format", "responses"))
    if api_format != "responses":
        payload.update(
            {
                "api_format": api_format,
                "enable_thinking": bool(getattr(args, "enable_thinking", False)),
                "api_max_tokens": int(getattr(args, "api_max_tokens", 0)),
                "api_temperature": getattr(args, "api_temperature", None),
                "api_top_p": getattr(args, "api_top_p", None),
            }
        )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def rank_one(
    manifest: Mapping[str, Any],
    args: Any,
    *,
    image_path_for_id,
) -> dict[str, Any]:
    query_id = str(manifest["query_id"])
    candidate_count = len(manifest.get("candidates", ()))
    active_prompt_version = prompt_version(
        str(manifest.get("dataset_name", getattr(args, "dataset_name", "fashioniq"))),
        candidate_count,
    )
    fingerprint = request_fingerprint(manifest, args)
    started = time.monotonic()
    last_error = ""
    for attempt in range(1, int(args.max_attempts) + 1):
        try:
            content, label_to_id = build_content(
                manifest,
                image_path_for_id=image_path_for_id,
                max_image_size=int(args.max_image_size),
                jpeg_quality=int(args.jpeg_quality),
                image_detail=str(args.image_detail),
            )
            raw_text, response_id, usage = create_response(
                base_url=str(args.base_url),
                api_key_env=str(args.api_key_env),
                model=str(args.model),
                content=content,
                timeout=float(args.timeout),
                trust_env=bool(args.trust_env),
                api_format=str(getattr(args, "api_format", "responses")),
                enable_thinking=bool(getattr(args, "enable_thinking", False)),
                max_tokens=int(getattr(args, "api_max_tokens", 0)),
                temperature=getattr(args, "api_temperature", None),
                top_p=getattr(args, "api_top_p", None),
            )
            ranking, complete, appended = parse_ranking(
                raw_text, str(args.parse_policy), candidate_count
            )
            ranked_ids = [label_to_id[label] for label in ranking]
            target_id = str(manifest["target_id"])
            return {
                "status": "ok",
                "version": VERSION,
                "teacher_backend": (
                    "qwen_chat_completions_api"
                    if str(getattr(args, "api_format", "responses"))
                    == "chat_completions"
                    else "gpt_responses_api"
                ),
                "prompt_version": active_prompt_version,
                "fingerprint": fingerprint,
                "candidate_set_sha256": candidate_set_sha256(manifest),
                "query_id": query_id,
                "category": str(manifest["category"]),
                "ref_id": str(manifest["ref_id"]),
                "edit": str(manifest["edit"]),
                "target_id": target_id,
                "model": str(args.model),
                "api_format": str(getattr(args, "api_format", "responses")),
                "enable_thinking": bool(getattr(args, "enable_thinking", False)),
                "label_to_candidate_id": label_to_id,
                "original_candidate_ids": [
                    str(item["candidate_id"])
                    for item in sorted(
                        manifest["candidates"],
                        key=lambda item: int(item["student_rank"]),
                    )
                ],
                "ranking_labels": ranking,
                "ranked_candidate_ids": ranked_ids,
                "gt_rank": (
                    ranked_ids.index(target_id) + 1 if target_id in ranked_ids else None
                ),
                "raw_response": raw_text,
                "parse_complete": complete,
                "fallback_appended_labels": appended,
                "response_id": response_id,
                "usage": usage,
                "attempts": attempt,
                "latency_seconds": time.monotonic() - started,
            }
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if is_content_inspection_failure(last_error):
                break
            if attempt < int(args.max_attempts):
                time.sleep(float(args.retry_delay) * (2 ** (attempt - 1)))
    return {
        "status": "error",
        "version": VERSION,
        "teacher_backend": (
            "qwen_chat_completions_api"
            if str(getattr(args, "api_format", "responses")) == "chat_completions"
            else "gpt_responses_api"
        ),
        "prompt_version": active_prompt_version,
        "fingerprint": fingerprint,
        "candidate_set_sha256": candidate_set_sha256(manifest),
        "query_id": query_id,
        "model": str(args.model),
        "attempts": attempt,
        "latency_seconds": time.monotonic() - started,
        "error": last_error,
    }
