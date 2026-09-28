"""Persistent Qwen3-VL teacher client for online OPD.

The student and vLLM use separate Python environments and exchange requests
through a Unix socket while sharing one physical GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from blip2_maxsim.data.image_resolver import resolve_manifest_image_path
from blip2_maxsim.teacher.ranking import (
    build_prompt,
    candidate_set_sha256,
    labels_for_count,
    parse_ranking,
    prompt_version,
)
from blip2_maxsim.training.student_core import append_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "stage1_online_opd_qwen32_ranker_v1"
PROMPT_VERSION = "fashioniq_reference_edit_AJ_full_permutation_v1"
LABELS = tuple("ABCDEFGHIJ")


def _send_packet(connection: socket.socket, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), ensure_ascii=False).encode("utf-8")
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("Qwen32 worker closed its socket unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_packet(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!Q", _recv_exact(connection, 8))[0]
    value = json.loads(_recv_exact(connection, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Qwen32 worker returned a non-object payload")
    return value


def request_fingerprint(manifest: Mapping[str, Any], args: Any) -> str:
    candidate_count = len(manifest.get("candidates", ()))
    active_prompt_version = prompt_version(
        str(manifest.get("dataset_name", getattr(args, "dataset_name", "fashioniq"))),
        candidate_count,
    )
    payload = {
        "version": VERSION,
        "prompt_version": active_prompt_version,
        "candidate_set_sha256": candidate_set_sha256(manifest),
        "candidate_count": candidate_count,
        "model_path": str(args.qwen_model_path),
        "max_model_len": int(args.qwen_max_model_len),
        "image_min_pixels": int(args.qwen_image_min_pixels),
        "image_max_pixels": int(args.qwen_image_max_pixels),
        "max_tokens": int(args.qwen_max_tokens),
        "temperature": float(args.qwen_temperature),
        "top_p": float(args.qwen_top_p),
        "parse_policy": str(args.parse_policy),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class Qwen32WorkerClient:
    """Own one persistent Qwen vLLM subprocess on a separate physical GPU."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.connection: socket.socket | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle = None
        self.defer_sleep = False
        # Linux AF_UNIX paths are usually limited to 108 bytes; the experiment
        # output path is longer than that, so keep only logs in the output dir.
        self.socket_path = Path("/tmp") / (
            f"blip2_qwen32_{os.getpid()}_{secrets.token_hex(4)}.sock"
        )

    def start(self) -> None:
        if self.process is not None:
            return
        self.socket_path.unlink(missing_ok=True)
        log_path = self.args.output_dir / "qwen32_worker.log"
        self.log_handle = log_path.open("ab", buffering=0)
        environment = os.environ.copy()
        if int(getattr(self.args, "ranking_spec_k", 0)):
            environment["FASTOPD_SPEC_LABEL_K"] = str(self.args.ranking_spec_k)
            environment["PYTHONPATH"] = (
                str(PROJECT_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
            )
        else:
            environment.pop("FASTOPD_SPEC_LABEL_K", None)
        # This is a physical index because the worker is a fresh exec process.
        environment["CUDA_VISIBLE_DEVICES"] = str(self.args.qwen_gpu)
        environment["PATH"] = (
            f"{Path(self.args.qwen_python).parent}:" + environment.get("PATH", "")
        )
        # Match the proven contextual_match_first worker setup.  This avoids a
        # runtime FlashInfer sampler JIT and uses vLLM's built-in sampler.
        environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        environment.setdefault("VLLM_DISABLE_FLASHINFER", "1")
        if str(getattr(self.args, "dataset_name", "fashioniq")).lower() != "fashioniq":
            raise ValueError("This FastOpd release supports FashionIQ only")
        command = [
            str(self.args.qwen_python),
            str(PROJECT_ROOT / "blip2_maxsim/teacher/teacher_worker.py"),
            "--socket-path",
            str(self.socket_path),
            "--model-path",
            str(self.args.qwen_model_path),
            "--metrics-path",
            str(self.args.output_dir / "teacher_request_metrics.jsonl"),
            "--max-model-len",
            str(self.args.qwen_max_model_len),
            "--max-num-batched-tokens",
            str(self.args.qwen_max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(self.args.qwen_gpu_memory_utilization),
            "--cpu-offload-gb",
            str(self.args.qwen_cpu_offload_gb),
            "--max-num-seqs",
            str(self.args.qwen_max_num_seqs),
            "--max-images-per-prompt",
            str(int(self.args.teacher_list_size) + 1),
            "--image-min-pixels",
            str(self.args.qwen_image_min_pixels),
            "--image-max-pixels",
            str(self.args.qwen_image_max_pixels),
        ]
        if self.args.qwen_sleep_between_requests:
            command.append("--sleep-between-requests")
        else:
            command.append("--no-sleep-between-requests")
        if self.args.qwen_trust_remote_code:
            command.append("--trust-remote-code")
        else:
            command.append("--no-trust-remote-code")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )

        deadline = time.monotonic() + float(self.args.qwen_startup_timeout)
        last_error = "worker socket has not appeared"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Qwen32 worker exited with code {self.process.returncode}; "
                    f"see {log_path}"
                )
            if self.socket_path.exists():
                connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                # The worker binds before loading the 32B model, so the first
                # ping may legitimately wait for most of the startup budget.
                connection.settimeout(max(deadline - time.monotonic(), 1.0))
                try:
                    connection.connect(str(self.socket_path))
                    _send_packet(connection, {"command": "ping"})
                    response = _recv_packet(connection)
                    if response.get("status") != "ready":
                        raise RuntimeError(f"Unexpected worker handshake: {response}")
                    connection.settimeout(None)
                    self.connection = connection
                    return
                except (ConnectionError, OSError, RuntimeError) as error:
                    last_error = str(error)
                    connection.close()
            time.sleep(1.0)
        raise TimeoutError(
            f"Qwen32 worker did not become ready within "
            f"{self.args.qwen_startup_timeout}s: {last_error}; see {log_path}"
        )

    def generate(
        self,
        requests: Sequence[Mapping[str, Any]],
        seed: int,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stage: str = "ranking",
    ) -> list[str]:
        if self.connection is None:
            raise RuntimeError("Qwen32 worker has not been started")
        if stage == "counterfactual" and bool(
            getattr(self.args, "cmi_three_branch", False)
        ):
            from blip2_maxsim.teacher.cmi_branches import branch_request

            branches = [
                branch_request(dict(request), label)
                for request in requests
                for label in "ABC"
            ]
            texts = self._generate_direct(
                branches,
                seed,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stage=stage,
                inference_batch_size=int(self.args.qwen_cmi_inference_batch_size),
            )
            return [
                json.dumps(dict(zip("ABC", texts[3 * i : 3 * i + 3])))
                for i in range(len(requests))
            ]
        return self._generate_direct(
            requests,
            seed,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stage=stage,
            inference_batch_size=int(self.args.qwen_inference_batch_size),
        )

    def _generate_direct(
        self,
        requests,
        seed,
        *,
        max_tokens,
        temperature,
        top_p,
        stage,
        inference_batch_size,
    ) -> list[str]:
        _send_packet(
            self.connection,
            {
                "command": "generate",
                "stage": stage,
                "requests": [dict(item) for item in requests],
                "inference_batch_size": inference_batch_size,
                "sampling": {
                    "max_tokens": int(
                        self.args.qwen_max_tokens if max_tokens is None else max_tokens
                    ),
                    "temperature": float(
                        self.args.qwen_temperature
                        if temperature is None
                        else temperature
                    ),
                    "top_p": float(self.args.qwen_top_p if top_p is None else top_p),
                    "seed": int(seed),
                },
                "sleep_after": not bool(self.defer_sleep),
            },
        )
        response = _recv_packet(self.connection)
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("error", response)))
        outputs = response.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(requests):
            raise RuntimeError("Qwen32 worker returned the wrong number of outputs")
        return [str(value) for value in outputs]

    def sleep(self) -> None:
        """Release worker model weights after a multi-stage logical request."""
        if self.connection is None:
            return
        _send_packet(self.connection, {"command": "sleep"})
        response = _recv_packet(self.connection)
        if response.get("status") != "sleeping":
            raise RuntimeError(f"Unexpected worker sleep response: {response}")

    def close(self) -> None:
        if self.connection is not None:
            try:
                _send_packet(self.connection, {"command": "close"})
                _recv_packet(self.connection)
            except (ConnectionError, OSError, RuntimeError):
                pass
            self.connection.close()
            self.connection = None
        if self.process is not None:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        self.socket_path.unlink(missing_ok=True)


def _worker_request(manifest: Mapping[str, Any], args: Any) -> dict[str, Any]:
    candidates = sorted(
        manifest["candidates"], key=lambda item: int(item["student_rank"])
    )
    labels = labels_for_count(len(candidates))
    label_to_id = {
        label: str(candidate["candidate_id"])
        for label, candidate in zip(labels, candidates)
    }
    return {
        "query_id": str(manifest["query_id"]),
        "prompt": build_prompt(
            str(manifest["edit"]),
            str(manifest.get("dataset_name", "fashioniq")),
            len(candidates),
        ),
        "candidate_labels": list(labels),
        "reference_path": str(
            resolve_manifest_image_path(args, manifest, str(manifest["ref_id"]))
        ),
        "candidate_paths": [
            str(resolve_manifest_image_path(args, manifest, label_to_id[label]))
            for label in labels
        ],
    }


def _success_result(
    manifest: Mapping[str, Any],
    args: Any,
    raw_text: str,
    attempt: int,
    elapsed: float,
) -> dict[str, Any]:
    candidates = sorted(
        manifest["candidates"], key=lambda item: int(item["student_rank"])
    )
    labels = labels_for_count(len(candidates))
    label_to_id = {
        label: str(candidate["candidate_id"])
        for label, candidate in zip(labels, candidates)
    }
    ranking, complete, appended = parse_ranking(
        raw_text, str(args.parse_policy), len(candidates)
    )
    ranked_ids = [label_to_id[label] for label in ranking]
    target_id = str(manifest["target_id"])
    active_prompt_version = prompt_version(
        str(manifest.get("dataset_name", getattr(args, "dataset_name", "fashioniq"))),
        len(candidates),
    )
    return {
        "status": "ok",
        "version": VERSION,
        "prompt_version": active_prompt_version,
        "teacher_backend": "qwen3_vl_32b_vllm",
        "fingerprint": request_fingerprint(manifest, args),
        "candidate_set_sha256": candidate_set_sha256(manifest),
        "query_id": str(manifest["query_id"]),
        "category": str(manifest["category"]),
        "ref_id": str(manifest["ref_id"]),
        "edit": str(manifest["edit"]),
        "target_id": target_id,
        "model": str(args.qwen_model_path),
        "label_to_candidate_id": label_to_id,
        "original_candidate_ids": [label_to_id[label] for label in labels],
        "ranking_labels": ranking,
        "ranked_candidate_ids": ranked_ids,
        "gt_rank": ranked_ids.index(target_id) + 1 if target_id in ranked_ids else None,
        "raw_response": raw_text,
        "parse_complete": complete,
        "fallback_appended_labels": appended,
        "attempts": int(attempt),
        "latency_seconds": float(elapsed),
    }


def request_qwen_chunk(
    manifests: Sequence[Mapping[str, Any]],
    *,
    args: Any,
    cache: dict[tuple[str, str], dict[str, Any]],
    cache_path: Path,
    client: Qwen32WorkerClient,
) -> tuple[list[dict[str, Any]], int, int]:
    """Rank one OPD micro-batch with Qwen while preserving the GPT cache schema."""
    ordered: list[dict[str, Any] | None] = [None] * len(manifests)
    pending: list[tuple[int, Mapping[str, Any]]] = []
    cached_count = 0
    for index, manifest in enumerate(manifests):
        fingerprint = request_fingerprint(manifest, args)
        cached = cache.get((str(manifest["query_id"]), fingerprint))
        if (
            cached
            and cached.get("status") == "ok"
            and cached.get("candidate_set_sha256") == candidate_set_sha256(manifest)
        ):
            ordered[index] = cached
            cached_count += 1
        else:
            pending.append((index, manifest))

    unresolved = list(pending)
    started = {index: time.monotonic() for index, _ in pending}
    last_errors: dict[int, str] = {}
    for attempt in range(1, int(args.max_attempts) + 1):
        if not unresolved:
            break
        worker_requests = [
            _worker_request(manifest, args) for _, manifest in unresolved
        ]
        try:
            raw_outputs = client.generate(worker_requests, int(args.seed) + attempt - 1)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            for index, _ in unresolved:
                last_errors[index] = message
            if attempt < int(args.max_attempts):
                time.sleep(float(args.retry_delay) * (2 ** (attempt - 1)))
            continue
        retry = []
        for (index, manifest), raw_text in zip(unresolved, raw_outputs):
            try:
                result = _success_result(
                    manifest,
                    args,
                    raw_text,
                    attempt,
                    time.monotonic() - started[index],
                )
                append_jsonl(result, cache_path)
                cache[(str(result["query_id"]), str(result["fingerprint"]))] = result
                ordered[index] = result
            except Exception as error:
                last_errors[index] = f"{type(error).__name__}: {error}"
                retry.append((index, manifest))
        unresolved = retry
        if unresolved and attempt < int(args.max_attempts):
            time.sleep(float(args.retry_delay) * (2 ** (attempt - 1)))

    if unresolved:
        failures = []
        for index, manifest in unresolved:
            active_prompt_version = prompt_version(
                str(
                    manifest.get(
                        "dataset_name", getattr(args, "dataset_name", "fashioniq")
                    )
                ),
                len(manifest.get("candidates", ())),
            )
            result = {
                "status": "error",
                "version": VERSION,
                "prompt_version": active_prompt_version,
                "teacher_backend": "qwen3_vl_32b_vllm",
                "fingerprint": request_fingerprint(manifest, args),
                "candidate_set_sha256": candidate_set_sha256(manifest),
                "query_id": str(manifest["query_id"]),
                "model": str(args.qwen_model_path),
                "attempts": int(args.max_attempts),
                "latency_seconds": time.monotonic() - started[index],
                "error": last_errors.get(index, "unknown Qwen32 failure"),
            }
            append_jsonl(result, cache_path)
            cache[(str(result["query_id"]), str(result["fingerprint"]))] = result
            failures.append(f"{result['query_id']}: {result['error']}")
        raise RuntimeError("Qwen32 failed: " + "; ".join(failures))

    if any(result is None for result in ordered):
        raise RuntimeError("Internal error: missing online Qwen32 result")
    return (
        [result for result in ordered if result is not None],
        cached_count,
        len(pending),
    )
