#!/usr/bin/env python3
"""Persistent vLLM worker for Qwen3-VL-32B online list-wise ranking."""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


LABELS = tuple("ABCDEFGHIJ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--max-images-per-prompt",
        type=int,
        default=11,
        help="Maximum reference-plus-candidate images accepted by vLLM per request.",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=128 * 28 * 28,
        help="Lower bound for Qwen-VL's per-image resize budget.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=512 * 28 * 28,
        help="Upper bound for Qwen-VL's per-image resize budget.",
    )
    parser.add_argument(
        "--sleep-between-requests",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def send_packet(connection: socket.socket, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), ensure_ascii=False).encode("utf-8")
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("controller disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_packet(connection: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!Q", recv_exact(connection, 8))[0]
    value = json.loads(recv_exact(connection, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("controller sent a non-object payload")
    return value


class Qwen32Teacher:
    def __init__(self, args: argparse.Namespace) -> None:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.process_vision_info = process_vision_info
        self.metrics_path = args.metrics_path
        self.image_min_pixels = int(args.image_min_pixels)
        self.image_max_pixels = int(args.image_max_pixels)
        if self.image_min_pixels <= 0 or self.image_max_pixels < self.image_min_pixels:
            raise ValueError("invalid Qwen image pixel budget")
        self.SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(
            str(args.model_path), trust_remote_code=bool(args.trust_remote_code)
        )
        spec_k = os.environ.get("FASTOPD_SPEC_LABEL_K")
        spec_kwargs = (
            {
                "spec_method": "custom_class",
                "spec_model": "blip2_maxsim.teacher.student_rank_proposer.StudentRankProposer",
                "spec_tokens": 40,
            }
            if spec_k is not None
            else {}
        )
        self.llm = LLM(
            model=str(args.model_path),
            trust_remote_code=bool(args.trust_remote_code),
            max_model_len=int(args.max_model_len),
            max_num_batched_tokens=int(args.max_num_batched_tokens),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            cpu_offload_gb=float(args.cpu_offload_gb),
            max_num_seqs=int(args.max_num_seqs),
            disable_log_stats=False,
            enable_sleep_mode=bool(args.sleep_between_requests),
            skip_mm_profiling=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            limit_mm_per_prompt={"image": int(args.max_images_per_prompt)},
            **spec_kwargs,
        )
        self.record({"event": "engine_initialized_awake"})
        self.sleep_between_requests = bool(args.sleep_between_requests)
        self.awake = True
        if self.sleep_between_requests:
            self.llm.sleep(level=1)
            self.awake = False
        self.record({"event": "worker_ready", "awake": self.awake})

    def record(self, row: Mapping[str, Any]) -> None:
        if self.metrics_path is None:
            return
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"time": time.time(), **row}, ensure_ascii=False) + "\n"
            )

    def wake_up(self) -> None:
        if self.sleep_between_requests and not self.awake:
            started = time.monotonic()
            self.record({"event": "wake_begin"})
            self.llm.wake_up()
            self.awake = True
            self.record(
                {"event": "wake_end", "duration_seconds": time.monotonic() - started}
            )

    def sleep(self) -> None:
        if self.sleep_between_requests and self.awake:
            started = time.monotonic()
            self.record({"event": "sleep_begin"})
            self.llm.sleep(level=1)
            self.awake = False
            self.record(
                {"event": "sleep_end", "duration_seconds": time.monotonic() - started}
            )

    def make_request(self, item: Mapping[str, Any]) -> dict[str, Any]:
        candidate_paths = [str(value) for value in item["candidate_paths"]]
        candidate_labels = tuple(
            str(value) for value in item.get("candidate_labels", LABELS)
        )
        if not candidate_labels or len(candidate_paths) != len(candidate_labels):
            raise ValueError("candidate paths and labels must be non-empty and aligned")
        prompt_after_images = str(item.get("prompt_position", "before_images")) == (
            "after_images"
        )
        content: list[dict[str, Any]] = []
        if not prompt_after_images:
            content.append({"type": "text", "text": str(item["prompt"])})
        content.extend(
            [
                {"type": "text", "text": "Reference image:"},
                {
                    "type": "image",
                    "image": str(item["reference_path"]),
                    "min_pixels": self.image_min_pixels,
                    "max_pixels": self.image_max_pixels,
                },
            ]
        )
        for label, path in zip(candidate_labels, candidate_paths):
            content.extend(
                [
                    {"type": "text", "text": f"Candidate [{label}]:"},
                    {
                        "type": "image",
                        "image": path,
                        "min_pixels": self.image_min_pixels,
                        "max_pixels": self.image_max_pixels,
                    },
                ]
            )
        if prompt_after_images:
            content.append({"type": "text", "text": str(item["prompt"])})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        multimodal = {}
        if image_inputs is not None:
            multimodal["image"] = image_inputs
        if video_inputs is not None:
            multimodal["video"] = video_inputs
        return {"prompt": prompt, "multi_modal_data": multimodal}

    def generate(self, payload: Mapping[str, Any]) -> list[str]:
        sampling = payload["sampling"]
        params = self.SamplingParams(
            max_tokens=int(sampling["max_tokens"]),
            temperature=max(float(sampling["temperature"]), 0.0),
            top_p=float(sampling["top_p"]),
            seed=int(sampling["seed"]),
        )
        requests = [self.make_request(item) for item in payload["requests"]]
        image_sizes = [
            [list(image.size) for image in request["multi_modal_data"].get("image", [])]
            for request in requests
        ]
        batch_size = max(int(payload.get("inference_batch_size", 1)), 1)
        texts: list[str] = []
        for start in range(0, len(requests), batch_size):
            started = time.monotonic()
            self.record(
                {
                    "event": "generate_begin",
                    "stage": payload.get("stage"),
                    "count": len(requests[start : start + batch_size]),
                }
            )
            outputs = self.llm.generate(
                requests[start : start + batch_size], params, use_tqdm=False
            )
            elapsed = time.monotonic() - started
            for offset, output in enumerate(outputs):
                item = payload["requests"][start + offset]
                completion = output.outputs[0]
                metrics = output.metrics
                token_count = len(completion.token_ids)
                first_token = getattr(metrics, "first_token_latency", None)
                queued_ts = getattr(metrics, "queued_ts", 0.0)
                scheduled_ts = getattr(metrics, "scheduled_ts", 0.0)
                first_token_ts = getattr(metrics, "first_token_ts", 0.0)
                scheduled_to_first = (
                    first_token_ts - scheduled_ts
                    if scheduled_ts and first_token_ts
                    else None
                )
                scheduler_queue = (
                    scheduled_ts - queued_ts if queued_ts and scheduled_ts else None
                )
                decode = (
                    getattr(metrics, "last_token_ts", 0.0)
                    - getattr(metrics, "first_token_ts", 0.0)
                    if metrics is not None
                    else None
                )
                self.record(
                    {
                        "event": "request_done",
                        "stage": payload.get("stage"),
                        "query_id": item.get("query_id"),
                        "prompt_tokens": len(output.prompt_token_ids or []),
                        "output_tokens": token_count,
                        "first_token_seconds": first_token,
                        "queued_ts": queued_ts,
                        "scheduled_ts": scheduled_ts,
                        "first_token_ts": first_token_ts,
                        "last_token_ts": getattr(metrics, "last_token_ts", 0.0),
                        "scheduled_to_first_seconds": scheduled_to_first,
                        "scheduler_queue_seconds": scheduler_queue,
                        "decode_seconds": decode,
                        "tpot_seconds": (
                            decode / (token_count - 1)
                            if decode is not None and token_count > 1
                            else None
                        ),
                        "generate_call_seconds": elapsed,
                        "num_cached_tokens": output.num_cached_tokens,
                        "image_sizes": image_sizes[start + offset],
                        "finish_reason": completion.finish_reason,
                    }
                )
                texts.append(str(completion.text).strip())
            self.record(
                {
                    "event": "generate_end",
                    "stage": payload.get("stage"),
                    "duration_seconds": elapsed,
                }
            )
        return texts


def main() -> None:
    args = parse_args()
    args.socket_path = args.socket_path.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    if args.metrics_path is not None:
        args.metrics_path = args.metrics_path.expanduser().resolve()
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    args.socket_path.parent.mkdir(parents=True, exist_ok=True)
    args.socket_path.unlink(missing_ok=True)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket_path))
    server.listen(1)
    try:
        teacher = Qwen32Teacher(args)
        connection, _ = server.accept()
        with connection:
            while True:
                payload = recv_packet(connection)
                command = payload.get("command")
                if command == "ping":
                    send_packet(connection, {"status": "ready"})
                elif command == "generate":
                    try:
                        teacher.wake_up()
                        try:
                            outputs = teacher.generate(payload)
                        finally:
                            if bool(payload.get("sleep_after", True)):
                                teacher.sleep()
                        send_packet(connection, {"status": "ok", "outputs": outputs})
                    except Exception as error:
                        send_packet(
                            connection,
                            {
                                "status": "error",
                                "error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc(),
                            },
                        )
                elif command == "sleep":
                    teacher.sleep()
                    send_packet(connection, {"status": "sleeping"})
                elif command == "close":
                    send_packet(connection, {"status": "closed"})
                    break
                else:
                    send_packet(
                        connection,
                        {"status": "error", "error": f"unknown command: {command}"},
                    )
    finally:
        server.close()
        args.socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
