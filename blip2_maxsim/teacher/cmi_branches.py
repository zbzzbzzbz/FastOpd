"""Split a joint CMI prompt into parallel A/B/C generation requests."""

from __future__ import annotations

from typing import Any


def branch_request(request: dict[str, Any], label: str) -> dict[str, Any]:
    if label not in "ABC":
        raise ValueError(f"Unexpected CMI candidate label: {label}")
    prompt = request["prompt"]
    marker = "Return only this JSON object, with no explanation or Markdown:"
    if marker not in prompt:
        raise ValueError(
            "CMI prompt format changed; cannot form a shared branch prefix"
        )
    common = prompt.split(marker, 1)[0].rstrip()
    suffix = (
        "\n\nFor this request, output only one modification instruction for the "
        "specified candidate. Do not output JSON, a label, quotation marks, "
        "or an explanation.\nCandidate: "
    )
    return {
        **request,
        "query_id": f"{request['query_id']}::{label}",
        "prompt": common + suffix + label,
    }
