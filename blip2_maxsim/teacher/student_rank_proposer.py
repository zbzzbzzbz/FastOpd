"""vLLM proposer for Student-order Top-10 ranking drafts.

vLLM invokes custom proposers after the target model's first generated token.
This proposer follows the accepted output prefix and fills remaining labels in
Student order (A..J), proposing at most FASTOPD_SPEC_LABEL_K new labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


class StudentRankProposer:
    def __init__(self, vllm_config):
        self.k = int(os.environ["FASTOPD_SPEC_LABEL_K"])
        if not 1 <= self.k <= 10:
            raise ValueError("FASTOPD_SPEC_LABEL_K must be 1..10")
        self.max_tokens = vllm_config.speculative_config.num_speculative_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            vllm_config.model_config.model, trust_remote_code=True
        )
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.assistant_start = np.asarray([151644, 77091, 198], dtype=np.int32)
        self.token_bytes_cache: dict[int, bytes] = {}
        # The ranking grammar is ASCII-only. These single-character tokens are
        # a safe fallback if standalone suffix tokenization does not round-trip.
        self.char_ids = {}
        for char in "[] >ABCDEFGHIJ":
            token_ids = self.tokenizer.encode(char, add_special_tokens=False)
            if len(token_ids) != 1 or self._decode(token_ids) != char:
                raise ValueError(f"No exact single-token grammar character: {char!r}")
            self.char_ids[char] = token_ids[0]
        fallback_log = os.environ.get("FASTOPD_SPEC_FALLBACK_LOG")
        self.fallback_log = Path(fallback_log) if fallback_log else None
        self.logged_fallbacks: set[tuple[str, str]] = set()
        if self.fallback_log is not None:
            with self.fallback_log.open("a") as handle:
                handle.write(
                    json.dumps({"event": "telemetry_ready", "k": self.k}) + "\n"
                )

    def _record_fallback(self, reason, ids, assistant_end, generated):
        if self.fallback_log is None:
            return
        prompt_hash = hashlib.sha256(
            np.asarray(ids[:assistant_end], dtype=np.int32).tobytes()
        ).hexdigest()
        key = (prompt_hash, reason)
        if key in self.logged_fallbacks:
            return
        self.logged_fallbacks.add(key)
        with self.fallback_log.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "prompt_sha256": prompt_hash,
                        "reason": reason,
                        "generated_tokens_at_first_fallback": len(generated),
                    }
                )
                + "\n"
            )

    def load_model(self, *args, **kwargs):
        pass

    def _decode(self, token_ids):
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def _token_bytes(self, token_id):
        token_id = int(token_id)
        if token_id not in self.token_bytes_cache:
            self.token_bytes_cache[token_id] = self._decode([token_id]).encode("utf-8")
        return self.token_bytes_cache[token_id]

    def _continuation_is_exact(self, generated, candidate, current_text, suffix):
        return (
            b"".join(self._token_bytes(token_id) for token_id in candidate)
            == suffix.encode("utf-8")
            and self._decode(generated + candidate) == current_text + suffix
        )

    def _encode_continuation(self, generated, suffix):
        """Encode only future text; never re-tokenize or alter generated IDs."""
        if not suffix:
            return []
        current_text = self._decode(generated)
        preferred = self.tokenizer.encode(suffix, add_special_tokens=False)
        if self._continuation_is_exact(generated, preferred, current_text, suffix):
            return preferred
        try:
            by_character = [self.char_ids[char] for char in suffix]
        except KeyError:
            return None
        if self._continuation_is_exact(generated, by_character, current_text, suffix):
            return by_character
        return None

    def propose(
        self,
        sampled_token_ids,
        num_tokens_no_spec,
        token_ids_cpu,
        slot_mappings=None,
    ):
        proposals = []
        for row, sampled in enumerate(sampled_token_ids):
            if not sampled:
                proposals.append([])
                continue
            ids = token_ids_cpu[row, : int(num_tokens_no_spec[row])]
            positions = np.flatnonzero(
                (ids[:-2] == self.assistant_start[0])
                & (ids[1:-1] == self.assistant_start[1])
                & (ids[2:] == self.assistant_start[2])
            )
            if not len(positions):
                proposals.append([])
                continue
            assistant_end = int(positions[-1]) + 3
            generated = ids[assistant_end:].tolist()
            if not generated or self.eos_id in generated:
                proposals.append([])
                continue
            output = self._decode(generated)
            selected = re.findall(r"\[([A-J])", output)
            if len(selected) != len(set(selected)) or len(selected) > 10:
                proposals.append([])
                continue
            remaining = [letter for letter in "ABCDEFGHIJ" if letter not in selected]
            full = " > ".join(f"[{letter}]" for letter in selected + remaining)
            if not full.startswith(output):
                self._record_fallback(
                    "text_not_canonical_prefix", ids, assistant_end, generated
                )
                proposals.append([])
                continue
            total_labels = min(len(selected) + self.k, 10)
            endpoint = " > ".join(
                f"[{letter}]" for letter in (selected + remaining)[:total_labels]
            )
            if len(output) > len(endpoint):
                self._record_fallback(
                    "past_chunk_endpoint", ids, assistant_end, generated
                )
                proposals.append([])
                continue
            suffix = endpoint[len(output) :]
            draft = self._encode_continuation(generated, suffix)
            if draft is None:
                self._record_fallback(
                    "suffix_tokenization_failed", ids, assistant_end, generated
                )
                proposals.append([])
                continue
            if total_labels == 10 and endpoint == full:
                draft.append(self.eos_id)
            proposals.append(draft[: self.max_tokens])
        return proposals
