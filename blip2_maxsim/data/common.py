"""
文件名称: common.py
功能描述: 提供 FashionIQ 数据解析、路径解析与通用评测汇总工具
创建日期: 2026-05-22
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

FASHIONIQ_CATEGORIES = ("dress", "shirt", "toptee")


@dataclass(frozen=True)
class FashionIQLayout:
    dataset_root: Path
    captions_dir: Path
    image_splits_dir: Path
    images_dir: Path


def sanitize_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return " ".join(
        x.replace("\r", " ").replace("\n", " ").replace("\t", " ").split()
    ).strip()


def parse_ks(raw: str) -> List[int]:
    return [int(k) for k in str(raw).split(",") if str(k).strip()]


def parse_categories(raw: str | Sequence[str]) -> List[str]:
    if isinstance(raw, str):
        values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    else:
        values = [sanitize_text(part).lower() for part in raw if sanitize_text(part)]
    if not values:
        raise ValueError("At least one FashionIQ category is required")
    invalid = [value for value in values if value not in FASHIONIQ_CATEGORIES]
    if invalid:
        raise ValueError(f"Unsupported FashionIQ categories: {invalid}")
    seen: Set[str] = set()
    output: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def compose_fashioniq_edit(captions: Sequence[Any]) -> str:
    parts = [sanitize_text(item) for item in captions if sanitize_text(item)]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    # 使用自然语言连接两个修改短语，兼顾 CLIP 文本编码的可读性与稳定性。
    return " and ".join(parts)


def _resolve_existing_dir(dataset_root: Path, candidates: Sequence[str]) -> Path:
    for rel in candidates:
        path = dataset_root / rel
        if path.exists() and path.is_dir():
            return path
    raise FileNotFoundError(
        f"Cannot resolve required directory under {dataset_root}. Tried: {list(candidates)}"
    )


def _dir_contains_images(path: Path) -> bool:
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        if any(path.glob(pattern)):
            return True
    return False


def _resolve_image_dir(dataset_root: Path, candidates: Sequence[str]) -> Path:
    fallback: Path | None = None
    for rel in candidates:
        path = dataset_root / rel
        if not path.exists() or not path.is_dir():
            continue
        if fallback is None:
            fallback = path
        # 图片根目录可能有一层额外包装目录，这里优先选择真正包含图片文件的目录，
        # 避免误返回外层壳目录（例如 resized_images/ 而不是 resized_images/resized_images/）。
        if _dir_contains_images(path):
            return path
    if fallback is not None:
        return fallback
    raise FileNotFoundError(
        f"Cannot resolve image directory under {dataset_root}. Tried: {list(candidates)}"
    )


def resolve_fashioniq_layout(dataset_root: str | Path) -> FashionIQLayout:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"FashionIQ dataset root not found: {root}")
    return FashionIQLayout(
        dataset_root=root,
        captions_dir=_resolve_existing_dir(
            root,
            [
                "captions",
                "fashion-iq-master/fashion-iq-master/captions",
            ],
        ),
        image_splits_dir=_resolve_existing_dir(
            root,
            [
                "image_splits",
                "fashion-iq-master/fashion-iq-master/image_splits",
            ],
        ),
        images_dir=_resolve_image_dir(
            root,
            [
                "images",
                "resized_images/resized_images",
                "resized_images",
            ],
        ),
    )


def load_fashioniq_rows(
    dataset_root: str | Path,
    *,
    split: str,
    categories: Sequence[str],
    max_rows: int = 0,
) -> List[Dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    layout = resolve_fashioniq_layout(dataset_root)
    rows: List[Dict[str, Any]] = []
    query_id = 0
    for dress_type in parse_categories(list(categories)):
        annotation_path = layout.captions_dir / f"cap.{dress_type}.{split}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"FashionIQ caption annotation not found: {annotation_path}"
            )
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        for ann in annotations:
            captions = [sanitize_text(item) for item in ann.get("captions", [])]
            row = {
                "query_id": int(query_id),
                "dress_type": dress_type,
                "candidate_id": sanitize_text(ann.get("candidate", "")),
                "target_id": sanitize_text(ann.get("target", "")),
                "captions": captions,
                "caption_1": captions[0] if len(captions) > 0 else "",
                "caption_2": captions[1] if len(captions) > 1 else "",
                "edit": compose_fashioniq_edit(captions),
            }
            if (
                row["candidate_id"]
                and row["edit"]
                and (split == "test" or row["target_id"])
            ):
                rows.append(row)
                query_id += 1
            if max_rows > 0 and len(rows) >= int(max_rows):
                return rows
    return rows


def load_fashioniq_candidate_ids(
    dataset_root: str | Path,
    *,
    split: str,
    categories: Sequence[str],
) -> Dict[str, List[str]]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    layout = resolve_fashioniq_layout(dataset_root)
    output: Dict[str, List[str]] = {}
    for dress_type in parse_categories(list(categories)):
        split_path = layout.image_splits_dir / f"split.{dress_type}.{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"FashionIQ image split not found: {split_path}")
        values = json.loads(split_path.read_text(encoding="utf-8"))
        output[dress_type] = [
            sanitize_text(item) for item in values if sanitize_text(item)
        ]
    return output


def iter_unique_values(rows: Sequence[Dict[str, Any]], key: str) -> List[str]:
    seen: Set[str] = set()
    uniq: List[str] = []
    for row in rows:
        value = sanitize_text(row.get(key))
        if not value or value in seen:
            continue
        seen.add(value)
        uniq.append(value)
    return uniq


def rank_of(gt: str, ranked: Sequence[str]) -> Optional[int]:
    try:
        return int(list(ranked).index(gt)) + 1
    except ValueError:
        return None


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_ranks(ranks: Sequence[int]) -> Dict[str, float]:
    if not ranks:
        return {"MnR": 0.0, "MdR": 0.0}
    values = [int(r) for r in ranks]
    return {
        "MnR": round(float(sum(values) / len(values)), 4),
        "MdR": round(float(median(values)), 4),
    }


def summarize_category_metrics(
    category_summaries: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    if not category_summaries:
        return {}
    metric_keys: Set[str] = set()
    evaluated_total = 0
    skipped_total = 0
    for summary in category_summaries.values():
        evaluated_total += int(summary.get("evaluated", 0))
        skipped_total += int(summary.get("skipped_missing_features", 0))
        for key in summary:
            if key.startswith("recall@") or key in {"MnR", "MdR"}:
                metric_keys.add(key)
    output: Dict[str, float] = {
        "evaluated_total": int(evaluated_total),
        "skipped_missing_features_total": int(skipped_total),
        "num_categories": int(len(category_summaries)),
    }
    for key in sorted(metric_keys):
        output[f"macro_{key}"] = round(
            float(
                sum(
                    float(summary.get(key, 0.0))
                    for summary in category_summaries.values()
                )
                / len(category_summaries)
            ),
            6,
        )
    return output
