"""
文件名称: fashioniq.py
功能描述: 提供 FashionIQ 在 BLIP-2 + MaxSim 框架下的数据集与 dataloader 封装
创建日期: 2026-05-24
作者: GPT-5.4
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as tvf

from blip2_maxsim.data.common import (
    compose_fashioniq_edit,
    load_fashioniq_candidate_ids,
    load_fashioniq_rows,
    parse_categories,
    resolve_fashioniq_layout,
)


Image.MAX_IMAGE_PIXELS = None


class TargetPad:
    """当图像长宽比过大时做对称 padding，尽量保持与 ReTrack 原预处理一致。"""

    def __init__(self, target_ratio: float) -> None:
        self.target_ratio = float(target_ratio)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        actual_ratio = max(width, height) / max(min(width, height), 1)
        if actual_ratio < self.target_ratio:
            return image
        scaled_max_wh = max(width, height) / self.target_ratio
        hp = max(int((scaled_max_wh - width) / 2), 0)
        vp = max(int((scaled_max_wh - height) / 2), 0)
        return tvf.pad(image, [hp, vp, hp, vp], 0, "constant")


def build_transform(image_size: int):
    """构建和现有 BLIP2 backbone 一致的图像预处理。"""
    return transforms.Compose(
        [
            TargetPad(1.25),
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def resolve_fashioniq_image_path(
    images_root: str | Path, image_id: str, dress_type: str = ""
) -> Path:
    """
    按多种常见目录布局解析 FashionIQ 图片路径。

    不同来源的数据集可能是平铺在同一目录，也可能按类别再套一层目录，
    这里统一兼容，避免脚本对单一目录结构过度假设。
    """
    root = Path(images_root)
    candidates: List[Path] = []
    if dress_type:
        candidates.extend(
            [
                root / dress_type / image_id,
                root / dress_type / f"{image_id}",
            ]
        )
    candidates.extend([root / image_id, root / f"{image_id}"])
    for base_path in candidates:
        for suffix in ("", ".jpg", ".jpeg", ".png", ".webp"):
            path = Path(str(base_path) + suffix)
            if path.exists():
                return path
    raise FileNotFoundError(
        f"Image file not found for image_id={image_id}, dress_type={dress_type}, root={root}"
    )


class FashionIQPairDataset(Dataset):
    """训练或离线抽取时使用的 query-target 成对样本数据集。"""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        split: str,
        categories: Sequence[str],
        transform,
        max_rows: int = 0,
    ) -> None:
        super().__init__()
        self.layout = resolve_fashioniq_layout(dataset_root)
        self.categories = parse_categories(list(categories))
        self.transform = transform
        self.rows = load_fashioniq_rows(
            dataset_root,
            split=split,
            categories=self.categories,
            max_rows=max_rows,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        dress_type = str(row["dress_type"])
        ref_path = resolve_fashioniq_image_path(
            self.layout.images_dir, str(row["candidate_id"]), dress_type
        )
        tar_path = resolve_fashioniq_image_path(
            self.layout.images_dir, str(row["target_id"]), dress_type
        )
        with Image.open(ref_path) as image:
            ref_img = self.transform(image.convert("RGB"))
        with Image.open(tar_path) as image:
            tar_img = self.transform(image.convert("RGB"))
        return {
            "ref_img": ref_img,
            "tar_img": tar_img,
            "edit": str(row["edit"]),
            "pair_id": int(row["query_id"]),
            "dress_type": dress_type,
            "candidate_id": str(row["candidate_id"]),
            "target_id": str(row["target_id"]),
            "captions": list(row.get("captions", [])),
            "ref_description": "",
            "tag_description": "",
            # 这两个键保留用于后续扩展额外文本描述，当前基线不使用。
            # 在 FashionIQ 里不承载实际语义信息。
            "ref_webvid_caption": "",
            "tag_webvid_caption": "",
        }


class FashionIQQueryDataset(Dataset):
    """评测 query 数据集，只返回参考图和编辑文本，不重复加载目标图。"""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        split: str,
        categories: Sequence[str],
        transform,
        max_rows: int = 0,
    ) -> None:
        super().__init__()
        self.layout = resolve_fashioniq_layout(dataset_root)
        self.categories = parse_categories(list(categories))
        self.transform = transform
        self.rows = load_fashioniq_rows(
            dataset_root,
            split=split,
            categories=self.categories,
            max_rows=max_rows,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        dress_type = str(row["dress_type"])
        ref_path = resolve_fashioniq_image_path(
            self.layout.images_dir, str(row["candidate_id"]), dress_type
        )
        with Image.open(ref_path) as image:
            ref_img = self.transform(image.convert("RGB"))
        return {
            "ref_img": ref_img,
            "edit": str(row["edit"]),
            "pair_id": int(row["query_id"]),
            "dress_type": dress_type,
            "candidate_id": str(row["candidate_id"]),
            "target_id": str(row["target_id"]),
            "captions": list(row.get("captions", [])),
            "ref_description": "",
            "tag_description": "",
            # 这两个键保留用于后续扩展额外文本描述，当前基线不使用。
            # 在 FashionIQ 里不承载实际语义信息。
            "ref_webvid_caption": "",
            "tag_webvid_caption": "",
        }


class FashionIQTargetDataset(Dataset):
    """按类别组织的候选图库数据集，用于验证/测试时构建 target bank。"""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        split: str,
        dress_type: str,
        transform,
        item_ids: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.layout = resolve_fashioniq_layout(dataset_root)
        self.dress_type = str(dress_type)
        self.transform = transform
        if item_ids is None:
            candidate_map = load_fashioniq_candidate_ids(
                dataset_root, split=split, categories=[self.dress_type]
            )
            self.item_ids = list(candidate_map.get(self.dress_type, []))
        else:
            self.item_ids = [str(item) for item in item_ids]

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item_id = self.item_ids[index]
        image_path = resolve_fashioniq_image_path(
            self.layout.images_dir, item_id, self.dress_type
        )
        with Image.open(image_path) as image:
            tar_img = self.transform(image.convert("RGB"))
        return {
            "tar_img": tar_img,
            "item_id": str(item_id),
            "dress_type": str(self.dress_type),
        }


class FashionIQDataModule:
    """训练 dataloader 与验证 query/target bank 构建器。"""

    def __init__(
        self,
        *,
        dataset_root: str,
        categories: Sequence[str],
        train_split: str = "train",
        eval_split: str = "val",
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        max_train_rows: int = 0,
        max_eval_rows: int = 0,
        eval_batch_size: int | None = None,
    ) -> None:
        self.dataset_root = str(dataset_root)
        self.categories = parse_categories(list(categories))
        self.train_split = str(train_split)
        self.eval_split = str(eval_split)
        self.batch_size = int(batch_size)
        self.eval_batch_size = (
            int(eval_batch_size) if eval_batch_size is not None else int(batch_size)
        )
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.train_transform = build_transform(int(image_size))
        self.eval_transform = build_transform(int(image_size))
        self.data_train = FashionIQPairDataset(
            dataset_root=self.dataset_root,
            split=self.train_split,
            categories=self.categories,
            transform=self.train_transform,
            max_rows=max_train_rows,
        )
        self.data_eval_queries = FashionIQQueryDataset(
            dataset_root=self.dataset_root,
            split=self.eval_split,
            categories=self.categories,
            transform=self.eval_transform,
            max_rows=max_eval_rows,
        )
        self.eval_candidate_ids_by_category = load_fashioniq_candidate_ids(
            self.dataset_root,
            split=self.eval_split,
            categories=self.categories,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def eval_query_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_eval_queries,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def build_eval_target_loader(
        self, dress_type: str, batch_size: int | None = None
    ) -> DataLoader:
        dataset = FashionIQTargetDataset(
            dataset_root=self.dataset_root,
            split=self.eval_split,
            dress_type=str(dress_type),
            transform=self.eval_transform,
            item_ids=self.eval_candidate_ids_by_category.get(str(dress_type), []),
        )
        return DataLoader(
            dataset=dataset,
            batch_size=(
                int(batch_size) if batch_size is not None else self.eval_batch_size
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


def summarize_rows_by_category(rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """统计各类别样本数，便于在训练启动前快速确认数据分布。"""
    summary: Dict[str, int] = {}
    for row in rows:
        dress_type = str(row.get("dress_type", ""))
        summary[dress_type] = int(summary.get(dress_type, 0)) + 1
    return summary


def build_candidate_bank_rows(
    *,
    dataset_root: str | Path,
    split: str,
    categories: Sequence[str],
) -> List[Dict[str, str]]:
    """
    把 FashionIQ 的候选图库展开成图片行列表，供离线提取 target bank 使用。
    """
    rows: List[Dict[str, str]] = []
    candidate_ids_by_category = load_fashioniq_candidate_ids(
        dataset_root, split=split, categories=categories
    )
    for dress_type, item_ids in candidate_ids_by_category.items():
        for item_id in item_ids:
            rows.append(
                {
                    "dress_type": str(dress_type),
                    "item_id": str(item_id),
                    "edit": compose_fashioniq_edit([]),
                }
            )
    return rows
