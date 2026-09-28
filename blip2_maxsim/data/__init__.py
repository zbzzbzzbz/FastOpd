"""
文件名称: __init__.py
功能描述: 本地化数据模块导出
创建日期: 2026-05-20
作者: GPT-5.4
"""

from .fashioniq import (
    FashionIQDataModule,
    FashionIQPairDataset,
    FashionIQQueryDataset,
    FashionIQTargetDataset,
    build_candidate_bank_rows,
    build_transform,
    resolve_fashioniq_image_path,
    summarize_rows_by_category,
)

__all__ = [
    "FashionIQDataModule",
    "FashionIQPairDataset",
    "FashionIQQueryDataset",
    "FashionIQTargetDataset",
    "build_candidate_bank_rows",
    "build_transform",
    "resolve_fashioniq_image_path",
    "summarize_rows_by_category",
]
