"""Chuẩn bị dữ liệu train NER (không phụ thuộc torch): nạp IOB2, chia fold theo tài liệu, giải mã nhãn."""

import json
import random
from pathlib import Path

from src.iob2 import IGNORE_INDEX
from src.models.taxonomy import IOB2_LABELS

ID2LABEL = dict(enumerate(IOB2_LABELS))


def load_iob2_rows(iob2_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(iob2_dir.glob("INC-*.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def make_document_folds(incident_ids: list[str], k: int, seed: int) -> list[list[str]]:
    """Chia tài liệu (không chia chunk) thành k nhóm để cross-validation. Các chunk
    cùng tài liệu chia sẻ từ vựng/văn phong nên phải nằm cùng một phía, nếu không
    điểm đánh giá bị thổi phồng do rò rỉ dữ liệu."""
    unique = sorted(set(incident_ids))
    k = min(k, len(unique))
    random.Random(seed).shuffle(unique)
    return [sorted(unique[i::k]) for i in range(k)]


def decode_tags(label_ids: list[int], pred_ids: list[int]) -> tuple[list[str], list[str]]:
    """Trả (nhãn đúng, nhãn dự đoán) dạng chuỗi IOB2, bỏ các vị trí -100 (token đặc biệt)."""
    true_tags, pred_tags = [], []
    for label, pred in zip(label_ids, pred_ids):
        if label == IGNORE_INDEX:
            continue
        true_tags.append(ID2LABEL[label])
        pred_tags.append(ID2LABEL[pred])
    return true_tags, pred_tags
