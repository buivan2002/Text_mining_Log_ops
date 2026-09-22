"""Tầng 2: Domain NER — DeBERTa đã fine-tune (Layer 2 training) + rule-based.

Nhận diện thực thể chuyên biệt trong log/sự cố: dịch vụ, thành phần hệ thống,
mã lỗi, chỉ số, thời gian, nguyên nhân, triệu chứng, tác động, biện pháp khắc
phục, đội ngũ phụ trách.

extract_entities() nhận nguyên bản ghi đã qua Layer 1 (dict có "chunks", mỗi
chunk có char_start/char_end) — không nhận thẳng 1 chuỗi text — vì model giới
hạn 512 token/lần và cần offset để gộp kết quả các chunk về tọa độ toàn tài
liệu. Đây là điểm khác với chữ ký gốc trong bản thiết kế ban đầu.
"""

from functools import lru_cache

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.iob2 import tags_to_spans
from src.labeling import dedupe_spans, suggest_keep_flags
from src.pipeline import rules_ner

MAX_LENGTH = 512  # phải khớp với max_length lúc train (scripts/train_ner.py)


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    """Nạp model + tokenizer một lần, dùng lại cho các lần gọi sau (model load
    mất vài giây, không nên load lại mỗi chunk/mỗi incident)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


@torch.no_grad()
def _predict_chunk(text: str, tokenizer, model, device) -> list[dict]:
    """Chạy DeBERTa trên 1 chunk, trả span kèm confidence = xác suất softmax
    trung bình của các token tạo nên span (chưa lọc theo ngưỡng)."""
    encoded = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    offsets = [tuple(o) for o in encoded.pop("offset_mapping")[0].tolist()]
    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = model(**encoded).logits[0]
    probs = torch.softmax(logits, dim=-1)
    confidences, pred_ids = probs.max(dim=-1)

    id2label = {int(i): label for i, label in model.config.id2label.items()}
    tags = [None if start == end else id2label[pred_id] for (start, end), pred_id in zip(offsets, pred_ids.tolist())]

    return tags_to_spans(text, offsets, tags, confidences.tolist())


def merge_rule_and_model_spans(rule_spans: list[dict], model_spans: list[dict]) -> list[dict]:
    """Rule thắng khi chồng lấn với model: 3 loại rule xử lý (ERROR_CODE/METRIC/
    TIME_EXPR) có định dạng cố định nên đáng tin hơn model học từ ít dữ liệu."""
    merged = list(rule_spans)
    for span in model_spans:
        overlaps_rule = any(span["start"] < r["end"] and span["end"] > r["start"] for r in rule_spans)
        if not overlaps_rule:
            merged.append(span)
    merged.sort(key=lambda s: (s["start"], s["end"]))
    return merged


def extract_entities_from_chunk(text: str, tokenizer, model, device, confidence_threshold: float) -> list[dict]:
    """NER cho 1 chunk: model + rule-based, đã lọc confidence và làm phẳng chồng lấn."""
    model_spans = [s for s in _predict_chunk(text, tokenizer, model, device) if s["confidence"] >= confidence_threshold]
    rule_spans = rules_ner.extract_entities(text)
    merged = merge_rule_and_model_spans(rule_spans, model_spans)

    unique = dedupe_spans(merged)
    keep = suggest_keep_flags(unique)
    return [s for s, k in zip(unique, keep) if k]


def extract_entities(processed_incident: dict, confidence_threshold: float | None = None) -> list[dict]:
    """Chạy NER trên toàn bộ tài liệu (tất cả chunk của processed_incident, dạng
    data/processed/<id>.json), gộp về tọa độ tài liệu, gán id toàn cục.

    Trả list[dict] khớp field của models.schemas.Entity (id, type, text,
    confidence), sẵn sàng để Layer 3 dùng.
    """
    from src.config import settings  # import trễ để test merge_rule_and_model_spans không cần settings/torch

    threshold = settings.ner_confidence_threshold if confidence_threshold is None else confidence_threshold
    tokenizer, model, device = _load_model(settings.ner_model_path)

    all_spans: list[dict] = []
    for chunk in processed_incident["chunks"]:
        chunk_spans = extract_entities_from_chunk(chunk["text"], tokenizer, model, device, threshold)
        offset = chunk["char_start"]
        for span in chunk_spans:
            all_spans.append({**span, "start": span["start"] + offset, "end": span["end"] + offset})

    # Layer 1 chia chunk có overlap_chars=150 -> cùng 1 entity có thể bị bắt ở 2 chunk liền kề.
    unique = dedupe_spans(all_spans)
    keep = suggest_keep_flags(unique)
    flat = sorted((s for s, k in zip(unique, keep) if k), key=lambda s: s["start"])

    return [
        {"id": f"e{i}", "type": s["type"], "text": s["text"], "confidence": round(s["confidence"], 3)}
        for i, s in enumerate(flat)
    ]
