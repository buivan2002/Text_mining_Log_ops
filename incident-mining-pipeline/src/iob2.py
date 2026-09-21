"""Chuyển span mức ký tự thành nhãn IOB2 mức token (logic thuần, không phụ thuộc tokenizer)."""

from src.labeling import dedupe_spans, suggest_keep_flags
from src.models.taxonomy import IOB2_LABELS

LABEL2ID = {label: i for i, label in enumerate(IOB2_LABELS)}
IGNORE_INDEX = -100  # token đặc biệt ([CLS], [SEP]): hàm loss của HuggingFace bỏ qua


def flatten_spans(spans: list[dict]) -> list[dict]:
    """Bỏ span trùng, rồi giữ span dài hơn khi chồng lấn để còn lại tập span phẳng
    (IOB2 mỗi token chỉ mang một nhãn)."""
    unique = dedupe_spans(spans)
    keep = suggest_keep_flags(unique)
    return sorted((s for s, k in zip(unique, keep) if k), key=lambda s: (s["start"], s["end"]))


def spans_to_iob2_tags(offsets: list[tuple[int, int]], spans: list[dict]) -> list[str | None]:
    """Gán nhãn cho từng token dựa trên offset ký tự (start, end) của token.

    Token có offset (0, 0) là token đặc biệt -> None. Token giao với một span thì mang
    nhãn của span đó: token đầu tiên của span là B-<TYPE>, các token sau (kể cả mảnh
    con của cùng một từ) là I-<TYPE>. Token không giao span nào là O.
    """
    tags: list[str | None] = []
    previous_span = None
    for token_start, token_end in offsets:
        if token_start == token_end:
            tags.append(None)
            previous_span = None
            continue
        hit = next(
            (i for i, s in enumerate(spans) if token_start < s["end"] and token_end > s["start"]),
            None,
        )
        if hit is None:
            tags.append("O")
        else:
            prefix = "I" if hit == previous_span else "B"
            tags.append(f"{prefix}-{spans[hit]['type']}")
        previous_span = hit
    return tags


def tags_to_label_ids(tags: list[str | None]) -> list[int]:
    return [IGNORE_INDEX if t is None else LABEL2ID[t] for t in tags]


def misaligned_spans(text: str, offsets: list[tuple[int, int]], spans: list[dict]) -> list[dict]:
    """Span mà ranh giới không trùng ranh giới token (vd span "503" nằm giữa token
    "503errors"): nhãn token khi đó bao trùm nhiều hơn span gốc."""
    bad = []
    for s in spans:
        covered = [(a, b) for a, b in offsets if a != b and a < s["end"] and b > s["start"]]
        if not covered:
            bad.append(s)
            continue
        covered_text = text[min(a for a, _ in covered):max(b for _, b in covered)].strip()
        if covered_text != s["text"]:
            bad.append(s)
    return bad
