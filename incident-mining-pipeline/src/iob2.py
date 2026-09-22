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


def tags_to_spans(
    text: str,
    offsets: list[tuple[int, int]],
    tags: list[str | None],
    confidences: list[float] | None = None,
) -> list[dict]:
    """Chiều ngược lại của spans_to_iob2_tags: ghép nhãn B-/I- liên tiếp của model
    (hoặc dữ liệu gold) thành span mức ký tự. Dùng khi suy luận NER thật (Layer 2),
    không chỉ lúc chuẩn bị dữ liệu train.

    confidences (nếu có, cùng độ dài offsets) là xác suất model cho token đó; span
    lấy trung bình các token thuộc nó làm confidence.
    """
    spans: list[dict] = []
    current: dict | None = None

    def close_current() -> None:
        if current is None:
            return
        # Token đầu một từ trong tokenizer SentencePiece (vd DeBERTa-v3) có offset
        # bao gồm luôn khoảng trắng phía trước ("▁WAF" -> offset trùm cả dấu cách),
        # nên phải cắt bớt whitespace ở 2 đầu span trước khi trả ra.
        start, end = current["start"], current["end"]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "type": current["type"],
                    "confidence": (sum(current["confs"]) / len(current["confs"])) if current["confs"] else 1.0,
                }
            )

    for i, ((start, end), tag) in enumerate(zip(offsets, tags)):
        if tag is None or tag == "O":
            close_current()
            current = None
            continue
        prefix, _, entity_type = tag.partition("-")
        if current is None or entity_type != current["type"] or prefix == "B":
            close_current()
            current = {"start": start, "end": end, "type": entity_type, "confs": []}
        else:
            current["end"] = end
        if confidences is not None:
            current["confs"].append(confidences[i])
    close_current()
    return spans


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
