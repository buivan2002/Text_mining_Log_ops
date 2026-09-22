"""NER dựa trên regex cho 3 loại entity dễ nhận diện bằng luật cố định.

Dùng làm phần "chắc chắn đúng" trong ensemble ở Layer 2: DeBERTa fine-tune trên
ít dữ liệu học các loại này không tốt bằng regex, vì chúng có định dạng cố định
(mã HTTP, số phần trăm/thời lượng, mốc thời gian) chứ không cần hiểu ngữ nghĩa.
Không xử lý SERVICE/COMPONENT/ROOT_CAUSE/SYMPTOM/IMPACT/MITIGATION/TEAM — các
loại đó cần hiểu ngữ cảnh, để DeBERTa (Layer 2) và Qwen (Layer 3) đảm nhiệm.
"""

import re

# Mỗi rule là (loại entity, regex đã compile). Rule đứng trước được ưu tiên khi
# hai rule cùng khớp một vị trí (xem _drop_overlaps).
_RULES: list[tuple[str, re.Pattern]] = [
    # Mã lỗi: HTTP 4xx/5xx (chỉ coi lỗi client/server là ERROR_CODE, không tính 1xx-3xx),
    # token lỗi kiểu POSIX (ETIMEDOUT, ECONNRESET...), tên class exception/error.
    ("ERROR_CODE", re.compile(r"\bHTTP\s?[45]\d{2}\b", re.IGNORECASE)),
    ("ERROR_CODE", re.compile(r"\b[45]\d{2}\b(?=\s*(errors?|status(?:es)?|responses?|codes?)\b)", re.IGNORECASE)),
    ("ERROR_CODE", re.compile(r"\bE[A-Z]{3,}\b")),
    ("ERROR_CODE", re.compile(r"\b[A-Z][a-zA-Z]*(?:Exception|Error)\b")),
    # Mốc thời gian: ISO8601, "HH:MM(:SS)? UTC/giờ", ngày dạng "Month D, YYYY".
    (
        "TIME_EXPR",
        re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?\b"),
    ),
    # \s và hậu tố nằm CHUNG một nhóm optional: nếu không có UTC/AM/PM thì dấu cách
    # cũng không được khớp theo (nếu tách riêng \s? sẽ nuốt nhầm dấu cách phía sau
    # số giờ ngay cả khi không có hậu tố, ví dụ "00:16 " kèm khoảng trắng thừa).
    ("TIME_EXPR", re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?(\s(UTC|GMT|AM|PM|am|pm))?\b")),
    (
        "TIME_EXPR",
        re.compile(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2},?\s+\d{4}\b"
        ),
    ),
    # Số liệu: phần trăm, thời lượng, hệ số nhân.
    ("METRIC", re.compile(r"\b\d+(\.\d+)?\s?%")),
    (
        "METRIC",
        re.compile(
            r"\b\d+(\.\d+)?\s?(milliseconds?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
            re.IGNORECASE,
        ),
    ),
    ("METRIC", re.compile(r"\b\d+(\.\d+)?x\b")),
]

_RULE_CONFIDENCE = 0.95


def _drop_overlaps(spans: list[dict]) -> list[dict]:
    """Giữ span xuất hiện sớm hơn trong _RULES khi hai rule cùng khớp một vùng
    (vd "500" vừa khớp ERROR_CODE vừa có thể khớp METRIC nếu thêm rule số thô)."""
    kept: list[dict] = []
    for span in sorted(spans, key=lambda s: s["_rule_index"]):
        if not any(span["start"] < k["end"] and span["end"] > k["start"] for k in kept):
            kept.append(span)
    kept.sort(key=lambda s: s["start"])
    for span in kept:
        del span["_rule_index"]
    return kept


def extract_entities(text: str) -> list[dict]:
    """Trích entity ERROR_CODE/TIME_EXPR/METRIC bằng regex. Trả span mức ký tự
    tương đối với `text` truyền vào (thường là 1 chunk)."""
    spans = []
    for rule_index, (entity_type, pattern) in enumerate(_RULES):
        for match in pattern.finditer(text):
            spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "type": entity_type,
                    "confidence": _RULE_CONFIDENCE,
                    "_rule_index": rule_index,
                }
            )
    return _drop_overlaps(spans)
