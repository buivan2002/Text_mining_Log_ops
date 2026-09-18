"""Tầng 1: Preprocessing & Log Parsing.

Chuẩn hóa văn bản postmortem thô lấy từ fetch_github_postmortems: lọc bản ghi
fetch lỗi, làm sạch boilerplate HTML còn sót, gắn cờ chất lượng dữ liệu
(ngôn ngữ, bị cắt cụt), và chia nhỏ thành chunk có offset để tầng NER map
ngược lại vị trí gốc.
"""

import re
import unicodedata

import pysbd

# Ngưỡng độ dài fetch lỗi: dưới mức này coi như không lấy được nội dung thật.
MIN_VALID_CHARS = 100

# Fetch script cắt cứng ở đúng 10000 ký tự với các bài dài -> đánh dấu để biết
# nội dung có thể bị cụt giữa chừng.
TRUNCATION_LENGTH = 10000

# Tỉ lệ ký tự non-ASCII vượt ngưỡng này thì coi là tài liệu không phải tiếng Anh.
NON_ASCII_LANGUAGE_THRESHOLD = 0.15

# Cụm rác lặp lại từ HTML->text extraction chưa sạch hoàn toàn. Có thể nằm
# trọn trên một dòng riêng, hoặc lẫn ngay giữa câu (nav text bị dính vào
# paragraph khi convert HTML sang text) -> phải xử lý cả hai trường hợp.
BOILERPLATE_PHRASES = [
    "skip to main content",
    "skip to content",
    "cookie policy",
    "accept cookies",
]
_BOILERPLATE_INLINE_RE = re.compile(
    "|".join(re.escape(p) for p in BOILERPLATE_PHRASES), re.IGNORECASE
)

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)

_LOG_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?\b"
)
_HTTP_STATUS_RE = re.compile(r"\b[1-5]\d{2}\b")
_ERROR_TOKEN_RE = re.compile(
    r"\b(E[A-Z]{3,}|[A-Z][a-z]*Exception|[A-Z][a-z]*Error)\b"
)


def is_valid_fetch(raw_text: str) -> bool:
    """True nếu raw_text trông như nội dung thật, False nếu là fetch lỗi
    (trang redirect, rỗng, hoặc quá ngắn để có ý nghĩa)."""
    if raw_text is None:
        return False
    stripped = raw_text.strip()
    if len(stripped) < MIN_VALID_CHARS:
        return False
    if stripped.lower() == "redirecting...":
        return False
    return True


def is_truncated(raw_text: str) -> bool:
    """True nếu độ dài raw_text chạm đúng ngưỡng cắt cứng của fetch script."""
    return len(raw_text) >= TRUNCATION_LENGTH


def detect_language(text: str) -> str:
    """Heuristic rẻ tiền: tỉ lệ ký tự non-ASCII cao -> "non-en", ngược lại "en"."""
    if not text:
        return "en"
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ratio = non_ascii / len(text)
    return "non-en" if ratio > NON_ASCII_LANGUAGE_THRESHOLD else "en"


def preprocess(raw_text: str) -> str:
    """Làm sạch và chuẩn hóa văn bản đầu vào: NFKC normalize, loại boilerplate
    lặp lại, gộp khoảng trắng/dòng trống thừa."""
    normalized = unicodedata.normalize("NFKC", raw_text)
    text = _BOILERPLATE_INLINE_RE.sub(" ", normalized)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def chunk_text(cleaned_text: str, max_chars: int = 1200, overlap_chars: int = 150) -> list[dict]:
    """Chia văn bản đã làm sạch thành các chunk theo ranh giới câu, giữ
    char_start/char_end để tầng NER map span đã trích xuất về offset gốc.

    Chunk kế tiếp lùi lại `overlap_chars` từ cuối chunk trước để không cắt đứt
    ngữ cảnh ngay tại ranh giới.
    """
    if not cleaned_text:
        return []

    sentences = _SEGMENTER.segment(cleaned_text)

    # Định vị lại từng câu trong văn bản gốc để có offset chính xác
    # (pysbd không trả offset trực tiếp).
    spans = []
    cursor = 0
    for sentence in sentences:
        start = cleaned_text.find(sentence, cursor)
        if start == -1:
            # fallback hiếm gặp khi pysbd chỉnh sửa whitespace của câu
            start = cursor
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end

    chunks = []
    chunk_start_idx = 0
    while chunk_start_idx < len(spans):
        char_start = spans[chunk_start_idx][0]
        char_end = char_start
        idx = chunk_start_idx
        while idx < len(spans) and (spans[idx][1] - char_start) <= max_chars:
            char_end = spans[idx][1]
            idx += 1
        if idx == chunk_start_idx:
            # một câu đơn đã dài hơn max_chars -> vẫn phải lấy trọn câu đó
            char_end = spans[chunk_start_idx][1]
            idx = chunk_start_idx + 1

        chunks.append(
            {
                "chunk_id": f"c{len(chunks)}",
                "text": cleaned_text[char_start:char_end],
                "char_start": char_start,
                "char_end": char_end,
            }
        )

        if idx >= len(spans):
            break

        # lùi lại overlap_chars để tìm điểm bắt đầu chunk kế tiếp
        overlap_target = char_end - overlap_chars
        next_start_idx = idx - 1
        while next_start_idx > chunk_start_idx and spans[next_start_idx][0] > overlap_target:
            next_start_idx -= 1
        chunk_start_idx = max(next_start_idx, chunk_start_idx + 1) if idx > chunk_start_idx else idx

    return chunks


def parse_log_lines(raw_text: str) -> list[dict]:
    """Bóc các mảnh có cấu trúc xuất hiện trong văn xuôi postmortem (timestamp,
    mã HTTP status, token lỗi) để làm input tham khảo cho rule-based NER."""
    fragments = []
    for match in _LOG_TIMESTAMP_RE.finditer(raw_text):
        fragments.append({"type": "timestamp", "text": match.group(), "start": match.start(), "end": match.end()})
    for match in _HTTP_STATUS_RE.finditer(raw_text):
        fragments.append({"type": "http_status", "text": match.group(), "start": match.start(), "end": match.end()})
    for match in _ERROR_TOKEN_RE.finditer(raw_text):
        fragments.append({"type": "error_token", "text": match.group(), "start": match.start(), "end": match.end()})
    fragments.sort(key=lambda f: f["start"])
    return fragments
