"""Tầng 1: Preprocessing & Log Parsing.

Chuẩn hóa văn bản postmortem / log thô: loại bỏ ký tự nhiễu, tách câu,
parse timestamp, chuẩn hóa định dạng log trước khi đưa vào tầng NER.
"""


def preprocess(raw_text: str) -> str:
    """Làm sạch và chuẩn hóa văn bản đầu vào."""
    raise NotImplementedError


def parse_log_lines(raw_text: str) -> list[dict]:
    """Tách log thô thành các dòng có cấu trúc (timestamp, level, message)."""
    raise NotImplementedError
