"""Tầng 2: Domain Named Entity Recognition (DeBERTa / LogBERT).

Nhận diện các thực thể chuyên biệt trong log/sự cố: dịch vụ, thành phần
hệ thống, mã lỗi, chỉ số, thời gian xảy ra sự cố...
"""


def extract_entities(text: str) -> list[dict]:
    """Trích xuất thực thể domain-specific từ văn bản đã tiền xử lý."""
    raise NotImplementedError
