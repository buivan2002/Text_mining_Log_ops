"""Tầng 3: LLM Relation Extraction (Qwen / GPT-4o).

Dùng LLM để suy luận quan hệ nhân quả giữa các thực thể đã nhận diện
(nguyên nhân -> triệu chứng -> tác động -> giải pháp khắc phục).
"""


def extract_relations(text: str, entities: list[dict]) -> list[dict]:
    """Gọi LLM để trích xuất quan hệ giữa các thực thể trong văn bản."""
    raise NotImplementedError
