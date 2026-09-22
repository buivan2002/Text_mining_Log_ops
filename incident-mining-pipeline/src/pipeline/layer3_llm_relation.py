"""Tầng 3: LLM Relation Extraction bằng Qwen (local, qua Ollama).

Nhận danh sách entity đã trích ở Layer 2, dùng LLM để suy luận quan hệ nhân
quả giữa chúng (nguyên nhân -> triệu chứng -> tác động -> giải pháp khắc
phục...). Model chỉ được CHỌN id có sẵn trong danh sách entity, không được tự
bịa entity mới -- đây là cách giảm mạnh hallucination so với để model tự do
sinh cả hai đầu quan hệ.
"""

import json

from src.config import settings
from src.llm.ollama_client import LLMExtractionError, OllamaClient
from src.models.taxonomy import RELATION_TYPE_CONSTRAINTS, RELATION_TYPES

RELATION_TYPE_DESCRIPTIONS = {
    "CAUSES": "nguồn là nguyên nhân trực tiếp gây ra đích",
    "TRIGGERS": "nguồn châm ngòi/kích hoạt đích xảy ra tiếp theo",
    "IMPACTS": "nguồn gây ảnh hưởng/tác động tới đích",
    "MITIGATED_BY": "nguồn (triệu chứng/tác động/nguyên nhân) được khắc phục bằng hành động ở đích",
    "DETECTED_BY": "nguồn được phát hiện bởi đội ngũ hoặc số liệu giám sát ở đích",
    "AFFECTS": "nguồn (dịch vụ) có chứa/ảnh hưởng tới thành phần ở đích",
    "OWNED_BY": "nguồn (dịch vụ/thành phần) do đội ngũ ở đích phụ trách",
}

_FEW_SHOT_ENTITIES = [
    {"id": "e0", "type": "TIME_EXPR", "text": "18:50"},
    {"id": "e1", "type": "COMPONENT", "text": "distribution database"},
    {"id": "e2", "type": "ROOT_CAUSE", "text": "a deploy to change a field type"},
    {"id": "e3", "type": "SYMPTOM", "text": "distribution to fail"},
    {"id": "e4", "type": "MITIGATION", "text": "rolled back the change"},
]
_FEW_SHOT_TEXT = (
    "At 18:50, we rolled out a deploy to change a field type in our distribution "
    "database, which caused distribution to fail. We immediately rolled back the change."
)
_FEW_SHOT_RELATIONS = [
    {
        "source_id": "e2",
        "target_id": "e3",
        "relation_type": "CAUSES",
        "confidence": 0.9,
        "evidence": "a deploy to change a field type ... caused distribution to fail",
    },
    {
        "source_id": "e3",
        "target_id": "e4",
        "relation_type": "MITIGATED_BY",
        "confidence": 0.85,
        "evidence": "We immediately rolled back the change",
    },
]

_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "relation_type": {"type": "string", "enum": RELATION_TYPES},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["source_id", "target_id", "relation_type"],
            },
        }
    },
    "required": ["relations"],
}


def _format_entities(entities: list[dict]) -> str:
    return "\n".join(f'{e["id"]} [{e["type"]}] "{e["text"]}"' for e in entities)


def _format_constraints() -> str:
    lines = []
    for relation_type in RELATION_TYPES:
        sources, targets = RELATION_TYPE_CONSTRAINTS[relation_type]
        lines.append(
            f"- {relation_type}: {RELATION_TYPE_DESCRIPTIONS[relation_type]} "
            f"(nguồn phải thuộc {sorted(sources)}, đích phải thuộc {sorted(targets)})"
        )
    return "\n".join(lines)


def build_prompt(text: str, entities: list[dict], previous_errors: list[str] | None) -> str:
    prompt = f"""Bạn đang phân tích một bài postmortem sự cố kỹ thuật để dựng cây quan hệ nhân quả.

Các loại quan hệ được phép:
{_format_constraints()}

Quy tắc:
- CHỈ được dùng source_id/target_id lấy từ danh sách entity dưới đây, không được tự bịa id mới.
- source_id và target_id phải khác nhau (không tự nối vào chính nó).
- relation_type phải đúng một trong các giá trị ở trên.
- "evidence" là đoạn trích nguyên văn (hoặc gần nguyên văn) trong bài chứng minh cho quan hệ đó.
- Chỉ trả các quan hệ có bằng chứng rõ ràng trong văn bản, không suy đoán.
- Trả JSON theo đúng schema: {{"relations": [...]}}. Không thêm text nào khác.

Ví dụ:
Danh sách entity:
{_format_entities(_FEW_SHOT_ENTITIES)}
Văn bản: "{_FEW_SHOT_TEXT}"
Kết quả: {json.dumps({"relations": _FEW_SHOT_RELATIONS})}

Bây giờ hãy làm với dữ liệu thật sau:
Danh sách entity:
{_format_entities(entities)}
Văn bản:
\"\"\"{text}\"\"\"
"""
    if previous_errors:
        errors_block = "\n".join(f"- {e}" for e in previous_errors)
        prompt += (
            f"\nLần trả lời trước của bạn có các quan hệ không hợp lệ, lý do:\n{errors_block}\n"
            "Hãy sửa lại và chỉ trả các quan hệ hợp lệ theo đúng quy tắc ở trên."
        )
    return prompt


def extract_relations(
    text: str, entities: list[dict], previous_errors: list[str] | None = None
) -> list[dict]:
    """Gọi Qwen để trích quan hệ giữa các entity đã có. Trả list[dict] thô (chưa
    qua Pydantic) -- Layer 4 chịu trách nhiệm validate đầy đủ.

    Trả [] ngay (không gọi LLM) nếu không có entity nào, vì không có gì để nối.
    Trả [] và in cảnh báo nếu Ollama lỗi liên tục, để một incident lỗi không
    làm crash cả batch.
    """
    if not entities:
        return []

    client = OllamaClient(base_url=settings.ollama_base_url, model=settings.llm_model_name)
    prompt = build_prompt(text, entities, previous_errors)

    try:
        result = client.generate_structured(prompt, schema=_SCHEMA, max_retries=3)
    except LLMExtractionError as exc:
        print(f"  [WARN] layer3_llm_relation: trích quan hệ thất bại: {exc}")
        return []

    return result.get("relations", [])
