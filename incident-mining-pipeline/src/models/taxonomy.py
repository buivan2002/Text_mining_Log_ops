"""Danh mục entity/relation types dùng chung cho toàn bộ pipeline.

Đây là nguồn chân lý duy nhất: Layer 2 (NER) dùng ENTITY_TYPES/IOB2_LABELS để
biết cần nhận diện loại nào, Layer 3 (LLM) dùng RELATION_TYPES để biết được
phép sinh quan hệ nào, Layer 4 dùng RELATION_TYPE_CONSTRAINTS để validate logic.
"""

ENTITY_TYPES = [
    "SERVICE",
    "COMPONENT",
    "ERROR_CODE",
    "METRIC",
    "TIME_EXPR",
    "ROOT_CAUSE",
    "SYMPTOM",
    "IMPACT",
    "MITIGATION",
    "TEAM",
]

RELATION_TYPES = [
    "CAUSES",
    "TRIGGERS",
    "IMPACTS",
    "MITIGATED_BY",
    "DETECTED_BY",
    "AFFECTS",
    "OWNED_BY",
]

# relation_type -> (allowed source entity types, allowed target entity types)
RELATION_TYPE_CONSTRAINTS: dict[str, tuple[set[str], set[str]]] = {
    "CAUSES": ({"ROOT_CAUSE", "COMPONENT", "SERVICE"}, {"SYMPTOM", "IMPACT", "COMPONENT"}),
    "TRIGGERS": ({"SYMPTOM", "COMPONENT"}, {"SYMPTOM", "IMPACT"}),
    "IMPACTS": ({"SYMPTOM", "COMPONENT", "SERVICE"}, {"IMPACT"}),
    "MITIGATED_BY": ({"SYMPTOM", "IMPACT", "ROOT_CAUSE"}, {"MITIGATION"}),
    "DETECTED_BY": ({"SYMPTOM", "IMPACT"}, {"TEAM", "METRIC"}),
    "AFFECTS": ({"SERVICE"}, {"COMPONENT"}),
    "OWNED_BY": ({"SERVICE", "COMPONENT"}, {"TEAM"}),
}

# Nhãn IOB2 cho fine-tune NER: O + (B-/I-) cho mỗi entity type.
IOB2_LABELS = ["O"] + [f"{prefix}-{t}" for t in ENTITY_TYPES for prefix in ("B", "I")]
