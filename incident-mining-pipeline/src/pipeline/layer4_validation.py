"""Tầng 4: Logic & Schema Validation.

Kiểm tra các quan hệ do Layer 3 (Qwen) sinh ra trước khi ghi vào Neo4j: đúng
schema, endpoint tồn tại, không self-loop, đủ ngưỡng confidence, đúng ràng
buộc loại quan hệ <-> loại entity. Mỗi lý do bị loại sinh ra một message —
orchestrator dùng các message này để feed ngược lại prompt Layer 3, cho Qwen
cơ hội tự sửa (xem src/pipeline/layer3_llm_relation.py::build_prompt).
"""

import json
from pathlib import Path

from pydantic import ValidationError

from src.config import settings
from src.models.schemas import Relation
from src.models.taxonomy import RELATION_TYPE_CONSTRAINTS

# Qwen không bắt buộc phải trả "confidence" (JSON Schema chỉ required source_id/
# target_id/relation_type) -- thiếu thì coi như model không chắc, gán mức trung bình
# thay vì loại bỏ oan một quan hệ có cấu trúc đúng.
DEFAULT_CONFIDENCE = 0.5


def _format_pydantic_error(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())


def _dedup(relations: list[Relation]) -> list[Relation]:
    """Nếu Qwen trả trùng một cặp (source, target, relation_type), giữ bản có
    confidence cao hơn."""
    best: dict[tuple[str, str, str], Relation] = {}
    for relation in relations:
        key = (relation.source_id, relation.target_id, relation.relation_type)
        if key not in best or relation.confidence > best[key].confidence:
            best[key] = relation
    return list(best.values())


def _write_constraint_violations(log_path: Path, incident_id: str, violations: list[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({"incident_id": incident_id, "violations": violations}, f, ensure_ascii=False, indent=2)


def validate(
    entities: list[dict], relations: list[dict], incident_id: str, log_path: Path | None = None
) -> tuple[list[Relation], list[str]]:
    """Lọc `relations` (dict thô từ Layer 3) theo entity của CHÍNH incident này.

    entities: list[dict] dạng Layer 2 trả ra ({"id","type","text","confidence"}).
    log_path: nếu truyền vào, ghi chi tiết các quan hệ vi phạm ràng buộc loại ra
    file JSON để soi sau (mặc định None -- không ghi file, để hàm test được mà
    không cần đụng ổ đĩa).

    Trả (relations hợp lệ đã dedup, danh sách message lỗi để feed lại Layer 3).
    """
    entity_types = {e["id"]: e["type"] for e in entities}
    errors: list[str] = []
    constraint_violations: list[dict] = []
    accepted: list[Relation] = []

    for i, raw in enumerate(relations):
        candidate = dict(raw)
        candidate.setdefault("confidence", DEFAULT_CONFIDENCE)
        try:
            relation = Relation(**candidate)
        except ValidationError as exc:
            errors.append(f"Quan hệ #{i} sai cấu trúc ({_format_pydantic_error(exc)}), dữ liệu: {raw!r}")
            continue

        label = f"{relation.source_id} -[{relation.relation_type}]-> {relation.target_id}"

        if relation.source_id == relation.target_id:
            errors.append(f"{label}: source_id và target_id giống nhau (self-loop), không hợp lệ.")
            continue
        if relation.source_id not in entity_types:
            errors.append(f"{label}: source_id '{relation.source_id}' không tồn tại trong danh sách entity đã trích.")
            continue
        if relation.target_id not in entity_types:
            errors.append(f"{label}: target_id '{relation.target_id}' không tồn tại trong danh sách entity đã trích.")
            continue
        if relation.confidence < settings.relation_confidence_threshold:
            errors.append(
                f"{label}: confidence {relation.confidence} thấp hơn ngưỡng {settings.relation_confidence_threshold}."
            )
            continue

        source_type, target_type = entity_types[relation.source_id], entity_types[relation.target_id]
        allowed_sources, allowed_targets = RELATION_TYPE_CONSTRAINTS[relation.relation_type]
        if source_type not in allowed_sources or target_type not in allowed_targets:
            errors.append(
                f"{label}: {relation.relation_type} yêu cầu nguồn thuộc {sorted(allowed_sources)} "
                f"và đích thuộc {sorted(allowed_targets)}, nhưng nguồn là {source_type} và đích là {target_type}."
            )
            constraint_violations.append(
                {**relation.model_dump(), "source_type": source_type, "target_type": target_type}
            )
            continue

        accepted.append(relation)

    if log_path is not None and constraint_violations:
        _write_constraint_violations(log_path, incident_id, constraint_violations)

    return _dedup(accepted), errors
