import json

import pytest

from src.pipeline import layer4_validation as layer4

ENTITIES = [
    {"id": "e0", "type": "ROOT_CAUSE", "text": "a deploy changed a field type", "confidence": 0.9},
    {"id": "e1", "type": "SYMPTOM", "text": "distribution failed", "confidence": 0.9},
    {"id": "e2", "type": "MITIGATION", "text": "rolled back the change", "confidence": 0.9},
    {"id": "e3", "type": "TIME_EXPR", "text": "18:50", "confidence": 0.9},
]


@pytest.fixture(autouse=True)
def fixed_confidence_threshold(monkeypatch):
    # Cố định ngưỡng để test không phụ thuộc giá trị mặc định trong .env/config.
    monkeypatch.setattr(layer4.settings, "relation_confidence_threshold", 0.6)


def relation(source, target, rel_type, confidence=0.9):
    return {"source_id": source, "target_id": target, "relation_type": rel_type, "confidence": confidence}


def test_valid_relation_is_kept():
    valid, errors = layer4.validate(ENTITIES, [relation("e0", "e1", "CAUSES")], "INC-1")
    assert errors == []
    assert len(valid) == 1 and valid[0].source_id == "e0" and valid[0].target_id == "e1"


def test_missing_source_id_is_dropped():
    valid, errors = layer4.validate(ENTITIES, [relation("e99", "e1", "CAUSES")], "INC-1")
    assert valid == []
    assert "e99" in errors[0] and "không tồn tại" in errors[0]


def test_missing_target_id_is_dropped():
    valid, errors = layer4.validate(ENTITIES, [relation("e0", "e99", "CAUSES")], "INC-1")
    assert valid == []
    assert "e99" in errors[0] and "không tồn tại" in errors[0]


def test_self_loop_is_dropped():
    valid, errors = layer4.validate(ENTITIES, [relation("e0", "e0", "CAUSES")], "INC-1")
    assert valid == []
    assert "self-loop" in errors[0]


def test_low_confidence_is_dropped():
    valid, errors = layer4.validate(ENTITIES, [relation("e0", "e1", "CAUSES", confidence=0.3)], "INC-1")
    assert valid == []
    assert "0.3" in errors[0] and "ngưỡng" in errors[0]


def test_invalid_relation_type_enum_is_dropped():
    valid, errors = layer4.validate(ENTITIES, [relation("e0", "e1", "DESTROYS")], "INC-1")
    assert valid == []
    assert "sai cấu trúc" in errors[0]


def test_constraint_violation_is_dropped_and_logged(tmp_path):
    # TIME_EXPR không nằm trong nguồn hợp lệ của CAUSES -> vi phạm ma trận ràng buộc.
    log_path = tmp_path / "INC-1.validation_errors.json"
    valid, errors = layer4.validate(ENTITIES, [relation("e3", "e1", "CAUSES")], "INC-1", log_path=log_path)
    assert valid == []
    assert "yêu cầu nguồn thuộc" in errors[0]
    assert log_path.exists()
    logged = json.loads(log_path.read_text(encoding="utf-8"))
    assert logged["incident_id"] == "INC-1"
    assert logged["violations"][0]["source_id"] == "e3"


def test_no_log_file_written_when_log_path_is_none():
    # Không truyền log_path -> validate() không được đụng ổ đĩa (mặc định test được thuần).
    valid, errors = layer4.validate(ENTITIES, [relation("e3", "e1", "CAUSES")], "INC-1", log_path=None)
    assert valid == [] and errors


def test_no_log_file_written_when_there_are_no_violations(tmp_path):
    log_path = tmp_path / "INC-1.validation_errors.json"
    layer4.validate(ENTITIES, [relation("e0", "e1", "CAUSES")], "INC-1", log_path=log_path)
    assert not log_path.exists()


def test_duplicate_triple_deduped_keeps_higher_confidence():
    relations = [
        relation("e0", "e1", "CAUSES", confidence=0.7),
        relation("e0", "e1", "CAUSES", confidence=0.95),
    ]
    valid, _ = layer4.validate(ENTITIES, relations, "INC-1")
    assert len(valid) == 1 and valid[0].confidence == 0.95


def test_missing_confidence_defaults_to_0_5(monkeypatch):
    monkeypatch.setattr(layer4.settings, "relation_confidence_threshold", 0.0)
    raw = {"source_id": "e0", "target_id": "e1", "relation_type": "CAUSES"}  # thiếu confidence, đúng như Qwen được phép trả
    valid, _ = layer4.validate(ENTITIES, [raw], "INC-1")
    assert valid[0].confidence == layer4.DEFAULT_CONFIDENCE


def test_missing_confidence_can_be_dropped_by_default_threshold():
    # Ngưỡng cố định 0.6 trong fixture > DEFAULT_CONFIDENCE (0.5) -> vẫn bị lọc như confidence thấp bình thường.
    raw = {"source_id": "e0", "target_id": "e1", "relation_type": "CAUSES"}
    valid, errors = layer4.validate(ENTITIES, [raw], "INC-1")
    assert valid == [] and "ngưỡng" in errors[0]


def test_bad_relation_does_not_crash_the_whole_batch():
    relations = [relation("e99", "e1", "CAUSES"), relation("e0", "e1", "CAUSES")]
    valid, errors = layer4.validate(ENTITIES, relations, "INC-1")
    assert len(valid) == 1 and len(errors) == 1
