from unittest.mock import patch

from src.llm.ollama_client import LLMExtractionError
from src.pipeline.layer3_llm_relation import build_prompt, extract_relations

ENTITIES = [
    {"id": "e0", "type": "ROOT_CAUSE", "text": "a deploy to change a field type"},
    {"id": "e1", "type": "SYMPTOM", "text": "distribution to fail"},
]


def test_extract_relations_returns_empty_without_calling_llm_when_no_entities():
    with patch("src.pipeline.layer3_llm_relation.OllamaClient") as mock_client_cls:
        result = extract_relations("some text", [])
    assert result == []
    mock_client_cls.assert_not_called()


@patch("src.pipeline.layer3_llm_relation.OllamaClient")
def test_extract_relations_returns_relations_from_llm(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.generate_structured.return_value = {
        "relations": [{"source_id": "e0", "target_id": "e1", "relation_type": "CAUSES", "confidence": 0.9}]
    }

    result = extract_relations("text", ENTITIES)

    assert result == [{"source_id": "e0", "target_id": "e1", "relation_type": "CAUSES", "confidence": 0.9}]
    prompt_used = mock_client.generate_structured.call_args.args[0]
    assert "e0" in prompt_used and "e1" in prompt_used


@patch("src.pipeline.layer3_llm_relation.OllamaClient")
def test_extract_relations_returns_empty_when_llm_extraction_fails(mock_client_cls):
    mock_client_cls.return_value.generate_structured.side_effect = LLMExtractionError("boom")
    assert extract_relations("text", ENTITIES) == []


@patch("src.pipeline.layer3_llm_relation.OllamaClient")
def test_extract_relations_defaults_to_empty_list_when_key_missing(mock_client_cls):
    mock_client_cls.return_value.generate_structured.return_value = {}
    assert extract_relations("text", ENTITIES) == []


@patch("src.pipeline.layer3_llm_relation.OllamaClient")
def test_extract_relations_passes_previous_errors_into_prompt(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.generate_structured.return_value = {"relations": []}

    extract_relations("text", ENTITIES, previous_errors=["source_id e9 không tồn tại"])

    prompt_used = mock_client.generate_structured.call_args.args[0]
    assert "e9 không tồn tại" in prompt_used


def test_build_prompt_lists_all_entities_and_relation_constraints():
    prompt = build_prompt("some incident text", ENTITIES, previous_errors=None)
    assert 'e0 [ROOT_CAUSE] "a deploy to change a field type"' in prompt
    assert 'e1 [SYMPTOM] "distribution to fail"' in prompt
    assert "CAUSES" in prompt and "MITIGATED_BY" in prompt
    assert "some incident text" in prompt
    assert "Lần trả lời trước" not in prompt


def test_build_prompt_omits_previous_errors_block_when_none():
    prompt = build_prompt("text", [], previous_errors=[])
    assert "Lần trả lời trước" not in prompt
