from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.llm.ollama_client import LLMExtractionError, OllamaClient


def _mock_response(response_text: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": response_text}
    return mock_resp


@patch("src.llm.ollama_client.httpx.post")
def test_generate_structured_succeeds_first_try(mock_post):
    mock_post.return_value = _mock_response('{"entities": []}')
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:3b-instruct")

    result = client.generate_structured("extract entities", schema={"type": "object"})

    assert result == {"entities": []}
    assert mock_post.call_count == 1


@patch("src.llm.ollama_client.httpx.post")
def test_generate_structured_retries_on_invalid_json_then_succeeds(mock_post):
    mock_post.side_effect = [
        _mock_response("not json at all"),
        _mock_response('{"entities": [{"type": "SERVICE"}]}'),
    ]
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:3b-instruct")

    result = client.generate_structured("extract entities", schema={"type": "object"}, max_retries=3)

    assert result == {"entities": [{"type": "SERVICE"}]}
    assert mock_post.call_count == 2
    second_call_prompt = mock_post.call_args_list[1].kwargs["json"]["prompt"]
    assert "not json at all" in second_call_prompt or "lỗi" in second_call_prompt.lower()


@patch("src.llm.ollama_client.httpx.post")
def test_generate_structured_raises_after_exhausting_retries(mock_post):
    mock_post.return_value = _mock_response("still not json")
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:3b-instruct")

    with pytest.raises(LLMExtractionError):
        client.generate_structured("extract entities", schema={"type": "object"}, max_retries=2)

    assert mock_post.call_count == 2


@patch("src.llm.ollama_client.httpx.post")
def test_generate_structured_retries_on_timeout_then_succeeds(mock_post):
    mock_post.side_effect = [
        httpx.ReadTimeout("timed out"),
        _mock_response('{"entities": []}'),
    ]
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:3b-instruct")

    assert client.generate_structured("p", schema={"type": "object"}) == {"entities": []}
    assert mock_post.call_count == 2


@patch("src.llm.ollama_client.httpx.post")
def test_generate_structured_raises_when_every_attempt_times_out(mock_post):
    mock_post.side_effect = httpx.ReadTimeout("timed out")
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:3b-instruct")

    with pytest.raises(LLMExtractionError):
        client.generate_structured("p", schema={"type": "object"}, max_retries=2)


@patch("src.llm.ollama_client.httpx.post")
def test_generate_structured_caps_output_tokens(mock_post):
    mock_post.return_value = _mock_response('{"entities": []}')
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:3b-instruct")

    client.generate_structured("p", schema={"type": "object"}, max_output_tokens=512)

    assert mock_post.call_args.kwargs["json"]["options"] == {"num_predict": 512}


@patch("src.llm.ollama_client.httpx.post")
def test_generate_text_returns_raw_response(mock_post):
    mock_post.return_value = _mock_response("Ket qua tra loi.")
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:3b-instruct")

    assert client.generate_text("hoi gi do") == "Ket qua tra loi."
