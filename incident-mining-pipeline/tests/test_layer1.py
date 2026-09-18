from src.pipeline import layer1_preprocessing as layer1


def test_is_valid_fetch_rejects_redirect_page():
    assert layer1.is_valid_fetch("Redirecting...") is False


def test_is_valid_fetch_rejects_empty():
    assert layer1.is_valid_fetch("") is False
    assert layer1.is_valid_fetch(None) is False


def test_is_valid_fetch_accepts_real_content():
    assert layer1.is_valid_fetch("x" * 200) is True


def test_is_truncated_boundary():
    assert layer1.is_truncated("x" * 9999) is False
    assert layer1.is_truncated("x" * 10000) is True


def test_detect_language_english():
    assert layer1.detect_language("The service went down at 10:00 UTC.") == "en"


def test_detect_language_non_english():
    assert layer1.detect_language("서비스 장애가 발생했습니다" * 5) == "non-en"


def test_preprocess_strips_inline_boilerplate():
    raw = "Incident summary Skip to main content the outage started at noon."
    cleaned = layer1.preprocess(raw)
    assert "skip to main content" not in cleaned.lower()
    assert "outage started at noon" in cleaned


def test_preprocess_collapses_whitespace():
    raw = "Line one.\n\n\n\nLine   two."
    cleaned = layer1.preprocess(raw)
    assert "\n\n\n" not in cleaned
    assert "Line   two" not in cleaned


def test_chunk_text_offsets_are_correct():
    text = layer1.preprocess(
        "The API gateway failed at 10:00 UTC. "
        "This caused a 503 error for most users. "
        "The team rolled back the deploy within 20 minutes."
    )
    chunks = layer1.chunk_text(text, max_chars=1200, overlap_chars=150)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert text[chunk["char_start"]:chunk["char_end"]] == chunk["text"]


def test_chunk_text_respects_max_chars_for_multi_sentence_input():
    sentence = "The service degraded due to a configuration error. "
    text = layer1.preprocess(sentence * 40)
    chunks = layer1.chunk_text(text, max_chars=200, overlap_chars=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 200 + len(sentence)


def test_chunk_text_empty_input():
    assert layer1.chunk_text("") == []


def test_parse_log_lines_extracts_timestamp_and_status():
    raw = "At 2021-11-25T10:00:00Z the API returned 503 due to ETIMEDOUT."
    fragments = layer1.parse_log_lines(raw)
    types = {f["type"] for f in fragments}
    assert "timestamp" in types
    assert "http_status" in types
    assert "error_token" in types
