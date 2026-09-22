from src.pipeline.rules_ner import extract_entities


def types_and_texts(text):
    return [(s["type"], s["text"]) for s in extract_entities(text)]


def test_http_error_status_codes():
    assert types_and_texts("The API returned 503 errors for most users.") == [("ERROR_CODE", "503")]
    assert types_and_texts("Clients saw HTTP 500 responses.") == [("ERROR_CODE", "HTTP 500")]


def test_non_error_status_codes_are_not_flagged_without_context():
    # "300 requests" không phải mã lỗi, chỉ là số lượng -> không nên bắt nhầm.
    assert extract_entities("We received 300 requests during the spike.") == []


def test_posix_error_tokens_and_exception_class_names():
    assert types_and_texts("Connections failed with ETIMEDOUT and ECONNRESET.") == [
        ("ERROR_CODE", "ETIMEDOUT"),
        ("ERROR_CODE", "ECONNRESET"),
    ]
    assert types_and_texts("A NullPointerException was thrown repeatedly.") == [
        ("ERROR_CODE", "NullPointerException")
    ]


def test_iso8601_and_clock_time_and_calendar_date():
    assert types_and_texts("At 2021-11-25T10:00:00Z the incident began.")[0] == ("TIME_EXPR", "2021-11-25T10:00:00Z")
    assert ("TIME_EXPR", "18:50 UTC") in types_and_texts("At 18:50 UTC we rolled back the deploy.")
    assert ("TIME_EXPR", "May 25, 2024") in types_and_texts("On May 25, 2024 the incident occurred.")


def test_percentage_duration_and_multiplier_metrics():
    result = types_and_texts("Traffic dropped 20% and error rates were 3x normal for 45 minutes.")
    assert ("METRIC", "20%") in result
    assert ("METRIC", "3x") in result
    assert ("METRIC", "45 minutes") in result


def test_overlapping_rule_matches_keep_earlier_rule_only():
    # "HTTP 500" khớp cả rule "HTTP [45]xx" (rule đầu) lẫn rule "[45]xx(?=...)" nếu theo sau có từ khoá.
    text = "Requests failed with HTTP 500 status."
    start = text.index("HTTP 500")
    spans = extract_entities(text)
    assert [(s["start"], s["end"], s["text"]) for s in spans] == [(start, start + len("HTTP 500"), "HTTP 500")]


def test_no_matches_on_plain_prose():
    assert extract_entities("The team investigated the root cause of the outage.") == []
