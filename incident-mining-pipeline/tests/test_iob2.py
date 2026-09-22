from src.iob2 import (
    IGNORE_INDEX,
    LABEL2ID,
    flatten_spans,
    misaligned_spans,
    spans_to_iob2_tags,
    tags_to_label_ids,
    tags_to_spans,
)

# "At 18:50 the API gateway failed" với token giả lập kiểu sentencepiece (offset không gồm khoảng trắng)
TEXT = "At 18:50 the API gateway failed"
OFFSETS = [(0, 0), (0, 2), (3, 5), (5, 6), (6, 8), (9, 12), (13, 16), (17, 24), (25, 31), (0, 0)]
#          [CLS]  At      18      :       50      the      API       gateway   failed    [SEP]


def span(start, end, type_):
    return {"start": start, "end": end, "text": TEXT[start:end], "type": type_}


def test_special_tokens_are_none_and_outside_tokens_are_O():
    tags = spans_to_iob2_tags(OFFSETS, [])
    assert tags[0] is None and tags[-1] is None
    assert set(tags[1:-1]) == {"O"}


def test_multi_token_span_gets_B_then_I():
    tags = spans_to_iob2_tags(OFFSETS, [span(3, 8, "TIME_EXPR"), span(13, 24, "SERVICE")])
    assert tags == [None, "O", "B-TIME_EXPR", "I-TIME_EXPR", "I-TIME_EXPR", "O", "B-SERVICE", "I-SERVICE", "O", None]


def test_adjacent_different_spans_each_start_with_B():
    tags = spans_to_iob2_tags(OFFSETS, [span(13, 16, "SERVICE"), span(17, 24, "COMPONENT")])
    assert tags[6:8] == ["B-SERVICE", "B-COMPONENT"]


def test_label_ids_use_ignore_index_for_special_tokens():
    ids = tags_to_label_ids([None, "O", "B-SERVICE"])
    assert ids == [IGNORE_INDEX, LABEL2ID["O"], LABEL2ID["B-SERVICE"]]
    assert LABEL2ID["O"] == 0 and len(LABEL2ID) == 21


def test_flatten_spans_dedupes_and_drops_shorter_overlap():
    long_ = span(13, 24, "SERVICE")
    short = span(13, 16, "COMPONENT")
    flat = flatten_spans([short, long_, dict(long_)])
    assert flat == [long_]


def test_misaligned_spans_detects_boundary_inside_token():
    aligned = span(13, 16, "SERVICE")
    inside_token = {"start": 6, "end": 7, "text": "5", "type": "METRIC"}  # token "50" là (6, 8)
    assert misaligned_spans(TEXT, OFFSETS, [aligned]) == []
    assert misaligned_spans(TEXT, OFFSETS, [inside_token]) == [inside_token]


def test_tags_to_spans_rebuilds_multi_token_span_with_mean_confidence():
    tags = [None, "O", "B-TIME_EXPR", "I-TIME_EXPR", "I-TIME_EXPR", "O", "B-SERVICE", "I-SERVICE", "O", None]
    confidences = [0, 0.9, 0.8, 0.6, 0.7, 0.9, 0.5, 0.9, 0.9, 0]
    spans = tags_to_spans(TEXT, OFFSETS, tags, confidences)
    assert [{**s, "confidence": round(s["confidence"], 6)} for s in spans] == [
        {"start": 3, "end": 8, "text": "18:50", "type": "TIME_EXPR", "confidence": round((0.8 + 0.6 + 0.7) / 3, 6)},
        {"start": 13, "end": 24, "text": "API gateway", "type": "SERVICE", "confidence": round((0.5 + 0.9) / 2, 6)},
    ]


def test_tags_to_spans_treats_missing_B_as_new_span_start():
    # "I-SERVICE" xuất hiện mà không có "B-SERVICE" trước đó (model dự đoán lỗi) -> vẫn tạo span mới.
    tags = [None, "I-SERVICE", "I-SERVICE", "O", None]
    spans = tags_to_spans(TEXT, OFFSETS[:5], tags)
    assert len(spans) == 1 and spans[0]["type"] == "SERVICE"


def test_tags_to_spans_starts_new_span_on_adjacent_different_type():
    tags = [None, "O", "O", "O", "O", "O", "B-SERVICE", "B-COMPONENT", "O", None]
    spans = tags_to_spans(TEXT, OFFSETS, tags)
    assert [s["type"] for s in spans] == ["SERVICE", "COMPONENT"]


def test_tags_to_spans_defaults_confidence_to_one_when_not_given():
    spans = tags_to_spans(TEXT, OFFSETS, [None, "O", "B-TIME_EXPR"] + ["O"] * 7)
    assert spans[0]["confidence"] == 1.0


def test_tags_to_spans_strips_leading_whitespace_from_sentencepiece_token_offset():
    # token "API" tokenize kiểu SentencePiece có offset bắt đầu từ dấu cách phía trước nó.
    text = "the API gateway failed"
    offsets = [(3, 7), (7, 15), (15, 22)]  # " API", " gateway", " failed"
    tags = ["B-SERVICE", "I-SERVICE", "O"]
    spans = tags_to_spans(text, offsets, tags)
    assert spans == [{"start": 4, "end": 15, "text": "API gateway", "type": "SERVICE", "confidence": 1.0}]


def test_tags_to_spans_drops_span_that_is_only_whitespace():
    spans = tags_to_spans("a  b", [(1, 3)], ["B-SERVICE"])
    assert spans == []
