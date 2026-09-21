from src.iob2 import (
    IGNORE_INDEX,
    LABEL2ID,
    flatten_spans,
    misaligned_spans,
    spans_to_iob2_tags,
    tags_to_label_ids,
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
