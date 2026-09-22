from src.pipeline.layer2_ner import merge_rule_and_model_spans


def span(start, end, type_, confidence=0.9, text="x"):
    return {"start": start, "end": end, "text": text, "type": type_, "confidence": confidence}


def test_rule_span_kept_even_when_model_disagrees_on_same_range():
    rule = span(0, 5, "ERROR_CODE", text="503")
    model = span(0, 5, "SERVICE", text="503")  # model đoán sai loại cho đúng vùng đó
    merged = merge_rule_and_model_spans([rule], [model])
    assert merged == [rule]


def test_model_span_dropped_when_it_partially_overlaps_a_rule_span():
    rule = span(10, 20, "TIME_EXPR")
    model = span(15, 25, "COMPONENT")  # chồng lấn một phần, không trùng khít
    assert merge_rule_and_model_spans([rule], [model]) == [rule]


def test_non_overlapping_model_span_is_kept():
    rule = span(0, 5, "ERROR_CODE")
    model = span(10, 20, "SERVICE")
    merged = merge_rule_and_model_spans([rule], [model])
    assert merged == sorted([rule, model], key=lambda s: s["start"])


def test_empty_rule_spans_keeps_all_model_spans():
    model_spans = [span(0, 5, "SERVICE"), span(10, 15, "COMPONENT")]
    assert merge_rule_and_model_spans([], model_spans) == model_spans


def test_empty_model_spans_keeps_all_rule_spans():
    rule_spans = [span(0, 5, "ERROR_CODE"), span(10, 15, "METRIC")]
    assert merge_rule_and_model_spans(rule_spans, []) == rule_spans


def test_result_sorted_by_position_regardless_of_input_order():
    rule = span(20, 25, "METRIC")
    model = span(0, 5, "SERVICE")
    assert merge_rule_and_model_spans([rule], [model]) == [model, rule]
