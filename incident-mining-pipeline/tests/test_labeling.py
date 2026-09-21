import json

from src.labeling import (
    suggest_keep_flags,
    dedupe_spans,
    find_overlaps,
    is_missing,
    load_chunk_spans,
    locate_span,
    render_highlighted_html,
    resolve_rows,
    reviewed_chunk_ids,
    save_gold_chunk,
)

TEXT = "At 18:50 the API gateway failed. The API gateway was rolled back at 19:10."


def test_is_missing_handles_none_nan_and_pd_na():
    import pandas as pd

    assert is_missing(None) and is_missing(float("nan")) and is_missing(pd.NA)
    assert not is_missing(0) and not is_missing("x")


def test_locate_span_prefers_untaken_occurrence():
    first = locate_span(TEXT, "API gateway", taken=[])
    second = locate_span(TEXT, "API gateway", taken=[first])
    assert first == (TEXT.index("API gateway"), TEXT.index("API gateway") + len("API gateway"))
    assert second != first and TEXT[second[0]:second[1]] == "API gateway"


def test_locate_span_returns_none_when_absent():
    assert locate_span(TEXT, "database", taken=[]) is None


def test_resolve_rows_keeps_valid_offsets_and_relocates_new_rows():
    rows = [
        {"keep": True, "text": "18:50", "type": "TIME_EXPR", "start": 3, "end": 8},
        {"keep": None, "text": "rolled back", "type": "MITIGATION", "start": None, "end": None},
    ]
    spans, errors = resolve_rows(TEXT, rows)
    assert errors == []
    assert [(s["text"], s["start"]) for s in spans] == [("18:50", 3), ("rolled back", TEXT.index("rolled back"))]


def test_resolve_rows_relocates_when_text_edited():
    rows = [{"keep": True, "text": "API gateway", "type": "SERVICE", "start": 3, "end": 8}]
    spans, errors = resolve_rows(TEXT, rows)
    assert errors == [] and spans[0]["start"] == TEXT.index("API gateway")


def test_resolve_rows_drops_unchecked_and_reports_errors():
    rows = [
        {"keep": False, "text": "18:50", "type": "TIME_EXPR", "start": 3, "end": 8},
        {"keep": True, "text": "nonexistent", "type": "SERVICE", "start": None, "end": None},
        {"keep": True, "text": "outage", "type": None, "start": None, "end": None},
    ]
    spans, errors = resolve_rows(TEXT, rows)
    assert spans == []
    assert len(errors) == 2


def test_find_overlaps():
    a = {"start": 0, "end": 10, "text": "x", "type": "SERVICE"}
    b = {"start": 5, "end": 15, "text": "y", "type": "SERVICE"}
    c = {"start": 15, "end": 20, "text": "z", "type": "SERVICE"}
    assert find_overlaps([a, c]) == []
    assert find_overlaps([a, b, c]) == [(a, b)]


def test_render_escapes_html_and_marks_spans():
    text = "a <b> c"
    out = render_highlighted_html(text, [{"start": 2, "end": 5, "text": "<b>", "type": "SERVICE"}])
    assert "&lt;b&gt;" in out and "<mark" in out


def test_save_gold_chunk_merges_and_load_prefers_gold(tmp_path):
    silver_dir, gold_dir = tmp_path / "silver", tmp_path / "gold"
    silver_dir.mkdir()
    silver_span = {"start": 3, "end": 8, "text": "18:50", "type": "TIME_EXPR"}
    (silver_dir / "INC-1.jsonl").write_text(
        json.dumps({"chunk_id": "c0", "spans": [silver_span]}) + "\n", encoding="utf-8"
    )

    assert load_chunk_spans("INC-1", "c0", silver_dir, gold_dir) == ([silver_span], "silver")
    assert load_chunk_spans("INC-1", "c9", silver_dir, gold_dir) == ([], "empty")

    save_gold_chunk(gold_dir, "INC-1", "c10", [])
    save_gold_chunk(gold_dir, "INC-1", "c2", [silver_span])
    save_gold_chunk(gold_dir, "INC-1", "c2", [])  # ghi đè
    lines = [json.loads(l) for l in (gold_dir / "INC-1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [l["chunk_id"] for l in lines] == ["c2", "c10"]  # sắp theo số, không theo chuỗi
    assert reviewed_chunk_ids("INC-1", gold_dir) == {"c2", "c10"}
    assert load_chunk_spans("INC-1", "c2", silver_dir, gold_dir) == ([], "gold")
    assert load_chunk_spans("INC-1", "c0", silver_dir, gold_dir, force_silver=True)[1] == "silver"


def test_dedupe_spans_removes_exact_duplicates_only():
    a = {"start": 0, "end": 5, "text": "18:50", "type": "TIME_EXPR"}
    conflicting = {"start": 0, "end": 5, "text": "18:50", "type": "METRIC"}
    assert dedupe_spans([a, dict(a), conflicting, dict(a)]) == [a, conflicting]


def test_suggest_keep_flags_prefers_longer_span_and_leaves_no_overlap():
    long_ = {"start": 0, "end": 20, "text": "x", "type": "SYMPTOM"}
    short = {"start": 0, "end": 5, "text": "y", "type": "SERVICE"}
    apart = {"start": 25, "end": 30, "text": "z", "type": "METRIC"}
    tie_a = {"start": 40, "end": 45, "text": "a", "type": "SERVICE"}
    tie_b = {"start": 40, "end": 45, "text": "a", "type": "COMPONENT"}
    spans = [short, long_, apart, tie_a, tie_b]
    flags = suggest_keep_flags(spans)
    assert flags == [False, True, True, True, False]
    assert find_overlaps([s for s, f in zip(spans, flags) if f]) == []
