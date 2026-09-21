"""Logic thuần cho UI review nhãn NER (silver -> gold), tách khỏi Streamlit để test được."""

import html
import json
import os
from pathlib import Path

from src.models.taxonomy import ENTITY_TYPES

# Mỗi entity type một màu (hue cách đều), alpha thấp để đọc được cả nền sáng lẫn tối.
TYPE_COLORS = {t: f"hsla({i * 36}, 70%, 50%, 0.35)" for i, t in enumerate(ENTITY_TYPES)}


def is_missing(value) -> bool:
    """True với None, NaN và pd.NA (ô trống trong bảng st.data_editor)."""
    if value is None:
        return True
    try:
        return bool(value != value)  # NaN != NaN
    except TypeError:  # pd.NA: so sánh trả NA, bool(NA) raise TypeError
        return True


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedupe_spans(spans: list[dict]) -> list[dict]:
    """Bỏ span trùng hệt (cùng vị trí, cùng loại): Qwen hay liệt kê một entity nhiều
    lần và script silver luôn định vị về lần xuất hiện đầu tiên."""
    seen: set[tuple[int, int, str]] = set()
    unique = []
    for span in spans:
        key = (span["start"], span["end"], span["type"])
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return unique


def suggest_keep_flags(spans: list[dict]) -> list[bool]:
    """Gợi ý tick sẵn để không còn span chồng lấn (NER phẳng): ưu tiên span dài
    hơn, bằng nhau thì span đứng trước. Chỉ là mặc định, người review đổi lại được."""
    order = sorted(range(len(spans)), key=lambda i: (-(spans[i]["end"] - spans[i]["start"]), i))
    kept: list[int] = []
    flags = [False] * len(spans)
    for i in order:
        s = spans[i]
        if all(s["end"] <= spans[j]["start"] or s["start"] >= spans[j]["end"] for j in kept):
            kept.append(i)
            flags[i] = True
    return flags


def load_chunk_spans(
    incident_id: str, chunk_id: str, silver_dir: Path, gold_dir: Path, force_silver: bool = False
) -> tuple[list[dict], str]:
    """Trả (spans, nguồn) với nguồn là "gold" | "silver" | "empty". Ưu tiên gold
    đã lưu, trừ khi force_silver (người dùng bấm khôi phục nhãn gốc)."""
    if not force_silver:
        for line in read_jsonl(gold_dir / f"{incident_id}.jsonl"):
            if line["chunk_id"] == chunk_id:
                return line["spans"], "gold"
    for line in read_jsonl(silver_dir / f"{incident_id}.jsonl"):
        if line["chunk_id"] == chunk_id:
            return dedupe_spans(line["spans"]), "silver"
    return [], "empty"


def reviewed_chunk_ids(incident_id: str, gold_dir: Path) -> set[str]:
    return {line["chunk_id"] for line in read_jsonl(gold_dir / f"{incident_id}.jsonl")}


def locate_span(text: str, entity_text: str, taken: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Tìm entity_text (khớp chính xác) trong text. Nếu xuất hiện nhiều lần, ưu
    tiên lần chưa bị span khác chiếm để có thể gắn nhãn nhiều lần cùng một cụm."""
    positions = []
    idx = text.find(entity_text)
    while idx != -1:
        positions.append(idx)
        idx = text.find(entity_text, idx + 1)
    if not positions:
        return None
    for pos in positions:
        end = pos + len(entity_text)
        if all(end <= s or pos >= e for s, e in taken):
            return pos, end
    return positions[0], positions[0] + len(entity_text)


def resolve_rows(chunk_text: str, rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Chuyển các dòng trong bảng chỉnh sửa thành span có offset chính xác.

    Dòng có start/end còn khớp với text thì giữ nguyên; dòng mới hoặc dòng bị
    sửa text thì tự định vị lại bằng str.find. Trả (spans, danh sách lỗi)."""
    spans: list[dict] = []
    errors: list[str] = []
    pending: list[tuple[int, str, str]] = []

    for i, row in enumerate(rows, start=1):
        text = row.get("text")
        if is_missing(text) or not str(text).strip():
            continue
        keep = row.get("keep")
        if not is_missing(keep) and not keep:
            continue
        text = str(text).strip()
        entity_type = row.get("type")
        if is_missing(entity_type) or entity_type not in ENTITY_TYPES:
            errors.append(f"Dòng {i} ({text!r}): chưa chọn loại entity hợp lệ.")
            continue
        start, end = row.get("start"), row.get("end")
        if not is_missing(start) and not is_missing(end):
            start, end = int(start), int(end)
            if chunk_text[start:end] == text:
                spans.append({"start": start, "end": end, "text": text, "type": entity_type})
                continue
        pending.append((i, text, entity_type))

    for i, text, entity_type in pending:
        located = locate_span(chunk_text, text, [(s["start"], s["end"]) for s in spans])
        if located is None:
            errors.append(f"Dòng {i}: không tìm thấy nguyên văn {text!r} trong chunk.")
            continue
        spans.append({"start": located[0], "end": located[1], "text": text, "type": entity_type})

    spans.sort(key=lambda s: (s["start"], s["end"]))
    return spans, errors


def find_overlaps(spans: list[dict]) -> list[tuple[dict, dict]]:
    """IOB2 mỗi token chỉ có 1 nhãn nên các span không được chồng lấn nhau."""
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    overlaps = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b["start"] >= a["end"]:
                break
            overlaps.append((a, b))
    return overlaps


def render_highlighted_html(text: str, spans: list[dict]) -> str:
    parts = []
    cursor = 0
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        if span["start"] < cursor:
            continue  # span chồng lấn: đã bị báo lỗi riêng, không vẽ đè
        parts.append(html.escape(text[cursor:span["start"]]))
        color = TYPE_COLORS.get(span["type"], "transparent")
        parts.append(
            f'<mark style="background:{color};color:inherit;padding:1px 3px;border-radius:3px" '
            f'title="{span["type"]}">{html.escape(text[span["start"]:span["end"]])}'
            f'<sup style="font-size:0.65em;opacity:0.8"> {span["type"]}</sup></mark>'
        )
        cursor = span["end"]
    parts.append(html.escape(text[cursor:]))
    return (
        '<div style="white-space:pre-wrap;line-height:1.9;font-size:0.95rem">'
        + "".join(parts)
        + "</div>"
    )


def _chunk_order(chunk_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in chunk_id if ch.isdigit())
    return (int(digits) if digits else 0, chunk_id)


def save_gold_chunk(gold_dir: Path, incident_id: str, chunk_id: str, spans: list[dict]) -> None:
    """Ghi (hoặc ghi đè) nhãn gold của một chunk vào file của incident, giữ nguyên các chunk khác."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    path = gold_dir / f"{incident_id}.jsonl"
    by_chunk = {line["chunk_id"]: line for line in read_jsonl(path)}
    by_chunk[chunk_id] = {"chunk_id": chunk_id, "spans": spans}

    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for cid in sorted(by_chunk, key=_chunk_order):
            f.write(json.dumps(by_chunk[cid], ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
