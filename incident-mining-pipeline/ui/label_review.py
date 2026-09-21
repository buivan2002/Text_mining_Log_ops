"""UI review nhãn NER: sửa nhãn nháp của Qwen (silver) thành nhãn chuẩn (gold).

Chạy (tại thư mục gốc dự án): streamlit run ui/label_review.py
Đọc:  data/processed/labeling_sample.json, data/processed/<id>.json, data/labeled/ner_silver/<id>.jsonl
Ghi:  data/labeled/ner_gold/<id>.jsonl (mỗi dòng một chunk đã review)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.labeling import (
    TYPE_COLORS,
    find_overlaps,
    load_chunk_spans,
    render_highlighted_html,
    resolve_rows,
    reviewed_chunk_ids,
    save_gold_chunk,
    suggest_keep_flags,
)
from src.models.taxonomy import ENTITY_TYPES

PROCESSED_DIR = Path("data/processed")
SILVER_DIR = Path("data/labeled/ner_silver")
GOLD_DIR = Path("data/labeled/ner_gold")

st.set_page_config(page_title="Review nhãn NER", layout="wide")


@st.cache_data
def load_sample() -> list[dict]:
    with (PROCESSED_DIR / "labeling_sample.json").open(encoding="utf-8") as f:
        return json.load(f)["sample"]


@st.cache_data
def load_incident(incident_id: str) -> dict:
    with (PROCESSED_DIR / f"{incident_id}.json").open(encoding="utf-8") as f:
        return json.load(f)


def spans_to_frame(spans: list[dict], keep_flags: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "keep": keep_flags,
            "text": [s["text"] for s in spans],
            "type": [s["type"] for s in spans],
            "start": pd.array([s["start"] for s in spans], dtype="Int64"),
            "end": pd.array([s["end"] for s in spans], dtype="Int64"),
        }
    )


sample = load_sample()

# Áp dụng điều hướng đã đặt ở lượt chạy trước, trước khi widget chọn chunk được tạo.
if "goto" in st.session_state:
    goto_key, goto_value = st.session_state.pop("goto")
    st.session_state[goto_key] = goto_value

# ---------- Sidebar: chọn incident + tiến độ ----------
progress = {}
for entry in sample:
    incident = load_incident(entry["incident_id"])
    done = reviewed_chunk_ids(entry["incident_id"], GOLD_DIR)
    progress[entry["incident_id"]] = (len(done & {c["chunk_id"] for c in incident["chunks"]}), len(incident["chunks"]))

total_done = sum(d for d, _ in progress.values())
total_all = sum(n for _, n in progress.values())

with st.sidebar:
    st.header("Review nhãn NER")
    st.progress(total_done / total_all if total_all else 0.0, text=f"Đã review {total_done}/{total_all} chunk")
    # key cố định: nhãn hiển thị chứa tiến độ (đổi sau mỗi lần lưu), không có key thì
    # Streamlit coi là widget mới và nhảy về tài liệu đầu tiên.
    incident_id = st.selectbox(
        "Tài liệu",
        [e["incident_id"] for e in sample],
        key="incident_select",
        format_func=lambda i: f"{i} · {next(e['title'] for e in sample if e['incident_id'] == i)} "
        f"({progress[i][0]}/{progress[i][1]})",
    )
    st.markdown("**Chú giải**")
    st.markdown(
        " ".join(
            f'<span style="background:{TYPE_COLORS[t]};padding:1px 6px;border-radius:3px;'
            f'font-size:0.8rem;display:inline-block;margin:2px 0">{t}</span>'
            for t in ENTITY_TYPES
        ),
        unsafe_allow_html=True,
    )

incident = load_incident(incident_id)
chunks = incident["chunks"]
reviewed = reviewed_chunk_ids(incident_id, GOLD_DIR)

idx_key = f"chunk_idx::{incident_id}"
if idx_key not in st.session_state:
    first_todo = next((i for i, c in enumerate(chunks) if c["chunk_id"] not in reviewed), 0)
    st.session_state[idx_key] = first_todo

st.subheader(f"{incident_id} · {incident['title']}")
if incident.get("language") == "non-en":
    st.warning("Tài liệu không phải tiếng Anh; nhãn từ model tiếng Anh có thể kém chính xác.")
if incident.get("truncated"):
    st.caption("Tài liệu bị cắt cụt ở 10000 ký tự nên chunk cuối có thể đứt giữa câu.")

chunk_idx = st.selectbox(
    "Chunk",
    range(len(chunks)),
    key=idx_key,
    format_func=lambda i: f"{chunks[i]['chunk_id']} {'✅ đã review' if chunks[i]['chunk_id'] in reviewed else '⬜ chưa review'}",
)
chunk = chunks[chunk_idx]
chunk_id, chunk_text = chunk["chunk_id"], chunk["text"]

version_key = f"version::{incident_id}::{chunk_id}"
version = st.session_state.get(version_key, 0)
force_silver = st.session_state.get(f"force_silver::{incident_id}::{chunk_id}", False)
spans, source = load_chunk_spans(incident_id, chunk_id, SILVER_DIR, GOLD_DIR, force_silver=force_silver)

source_label = {"gold": "nhãn gold đã lưu", "silver": "nhãn nháp của Qwen (chưa review)", "empty": "chưa có nhãn nháp"}[source]
keep_flags = suggest_keep_flags(spans) if source == "silver" else [True] * len(spans)
n_auto_unticked = keep_flags.count(False)
st.caption(
    f"Đang hiển thị: {source_label}"
    + (f" · đã tự bỏ tick {n_auto_unticked} span ngắn chồng lên span dài hơn (tick lại nếu span ngắn mới đúng)" if n_auto_unticked else "")
)

left, right = st.columns([3, 2], gap="large")

with right:
    st.markdown("**Danh sách span** (bỏ tick để xoá, sửa `type`, thêm dòng mới ở cuối bảng)")
    edited = st.data_editor(
        spans_to_frame(spans, keep_flags),
        key=f"editor::{incident_id}::{chunk_id}::{version}",
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "keep": st.column_config.CheckboxColumn("giữ", default=True, width="small"),
            "text": st.column_config.TextColumn("text (nguyên văn trong chunk)", required=True),
            "type": st.column_config.SelectboxColumn("type", options=ENTITY_TYPES, required=True),
            "start": st.column_config.NumberColumn("start", disabled=True, width="small"),
            "end": st.column_config.NumberColumn("end", disabled=True, width="small"),
        },
    )

final_spans, errors = resolve_rows(chunk_text, edited.to_dict("records"))
overlaps = find_overlaps(final_spans)

with left:
    st.markdown("**Chunk** (màu = loại entity theo chú giải)")
    st.markdown(render_highlighted_html(chunk_text, final_spans), unsafe_allow_html=True)

with right:
    for message in errors:
        st.error(message)
    for a, b in overlaps:
        st.error(f"Span chồng lấn: {a['text']!r} ({a['type']}) và {b['text']!r} ({b['type']}). Hãy bỏ tick một trong hai.")
    st.caption(f"{len(final_spans)} span sẽ được lưu")

    def save_current() -> None:
        save_gold_chunk(GOLD_DIR, incident_id, chunk_id, final_spans)
        st.session_state.pop(f"force_silver::{incident_id}::{chunk_id}", None)

    can_save = not errors and not overlaps
    save_col, next_col = st.columns(2)
    if save_col.button("Lưu", disabled=not can_save, width="stretch"):
        save_current()
        st.toast(f"Đã lưu {incident_id}/{chunk_id}")
        st.session_state["goto"] = (idx_key, chunk_idx)
        st.rerun()
    if next_col.button("Lưu & chunk tiếp", type="primary", disabled=not can_save, width="stretch"):
        save_current()
        st.session_state["goto"] = (idx_key, min(chunk_idx + 1, len(chunks) - 1))
        st.rerun()

    prev_col, skip_col = st.columns(2)
    if prev_col.button("← Chunk trước (không lưu)", disabled=chunk_idx == 0, width="stretch"):
        st.session_state["goto"] = (idx_key, chunk_idx - 1)
        st.rerun()
    if skip_col.button("Chunk sau → (không lưu)", disabled=chunk_idx == len(chunks) - 1, width="stretch"):
        st.session_state["goto"] = (idx_key, chunk_idx + 1)
        st.rerun()

    if source == "gold" and st.button("Khôi phục nhãn nháp gốc của Qwen", width="stretch"):
        st.session_state[f"force_silver::{incident_id}::{chunk_id}"] = True
        st.session_state[version_key] = version + 1
        st.rerun()
