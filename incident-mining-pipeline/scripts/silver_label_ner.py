"""Sinh nhãn NER nháp (silver) bằng Qwen cho các chunk trong mẫu đã chọn.

Chỉ chạy trên các incident liệt kê trong data/processed/labeling_sample.json
(~11 tài liệu / ~108 chunk), không chạy trên toàn bộ 28 tài liệu -- phần còn
lại sẽ được model NER đã fine-tune xử lý sau (Layer 2), không cần Qwen.

Yêu cầu: Ollama đang chạy (docker compose up -d ollama) và đã pull model
(docker compose exec ollama ollama pull qwen2.5:3b-instruct).

Chạy: python -m scripts.silver_label_ner
"""

import json
import sys
from pathlib import Path

from src.config import settings
from src.llm.ollama_client import LLMExtractionError, OllamaClient
from src.models.taxonomy import ENTITY_TYPES

PROCESSED_DIR = Path("data/processed")
SAMPLE_PATH = PROCESSED_DIR / "labeling_sample.json"
OUTPUT_DIR = Path("data/labeled/ner_silver")

ENTITY_TYPE_DESCRIPTIONS = {
    "SERVICE": "tên dịch vụ/hệ thống cấp cao, vd 'API gateway', 'EC2', 'CDN'",
    "COMPONENT": "thành phần kỹ thuật cụ thể hơn SERVICE, vd 'distribution database', 'load balancer'",
    "ERROR_CODE": "mã lỗi cụ thể, vd '503', 'ETIMEDOUT', 'ECONNRESET'",
    "METRIC": "số liệu định lượng, vd '20 minutes', '15% of requests', '3x traffic'",
    "TIME_EXPR": "mốc thời gian, vd '18:50 UTC', '2021-11-08'",
    "ROOT_CAUSE": "nguyên nhân gốc gây ra sự cố, vd 'a deploy changed a field type'",
    "SYMPTOM": "biểu hiện/triệu chứng quan sát được, vd 'requests started failing'",
    "IMPACT": "tác động tới người dùng/hệ thống, vd 'most customer requests', 'data loss'",
    "MITIGATION": "hành động khắc phục, vd 'rolled back the change', 'scaled up the cluster'",
    "TEAM": "đội ngũ/nhóm phụ trách, vd 'on-call SRE team', 'infrastructure team'",
}

FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "At 18:50, we rolled out a deploy to change a field type in our "
            "distribution database, which caused distribution to fail."
        ),
        "entities": [
            {"text": "18:50", "type": "TIME_EXPR"},
            {"text": "distribution database", "type": "COMPONENT"},
            {"text": "a deploy to change a field type", "type": "ROOT_CAUSE"},
            {"text": "distribution to fail", "type": "SYMPTOM"},
        ],
    },
    {
        "text": (
            "The API gateway returned 503 errors for approximately 20 minutes, "
            "affecting most customer requests."
        ),
        "entities": [
            {"text": "API gateway", "type": "SERVICE"},
            {"text": "503", "type": "ERROR_CODE"},
            {"text": "20 minutes", "type": "METRIC"},
            {"text": "most customer requests", "type": "IMPACT"},
        ],
    },
    {
        "text": (
            "The on-call SRE team rolled back the change, which mitigated the "
            "outage within 5 minutes."
        ),
        "entities": [
            {"text": "on-call SRE team", "type": "TEAM"},
            {"text": "rolled back the change", "type": "MITIGATION"},
            {"text": "outage", "type": "SYMPTOM"},
            {"text": "5 minutes", "type": "METRIC"},
        ],
    },
]

SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                },
                "required": ["text", "type"],
            },
        }
    },
    "required": ["entities"],
}


def build_prompt(chunk_text: str) -> str:
    type_lines = "\n".join(f"- {t}: {ENTITY_TYPE_DESCRIPTIONS[t]}" for t in ENTITY_TYPES)
    example_lines = []
    for ex in FEW_SHOT_EXAMPLES:
        example_lines.append(f'Text: "{ex["text"]}"')
        example_lines.append(f"Entities: {json.dumps({'entities': ex['entities']})}")
    examples_block = "\n".join(example_lines)

    return f"""You are annotating incident postmortem text with named entities.
Entity types:
{type_lines}

Rules:
- "text" must be copied EXACTLY (verbatim, same casing) from the input text -- it must be a literal substring.
- Only extract entities that are clearly present. Do not invent information.
- Return JSON only, matching the schema: {{"entities": [{{"text": ..., "type": ...}}]}}

Examples:
{examples_block}

Now extract entities from this text:
Text: "{chunk_text}"
"""


def locate_span(chunk_text: str, entity_text: str) -> tuple[int, int] | None:
    """Tìm vị trí char của entity_text trong chunk_text. Model được yêu cầu
    trả text nguyên văn thay vì tự tính offset (LLM tính offset hay sai),
    nên bước này tự định vị lại bằng str.find, khớp chính xác trước rồi mới
    thử không phân biệt hoa/thường."""
    idx = chunk_text.find(entity_text)
    if idx != -1:
        return idx, idx + len(entity_text)
    idx = chunk_text.lower().find(entity_text.lower())
    if idx != -1:
        return idx, idx + len(entity_text)
    return None


def label_chunk(client: OllamaClient, chunk_text: str) -> list[dict]:
    prompt = build_prompt(chunk_text)
    try:
        result = client.generate_structured(prompt, schema=SCHEMA, max_retries=3)
    except LLMExtractionError as exc:
        print(f"  [WARN] extraction failed for chunk: {exc}")
        return []

    spans = []
    for entity in result.get("entities", []):
        text = entity.get("text", "")
        entity_type = entity.get("type", "")
        if entity_type not in ENTITY_TYPES or not text:
            continue
        located = locate_span(chunk_text, text)
        if located is None:
            print(f"  [WARN] could not locate span in text: {text!r}")
            continue
        start, end = located
        spans.append({"start": start, "end": end, "text": chunk_text[start:end], "type": entity_type})
    return spans


def main() -> None:
    # Console Windows mặc định cp1252 không in được ký tự non-ASCII (vd tiếng Hàn
    # trong INC-GH-002) -> đổi errors="replace" để print cảnh báo không làm sập script.
    sys.stdout.reconfigure(errors="replace")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SAMPLE_PATH.open(encoding="utf-8") as f:
        sample = json.load(f)["sample"]

    client = OllamaClient(base_url=settings.ollama_base_url, model=settings.llm_model_name)

    total_chunks = 0
    total_spans = 0
    for entry in sample:
        incident_id = entry["incident_id"]
        out_path = OUTPUT_DIR / f"{incident_id}.jsonl"
        if out_path.exists():
            print(f"Skip {incident_id} (đã có {out_path}; xoá file này nếu muốn gán nhãn lại)")
            continue
        with (PROCESSED_DIR / f"{incident_id}.json").open(encoding="utf-8") as f:
            processed = json.load(f)

        print(f"Labeling {incident_id} ({len(processed['chunks'])} chunks)...")
        lines = []
        for chunk in processed["chunks"]:
            spans = label_chunk(client, chunk["text"])
            lines.append({"chunk_id": chunk["chunk_id"], "spans": spans})
            total_chunks += 1
            total_spans += len(spans)

        with out_path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Done. Labeled {total_chunks} chunks, {total_spans} spans total -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
