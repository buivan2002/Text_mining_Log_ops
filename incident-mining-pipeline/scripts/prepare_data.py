"""Chạy Layer 1 trên toàn bộ postmortems_dataset.json.

Đọc data/raw/postmortems_dataset.json, lọc bản ghi fetch lỗi, làm sạch +
chunk từng bản ghi hợp lệ, ghi ra data/processed/<incident_id>.json và một
manifest tổng hợp data/processed/dataset_manifest.json.

Chạy: python -m scripts.prepare_data
"""

import json
from pathlib import Path

from src.pipeline import layer1_preprocessing as layer1

RAW_PATH = Path("data/raw/postmortems_dataset.json")
PROCESSED_DIR = Path("data/processed")


def prepare_incident(record: dict) -> dict:
    raw_text = record["raw_text"]
    cleaned_text = layer1.preprocess(raw_text)
    return {
        "incident_id": record["incident_id"],
        "title": record["title"],
        "source_url": record["source_url"],
        "status": "ok",
        "language": layer1.detect_language(raw_text),
        "truncated": layer1.is_truncated(raw_text),
        "cleaned_text": cleaned_text,
        "log_fragments": layer1.parse_log_lines(raw_text),
        "chunks": layer1.chunk_text(cleaned_text),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with RAW_PATH.open(encoding="utf-8") as f:
        records = json.load(f)

    valid_ids: list[str] = []
    skipped_ids: list[str] = []
    language_counts: dict[str, int] = {}
    truncated_ids: list[str] = []

    for record in records:
        incident_id = record["incident_id"]
        if not layer1.is_valid_fetch(record.get("raw_text", "")):
            skipped_ids.append(incident_id)
            continue

        processed = prepare_incident(record)
        valid_ids.append(incident_id)
        language_counts[processed["language"]] = language_counts.get(processed["language"], 0) + 1
        if processed["truncated"]:
            truncated_ids.append(incident_id)

        out_path = PROCESSED_DIR / f"{incident_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

    manifest = {
        "total": len(records),
        "valid": len(valid_ids),
        "skipped": skipped_ids,
        "language_distribution": language_counts,
        "truncated_ids": truncated_ids,
    }
    with (PROCESSED_DIR / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(valid_ids)}/{len(records)} incidents, skipped {skipped_ids}")
    print(f"Truncated: {truncated_ids}")
    print(f"Language distribution: {language_counts}")


if __name__ == "__main__":
    main()
