"""Chuyển nhãn NER mức ký tự (span) sang IOB2 mức token để fine-tune DeBERTa.

Đọc:  data/labeled/ner_<source>/<incident_id>.jsonl  (mặc định source=silver, nhãn nháp của Qwen)
      data/processed/<incident_id>.json               (lấy text của chunk)
Ghi:  data/labeled/ner_iob2/<incident_id>.jsonl       (mỗi dòng một chunk)
      data/labeled/ner_iob2/label_map.json            (label -> id)

Mỗi dòng ghi ra có: incident_id, chunk_id, tokens, offsets, ner_tags (chuỗi IOB2),
input_ids, labels (id nhãn, -100 cho token đặc biệt). Span chồng lấn được làm phẳng
trước (giữ span dài hơn) vì mỗi token chỉ mang được một nhãn.

Chạy: python -m scripts.convert_spans_to_iob2 [--source silver|gold] [--model microsoft/deberta-v3-base]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src.iob2 import LABEL2ID, flatten_spans, misaligned_spans, spans_to_iob2_tags, tags_to_label_ids
from src.labeling import read_jsonl

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/labeled/ner_iob2")


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["silver", "gold"], default="silver")
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    source_dir = Path(f"data/labeled/ner_{args.source}")
    files = sorted(source_dir.glob("INC-*.jsonl"))
    if not files:
        sys.exit(f"Không có file nhãn trong {source_dir}/")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_chunks = n_tokens = n_spans_in = n_spans_used = n_truncated = 0
    tag_counts: Counter = Counter()
    mismatches: list[tuple[str, str, str]] = []
    max_len_seen = 0

    for path in files:
        incident_id = path.stem
        with (PROCESSED_DIR / f"{incident_id}.json").open(encoding="utf-8") as f:
            chunks = {c["chunk_id"]: c["text"] for c in json.load(f)["chunks"]}

        rows = []
        for line in read_jsonl(path):
            text = chunks[line["chunk_id"]]
            spans = flatten_spans(line["spans"])
            n_spans_in += len(line["spans"])
            n_spans_used += len(spans)

            enc = tokenizer(
                text, return_offsets_mapping=True, truncation=True, max_length=args.max_length
            )
            offsets = [tuple(o) for o in enc["offset_mapping"]]
            if enc["input_ids"] and len(enc["input_ids"]) >= args.max_length:
                n_truncated += 1

            tags = spans_to_iob2_tags(offsets, spans)
            tag_counts.update(t for t in tags if t is not None)
            for bad in misaligned_spans(text, offsets, spans):
                mismatches.append((incident_id, line["chunk_id"], bad["text"]))

            rows.append(
                {
                    "incident_id": incident_id,
                    "chunk_id": line["chunk_id"],
                    "tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"]),
                    "offsets": [list(o) for o in offsets],
                    "ner_tags": tags,
                    "input_ids": enc["input_ids"],
                    "labels": tags_to_label_ids(tags),
                }
            )
            n_chunks += 1
            n_tokens += len(offsets)
            max_len_seen = max(max_len_seen, len(offsets))

        with (OUTPUT_DIR / f"{incident_id}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (OUTPUT_DIR / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(LABEL2ID, f, indent=2)

    entity_tokens = sum(c for t, c in tag_counts.items() if t != "O")
    print(f"Nguồn: {source_dir}  ->  {OUTPUT_DIR}/")
    print(f"{len(files)} tài liệu, {n_chunks} chunk, {n_tokens} token (dài nhất {max_len_seen}), bị cắt do max-length: {n_truncated}")
    print(f"Span: {n_spans_in} đầu vào -> {n_spans_used} sau khi làm phẳng (bỏ trùng/chồng lấn)")
    print(f"Token mang nhãn entity: {entity_tokens}/{sum(tag_counts.values())} ({entity_tokens / max(sum(tag_counts.values()), 1):.1%})")
    print(f"Span lệch ranh giới token: {len(mismatches)}/{n_spans_used}")
    for incident_id, chunk_id, span_text in mismatches[:5]:
        print(f"    {incident_id}/{chunk_id}: {span_text!r}")
    print("Số token theo loại nhãn:")
    for tag, count in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {tag:16s} {count}")


if __name__ == "__main__":
    main()
