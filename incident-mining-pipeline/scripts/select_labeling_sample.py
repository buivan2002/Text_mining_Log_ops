"""Chọn mẫu tài liệu đại diện để gán nhãn thủ công (chuẩn bị cho fine-tune NER).

Dataset lệch nặng về Cloudflare (15/28 sau khi unwrap link web.archive.org
về đúng domain gốc) -- nếu chọn ngẫu nhiên/lấy N tài liệu đầu, mẫu dễ dính
phần lớn văn phong Cloudflare, model học lệch. Script này chọn đúng 1 tài
liệu đại diện (độ dài trung vị) cho mỗi domain, nên số tài liệu trong mẫu =
số domain khác nhau trong dataset. Tài liệu không phải tiếng Anh (cờ `language` từ
Layer 1) bị loại vì model NER tiếng Anh và prompt tiếng Anh không xử lý tốt.

Chạy: python -m scripts.select_labeling_sample
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "labeling_sample.json"

# Nhiều URL trong dataset là link web.archive.org bọc quanh URL gốc
# (vd .../web/20211006135542/https://blog.cloudflare.com/...) -- phải bóc
# URL gốc bên trong mới biết đúng domain/công ty nào viết bài.
_URL_RE = re.compile(r"https?://")


def real_domain(source_url: str) -> str:
    matches = list(_URL_RE.finditer(source_url))
    if len(matches) > 1:
        source_url = source_url[matches[-1].start():]
    return urlparse(source_url).netloc.replace("www.", "")


def load_incidents() -> list[dict]:
    incidents = []
    for path in sorted(PROCESSED_DIR.glob("INC-*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        incidents.append(
            {
                "incident_id": data["incident_id"],
                "title": data["title"],
                "source_url": data["source_url"],
                "num_chunks": len(data["chunks"]),
                "language": data.get("language", "en"),
                "domain": real_domain(data["source_url"]),
            }
        )
    return incidents


def select_sample(incidents: list[dict]) -> list[dict]:
    by_domain: dict[str, list[dict]] = {}
    for inc in incidents:
        by_domain.setdefault(inc["domain"], []).append(inc)

    selected = []
    for domain in sorted(by_domain):
        candidates = sorted(by_domain[domain], key=lambda x: x["num_chunks"])
        pick = candidates[len(candidates) // 2]  # độ dài trung vị trong domain này
        selected.append(
            {
                "incident_id": pick["incident_id"],
                "title": pick["title"],
                "domain": domain,
                "num_chunks": pick["num_chunks"],
                "reason": f"representative (median length) of {len(candidates)} doc(s) from {domain}",
            }
        )
    return selected


def main() -> None:
    all_incidents = load_incidents()
    incidents = [inc for inc in all_incidents if inc["language"] == "en"]
    excluded = [inc["incident_id"] for inc in all_incidents if inc["language"] != "en"]
    sample = select_sample(incidents)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_incidents": len(all_incidents),
                "excluded_non_english": excluded,
                "total_domains": len(sample),
                "sample_size": len(sample),
                "sample": sample,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    total_chunks = sum(s["num_chunks"] for s in sample)
    print(f"Excluded non-English: {excluded}")
    print(f"Selected {len(sample)} incidents across {len(sample)} domains ({total_chunks} chunks total)")
    for s in sample:
        print(f"  {s['incident_id']} | {s['domain']} | {s['num_chunks']} chunks | {s['title']}")


if __name__ == "__main__":
    main()
