# Incident Mining Pipeline — Tổng quan cấu trúc dự án

Hệ thống khai thác tri thức sự cố (incident) từ các bài postmortem, đi qua pipeline 5 tầng, lưu thành knowledge graph (Neo4j) và truy vấn theo kiểu GraphRAG (Neo4j + Qdrant + Qwen local qua Ollama).

Trạng thái: **✅ đã xong** · **🟡 đang làm / xong một phần** · **⬜ chưa làm (stub `NotImplementedError`)**

## Luồng dữ liệu tổng thể

```
data/raw/postmortems_dataset.json
  → [Layer 1] làm sạch + chunk            → data/processed/<id>.json
  → chọn mẫu 10 tài liệu                   → data/processed/labeling_sample.json
  → Qwen gán nhãn nháp (silver)            → data/labeled/ner_silver/*.jsonl
  → người review sửa (gold)                → data/labeled/ner_gold/*.jsonl      (chưa có)
  → convert IOB2 → fine-tune DeBERTa       → models/deberta-ner-incident/       (chưa có)
  → [Layer 2] NER → [Layer 3] quan hệ (Qwen) → [Layer 4] validate → [Layer 5] Neo4j + Qdrant + RAG
```

---

## `data/` — Dữ liệu

| Đường dẫn | Chức năng | TT |
|---|---|---|
| `data/raw/postmortems_dataset.json` | Dữ liệu thô: 30 bản ghi `{incident_id, title, source_url, raw_text}` lấy từ GitHub (`fetch_github_postmortems`). 2 bản ghi lỗi fetch (`INC-GH-001`, `INC-GH-021`); nhiều bài bị cắt cứng ở 10000 ký tự. | ✅ |
| `data/processed/INC-GH-*.json` | 28 bản ghi hợp lệ sau Layer 1: `cleaned_text`, `chunks` (kèm `char_start/char_end`), cờ `language`, `truncated`, `log_fragments`. | ✅ |
| `data/processed/dataset_manifest.json` | Thống kê: tổng/hợp lệ/bị loại, phân bố ngôn ngữ, danh sách bài bị cắt cụt. | ✅ |
| `data/processed/labeling_sample.json` | Mẫu 10 tài liệu tiếng Anh (1 tài liệu trung vị cho mỗi domain, 103 chunk); `INC-GH-002` (tiếng Hàn) bị loại. để gán nhãn thủ công. | ✅ |
| `data/labeled/ner_silver/*.jsonl` | Nhãn NER nháp do Qwen sinh, mỗi dòng một chunk: `{chunk_id, spans:[{start,end,text,type}]}`. | 🟡 |
| `data/labeled/ner_gold/*.jsonl` | Nhãn đã review (do `ui/label_review.py` ghi), mỗi dòng một chunk đã lưu. | ⬜ (chờ bạn review) |
| `data/labeled/ner_iob2/*.jsonl`, `label_map.json` | Nhãn IOB2 theo token (`tokens`, `offsets`, `ner_tags`, `input_ids`, `labels`) sinh từ nhãn silver, input trực tiếp cho `train_ner.py`. | ✅ |

## `src/` — Mã nguồn chính

### `src/config.py` ✅
Cấu hình qua `pydantic-settings` (đọc `.env`): URL/model Ollama (`qwen2.5:3b-instruct`), đường dẫn model NER, model embedding, thông tin Neo4j/Qdrant, ngưỡng confidence, số lần retry quan hệ.

### `src/models/` — Schema và taxonomy
| File | Chức năng | TT |
|---|---|---|
| `taxonomy.py` | Nguồn chân lý duy nhất: 10 `ENTITY_TYPES`, 7 `RELATION_TYPES`, ma trận ràng buộc `RELATION_TYPE_CONSTRAINTS`, 21 nhãn `IOB2_LABELS`. | ✅ |
| `schemas.py` | Pydantic `Entity`, `Relation` (dùng `Literal` để tự reject type lạ, có trường `evidence`), `IncidentGraph`. | ✅ |
| `baseline_tfidf_svm.py` | Mô hình baseline TF-IDF + SVM để đối chiếu hiệu năng. | 🟡 (chạy được, chưa dùng) |

### `src/labeling.py` ✅
Logic thuần cho UI review (tách khỏi Streamlit để test): loại span trùng, gợi ý bỏ tick span chồng lấn, định vị lại span theo nguyên văn, kiểm tra chồng lấn, tô màu HTML, đọc/ghi nhãn gold.

### `src/iob2.py` ✅
Logic thuần đổi span mức ký tự thành nhãn IOB2 mức token: làm phẳng span chồng lấn, gán `B-`/`I-`/`O` theo offset token, id nhãn (`-100` cho token đặc biệt), phát hiện span lệch ranh giới token.

### `src/ner_data.py` ✅
Hàm hỗ trợ train NER không phụ thuộc torch: nạp dữ liệu IOB2, chia fold cross-validation theo tài liệu, giải mã nhãn (bỏ `-100`).

### `src/llm/`
| File | Chức năng | TT |
|---|---|---|
| `ollama_client.py` | `OllamaClient`: `generate_text()` và `generate_structured()` (ép JSON Schema qua tham số `format` của Ollama, tự retry kèm lỗi khi output sai). Dùng chung cho gán nhãn nháp, trích quan hệ (Layer 3), sinh câu trả lời RAG (Layer 5). | ✅ |

### `src/pipeline/` — 5 tầng xử lý
| File | Chức năng | TT |
|---|---|---|
| `layer1_preprocessing.py` | Lọc bản ghi fetch lỗi, chuẩn hóa NFKC, xoá boilerplate HTML, nhận diện ngôn ngữ, cờ cắt cụt, chia chunk theo câu (`pysbd`) có offset, bóc timestamp/mã HTTP/token lỗi. | ✅ |
| `layer2_ner.py` | NER domain bằng DeBERTa đã fine-tune + rule-based (`rules_ner.py` dự kiến). | ⬜ |
| `layer3_llm_relation.py` | Qwen suy luận quan hệ (`CAUSES`, `MITIGATED_BY`...) giữa các entity đã trích. | ⬜ |
| `layer4_validation.py` | Kiểm tra logic: endpoint tồn tại, không self-loop, ràng buộc kiểu quan hệ, dedup, vòng retry. | ⬜ |
| `layer5_graph_rag.py` | Ghi cây sự cố vào Neo4j (mỗi incident một cây riêng), embed vào Qdrant, truy vấn GraphRAG. | ⬜ |
| `orchestrator.py` | Điều phối chạy tuần tự 5 tầng. | ⬜ (khung) |

### `src/api/` — Backend FastAPI
| File | Chức năng | TT |
|---|---|---|
| `main.py` | Khởi tạo FastAPI app, gắn router `/api/v1`. | ✅ (khung) |
| `endpoints.py` | Route `POST /api/v1/analyze` (chạy pipeline) và `POST /api/v1/search` (truy vấn RAG). | ⬜ (phụ thuộc các layer) |

## `scripts/` — Script chạy pipeline (chạy: `python -m scripts.<tên>` tại thư mục gốc dự án)

| File | Chức năng | TT |
|---|---|---|
| `prepare_data.py` | Chạy Layer 1 trên toàn bộ dataset thô → `data/processed/*.json` + manifest. | ✅ |
| `select_labeling_sample.py` | Bóc URL gốc trong link `web.archive.org`, gom theo domain, chọn 1 tài liệu trung vị/domain, bỏ tài liệu không phải tiếng Anh → `labeling_sample.json`. Tránh việc Cloudflare (15/28 bài) áp đảo mẫu. | ✅ |
| `convert_spans_to_iob2.py` | Tokenize bằng tokenizer DeBERTa-v3, đổi nhãn span (mặc định lấy silver, `--source gold` nếu có) sang IOB2 theo token → `data/labeled/ner_iob2/`, in thống kê chất lượng chuyển đổi. | ✅ |
| `train_ner.py` | Fine-tune `microsoft/deberta-v3-base` (21 nhãn IOB2): cross-validation 4 fold theo tài liệu → chọn số epoch tốt nhất → train lại trên toàn bộ dữ liệu → lưu `models/deberta-ner-incident/` + `eval_metrics.json` (F1 seqeval strict, theo fold và theo loại entity). | 🟡 (đã viết, chưa chạy full) |
| `silver_label_ner.py` | Gọi Qwen (few-shot, JSON Schema) gán nhãn nháp cho 103 chunk của mẫu; tự định vị offset bằng tìm chuỗi thay vì tin offset do LLM tính; bỏ qua tài liệu đã có kết quả. | ✅ (đã chạy xong 10/10 tài liệu) |

## `ui/`
| File | Chức năng | TT |
|---|---|---|
| `app.py` | Dashboard Streamlit: nhập log → xem kết quả phân tích. | ⬜ (khung) |
| `label_review.py` | Giao diện review nhãn silver → gold (`streamlit run ui/label_review.py`): tô màu span theo loại, bảng sửa/thêm/xoá span, chặn lưu khi span chồng lấn, tự loại span trùng và tự bỏ tick span ngắn chồng lên span dài, tiến độ theo chunk. | ✅ |

## `tests/` — Unit test (`python -m pytest tests/`)
| File | Kiểm tra | TT |
|---|---|---|
| `test_layer1.py` | Lọc fetch lỗi, biên cắt cụt, nhận diện ngôn ngữ, xoá boilerplate, offset chunk, parse log (12 test). | ✅ |
| `test_llm_client.py` | `OllamaClient` với `httpx` được mock: thành công lần đầu, retry khi JSON lỗi/timeout, giới hạn output, hết retry thì raise (7 test). | ✅ |
| `test_iob2.py` | Gán nhãn B/I/O theo token, token đặc biệt, làm phẳng span, span lệch ranh giới (6 test). | ✅ |
| `test_ner_data.py` | Chia fold theo tài liệu, giải mã nhãn, nạp IOB2 (4 test). | ✅ |
| `test_labeling.py` | Logic UI review: dedupe, gợi ý tick, định vị span, chồng lấn, lưu/đọc gold (11 test). | ✅ |
| `test_layer4_validation.py` | Hiện chỉ là stub (`NotImplementedError`). | ⬜ |

## Hạ tầng và cấu hình gốc

| File | Chức năng |
|---|---|
| `docker-compose.yml` | Dựng `api`, `ollama` (có cấp GPU NVIDIA), `neo4j`, `qdrant`. |
| `Dockerfile` | Đóng gói API Python (`uvicorn src.api.main:app`). |
| `requirements.txt` | Danh sách thư viện Python. |
| `.env.example` | Mẫu biến môi trường (Ollama, Neo4j, Qdrant, ngưỡng). |
| `.gitignore` | Bỏ `__pycache__/`, `.env`, `venv/`, `models/`, `.pytest_cache/`. |
| `README.md` | Hướng dẫn setup và chạy. |

## Thư mục sinh ra khi chạy (không nên commit)
- `models/deberta-ner-incident/` — model NER sau fine-tune (đã có trong `.gitignore`).
- `__pycache__/`, `.pytest_cache/` — cache Python/pytest.

## Lệnh thường dùng

```bash
python -m scripts.prepare_data            # Layer 1: làm sạch + chunk
python -m scripts.select_labeling_sample  # chọn mẫu gán nhãn
docker compose up -d ollama               # bật Ollama (dùng GPU)
python -m scripts.silver_label_ner        # Qwen gán nhãn nháp
docker compose stop ollama                # nhường VRAM cho việc train
python -m scripts.convert_spans_to_iob2   # nhãn span -> IOB2
python -m scripts.train_ner               # fine-tune DeBERTa
python -m pytest tests/                   # chạy test
```

## Các bước tiếp theo (theo kế hoạch)
1. ~~Review nhãn~~ — đã quyết định train thẳng trên nhãn silver (UI `label_review.py` vẫn dùng được nếu sau này muốn chuẩn hoá).
2. ✅ `convert_spans_to_iob2.py` → `scripts/train_ner.py` (đã viết; chạy: `docker compose stop ollama` rồi `python -m scripts.train_ner`) (DeBERTa-v3, cross-validation theo tài liệu).
3. `src/pipeline/rules_ner.py` + hoàn thiện `layer2_ner.py`.
4. Layer 3 (quan hệ) → Layer 4 (validate) → Layer 5 (Neo4j + Qdrant + RAG) → script batch.
