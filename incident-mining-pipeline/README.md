# Incident Mining Pipeline

Hệ thống khai thác tri thức sự cố (incident) từ postmortem và log, qua pipeline 5 tầng xử lý, lưu trữ dưới dạng knowledge graph và phục vụ truy vấn RAG.

## Kiến trúc

- `data/` — dữ liệu thô và dữ liệu đã xử lý.
- `src/pipeline/` — 5 tầng xử lý: preprocessing, NER, LLM relation extraction, validation, graph/RAG.
- `src/models/` — baseline model và Pydantic schemas.
- `src/api/` — FastAPI backend.
- `ui/` — dashboard demo bằng Streamlit.
- `tests/` — unit test cho từng tầng.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # điền API keys, DB URLs
```

## Chạy với Docker Compose

```bash
docker-compose up --build
```

Dịch vụ API sẽ chạy tại `http://localhost:8000`, Neo4j browser tại `http://localhost:7474`.

## Chạy API độc lập

```bash
uvicorn src.api.main:app --reload
```

## Chạy UI demo

```bash
streamlit run ui/app.py
```

## Chạy tests

```bash
pytest tests/
```
