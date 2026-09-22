"""Tầng 5: Neo4j knowledge graph + Qdrant vector search cho GraphRAG.

Mỗi incident là MỘT CÂY RIÊNG (Incident -[HAS_ENTITY]-> Entity -[quan hệ]->
Entity): entity node dùng khóa `global_id = f"{incident_id}:{entity.id}"` để
không vô tình gộp entity trùng tên giữa các incident khác nhau -- cross-incident
entity resolution chưa làm, để dành pha sau.
"""

import re
import uuid
from functools import lru_cache

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.models.schemas import IncidentGraph
from src.models.taxonomy import RELATION_TYPES

# Neo4j không cho parameterize tên relationship trong Cypher (phải interpolate
# thẳng chuỗi), nên phải chắc chắn RELATION_TYPES chỉ chứa định danh an toàn
# trước khi dùng f-string -- chặn injection nếu sau này taxonomy đổi.
assert all(re.fullmatch(r"[A-Z_][A-Z0-9_]*", t) for t in RELATION_TYPES)

CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT incident_id_unique IF NOT EXISTS FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE",
    "CREATE CONSTRAINT entity_global_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.global_id IS UNIQUE",
]

RAG_PROMPT_TEMPLATE = """Bạn là trợ lý trả lời câu hỏi dựa trên tri thức về các sự cố kỹ thuật đã xảy ra.
Chỉ dùng thông tin trong CONTEXT dưới đây, không bịa thêm. Nếu không đủ thông tin để trả lời, hãy nói rõ là không biết.

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""


def global_id(incident_id: str, entity_id: str) -> str:
    return f"{incident_id}:{entity_id}"


# ---------------------------------------------------------------------------
# Neo4j: build câu Cypher (hàm thuần, test được không cần Neo4j thật) + thực thi
# ---------------------------------------------------------------------------


def build_ingest_queries(graph: IncidentGraph) -> list[tuple[str, dict]]:
    """Sinh danh sách (cypher, params) để ghi 1 IncidentGraph vào Neo4j.
    Toàn bộ dùng MERGE nên chạy lại nhiều lần trên cùng dữ liệu là an toàn
    (idempotent), không tạo trùng node/cạnh."""
    queries: list[tuple[str, dict]] = [
        ("MERGE (i:Incident {incident_id: $incident_id})", {"incident_id": graph.incident_id})
    ]

    if graph.entities:
        queries.append(
            (
                """
                UNWIND $entities AS e
                MATCH (i:Incident {incident_id: $incident_id})
                MERGE (ent:Entity {global_id: e.global_id})
                SET ent.entity_id = e.entity_id, ent.type = e.type,
                    ent.text = e.text, ent.confidence = e.confidence
                MERGE (i)-[:HAS_ENTITY]->(ent)
                """,
                {
                    "incident_id": graph.incident_id,
                    "entities": [
                        {
                            "global_id": global_id(graph.incident_id, e.id),
                            "entity_id": e.id,
                            "type": e.type,
                            "text": e.text,
                            "confidence": e.confidence,
                        }
                        for e in graph.entities
                    ],
                },
            )
        )

    # Nhóm theo relation_type vì Cypher không cho tham số hóa tên relationship
    # -- mỗi loại quan hệ ra 1 câu UNWIND riêng, tên loại interpolate thẳng
    # (an toàn nhờ assert RELATION_TYPES ở đầu file).
    rows_by_type: dict[str, list[dict]] = {}
    for r in graph.relations:
        rows_by_type.setdefault(r.relation_type, []).append(
            {
                "source": global_id(graph.incident_id, r.source_id),
                "target": global_id(graph.incident_id, r.target_id),
                "confidence": r.confidence,
                "evidence": r.evidence,
            }
        )

    for relation_type, rows in rows_by_type.items():
        queries.append(
            (
                f"""
                UNWIND $rows AS row
                MATCH (a:Entity {{global_id: row.source}})
                MATCH (b:Entity {{global_id: row.target}})
                MERGE (a)-[rel:{relation_type}]->(b)
                SET rel.confidence = row.confidence, rel.evidence = row.evidence
                """,
                {"rows": rows},
            )
        )

    return queries


@lru_cache(maxsize=1)
def _driver():
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


def ensure_constraints() -> None:
    with _driver().session() as session:
        for cypher in CONSTRAINT_QUERIES:
            session.run(cypher)


def ingest(graph: IncidentGraph) -> dict:
    """Ghi 1 IncidentGraph vào Neo4j."""
    with _driver().session() as session:
        for cypher, params in build_ingest_queries(graph):
            session.run(cypher, params)
    return {"incident_id": graph.incident_id, "entities": len(graph.entities), "relations": len(graph.relations)}


def _fetch_subgraph_facts(tx, incident_id: str) -> list[str]:
    result = tx.run(
        """
        MATCH (:Incident {incident_id: $incident_id})-[:HAS_ENTITY]->(a:Entity)-[rel]->(b:Entity)
        RETURN a.text AS source_text, a.type AS source_type, type(rel) AS relation_type,
               b.text AS target_text, b.type AS target_type
        """,
        incident_id=incident_id,
    )
    return [
        f"{r['source_text']} ({r['source_type']}) -[{r['relation_type']}]-> {r['target_text']} ({r['target_type']})"
        for r in result
    ]


def fetch_incident_facts(incident_id: str) -> list[str]:
    with _driver().session() as session:
        return session.execute_read(_fetch_subgraph_facts, incident_id)


# ---------------------------------------------------------------------------
# Embedding + Qdrant
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embed(texts: list[str]) -> list[list[float]]:
    """Embedding chạy local qua sentence-transformers, không gọi API ngoài."""
    return _embedder().encode(list(texts), normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_qdrant_collection() -> None:
    client = _qdrant()
    if not client.collection_exists(settings.qdrant_collection):
        dim = len(embed(["_"])[0])
        client.create_collection(
            settings.qdrant_collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )


def _point_id(*parts: str) -> str:
    # Qdrant chỉ nhận id là số nguyên hoặc UUID -- uuid5 để cùng (incident_id,
    # chunk_id) luôn sinh cùng 1 id, upsert lại không tạo điểm trùng.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts)))


def build_summary_text(processed_incident: dict, graph: IncidentGraph) -> str:
    """Tóm tắt ngắn cấp incident, dùng làm 1 điểm embedding riêng (doc_type
    "summary") bên cạnh các chunk, để câu hỏi tổng quát vẫn tìm ra được incident
    dù không khớp nguyên văn với chunk nào."""
    entity_texts = ", ".join(sorted({e.text for e in graph.entities})[:15])
    return f"{processed_incident['title']}. Thực thể liên quan: {entity_texts}" if entity_texts else processed_incident["title"]


def build_qdrant_points(processed_incident: dict, graph: IncidentGraph) -> list[PointStruct]:
    incident_id = processed_incident["incident_id"]
    chunk_texts = [c["text"] for c in processed_incident["chunks"]]
    summary_text = build_summary_text(processed_incident, graph)

    vectors = embed(chunk_texts + [summary_text])
    common_payload = {
        "incident_id": incident_id,
        "title": processed_incident["title"],
        "source_url": processed_incident["source_url"],
    }

    points = [
        PointStruct(
            id=_point_id(incident_id, chunk["chunk_id"]),
            vector=vector,
            payload={**common_payload, "chunk_id": chunk["chunk_id"], "doc_type": "chunk", "text": chunk["text"]},
        )
        for chunk, vector in zip(processed_incident["chunks"], vectors[:-1])
    ]
    points.append(
        PointStruct(
            id=_point_id(incident_id, "summary"),
            vector=vectors[-1],
            payload={**common_payload, "doc_type": "summary", "text": summary_text},
        )
    )
    return points


def index_incident(processed_incident: dict, graph: IncidentGraph) -> int:
    ensure_qdrant_collection()
    points = build_qdrant_points(processed_incident, graph)
    _qdrant().upsert(settings.qdrant_collection, points=points)
    return len(points)


# ---------------------------------------------------------------------------
# Truy vấn GraphRAG: Qdrant tìm ứng viên -> Neo4j lấy subgraph -> Qwen trả lời
# ---------------------------------------------------------------------------


def build_context_blocks(hits, max_incidents: int = 3) -> tuple[list[str], list[dict]]:
    """Gom kết quả Qdrant về tối đa `max_incidents` incident khác nhau (nhiều
    chunk trùng 1 incident chỉ tính 1 lần), ghép với subgraph quan hệ lấy từ
    Neo4j. Tách riêng khỏi query() để test được mà không cần Qdrant/Neo4j thật
    (truyền hits giả)."""
    seen_incidents: set[str] = set()
    blocks, sources = [], []
    for hit in hits:
        incident_id = hit.payload["incident_id"]
        if incident_id in seen_incidents:
            continue
        if len(seen_incidents) >= max_incidents:
            break
        seen_incidents.add(incident_id)

        facts = fetch_incident_facts(incident_id)
        block = f"[{incident_id}] {hit.payload['title']}\nĐoạn liên quan: {hit.payload['text']}"
        if facts:
            block += "\nCác quan hệ đã biết:\n" + "\n".join(f"- {f}" for f in facts)
        blocks.append(block)
        sources.append(
            {"incident_id": incident_id, "title": hit.payload["title"], "source_url": hit.payload["source_url"]}
        )
    return blocks, sources


def query(question: str, top_k: int = 5) -> dict:
    """embed câu hỏi -> Qdrant top-k -> gom về tối đa 3 incident khác nhau ->
    lấy subgraph Neo4j mỗi incident -> ghép context -> Qwen sinh câu trả lời.

    Trả {"answer": str, "sources": [{"incident_id", "title", "source_url"}]}.
    """
    ensure_qdrant_collection()
    (question_vector,) = embed([question])
    hits = _qdrant().query_points(settings.qdrant_collection, query=question_vector, limit=top_k).points

    blocks, sources = build_context_blocks(hits)
    context = "\n\n".join(blocks) if blocks else "(không tìm thấy tài liệu liên quan)"
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    client = OllamaClient(base_url=settings.ollama_base_url, model=settings.llm_model_name)
    answer = client.generate_text(prompt)

    return {"answer": answer, "sources": sources}
