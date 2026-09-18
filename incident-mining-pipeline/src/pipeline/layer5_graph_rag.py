"""Tầng 5: Knowledge Graph & RAG Query (Neo4j).

Ghi dữ liệu đã xác thực vào Neo4j knowledge graph và cung cấp truy vấn
dạng RAG (retrieval-augmented generation) cho việc tra cứu sự cố tương tự.
"""

from src.models.schemas import IncidentGraph


def ingest(graph_data: IncidentGraph) -> dict:
    """Ghi dữ liệu cây sự cố vào Neo4j."""
    raise NotImplementedError


def query(question: str, top_k: int = 5) -> list[dict]:
    """Truy vấn RAG trên knowledge graph để trả lời câu hỏi về sự cố."""
    raise NotImplementedError
