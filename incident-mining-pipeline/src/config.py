"""Cấu hình tham số hệ thống: API keys, DB URLs, thresholds."""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM (Qwen local qua Ollama)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen2.5:7b-instruct")

    # NER model đã fine-tune (Layer 2)
    ner_model_path: str = os.getenv("NER_MODEL_PATH", "models/deberta-ner-incident")

    # Embedding model cho Qdrant (Layer 5), chạy local qua sentence-transformers
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    # Qdrant vector DB
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = "incident_embeddings"

    # Pipeline thresholds
    ner_confidence_threshold: float = float(os.getenv("NER_CONFIDENCE_THRESHOLD", "0.7"))
    relation_confidence_threshold: float = float(os.getenv("RELATION_CONFIDENCE_THRESHOLD", "0.6"))
    relation_max_retries: int = int(os.getenv("RELATION_MAX_RETRIES", "3"))

    class Config:
        env_file = ".env"


settings = Settings()
