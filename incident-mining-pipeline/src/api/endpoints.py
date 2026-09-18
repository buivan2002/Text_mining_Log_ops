"""API routes: /api/v1/analyze, /api/v1/search."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline import layer5_graph_rag

router = APIRouter()
orchestrator = PipelineOrchestrator()


class AnalyzeRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    return orchestrator.run(request.text)


@router.post("/search")
def search(request: SearchRequest):
    return layer5_graph_rag.query(request.query, request.top_k)
