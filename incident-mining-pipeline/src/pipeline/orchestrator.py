"""Controller chính điều khiển luồng dữ liệu qua 5 tầng xử lý."""

from src.pipeline import (
    layer1_preprocessing,
    layer2_ner,
    layer3_llm_relation,
    layer4_validation,
    layer5_graph_rag,
)


class PipelineOrchestrator:
    """Điều phối việc chạy tuần tự 5 tầng: preprocessing -> NER ->
    relation extraction -> validation -> graph/RAG."""

    def __init__(self):
        pass

    def run(self, raw_text: str) -> dict:
        cleaned = layer1_preprocessing.preprocess(raw_text)
        entities = layer2_ner.extract_entities(cleaned)
        relations = layer3_llm_relation.extract_relations(cleaned, entities)
        validated = layer4_validation.validate(relations)
        result = layer5_graph_rag.ingest(validated)
        return result
