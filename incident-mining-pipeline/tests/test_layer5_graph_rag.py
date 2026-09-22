from types import SimpleNamespace

from src.models.schemas import Entity, IncidentGraph, Relation
from src.pipeline import layer5_graph_rag as layer5


def make_graph() -> IncidentGraph:
    return IncidentGraph(
        incident_id="INC-1",
        entities=[
            Entity(id="e0", type="ROOT_CAUSE", text="a deploy changed a field type", confidence=0.9),
            Entity(id="e1", type="SYMPTOM", text="distribution failed", confidence=0.9),
            Entity(id="e2", type="MITIGATION", text="rolled back the change", confidence=0.9),
        ],
        relations=[
            Relation(source_id="e0", target_id="e1", relation_type="CAUSES", confidence=0.9, evidence="ev1"),
            Relation(source_id="e1", target_id="e2", relation_type="MITIGATED_BY", confidence=0.8, evidence="ev2"),
        ],
    )


def test_global_id_namespaces_by_incident():
    assert layer5.global_id("INC-1", "e0") == "INC-1:e0"
    assert layer5.global_id("INC-1", "e0") != layer5.global_id("INC-2", "e0")


def test_build_ingest_queries_incident_and_entity_query_present():
    queries = layer5.build_ingest_queries(make_graph())
    cyphers = [c for c, _ in queries]
    assert any("MERGE (i:Incident" in c for c in cyphers)
    assert any("MERGE (ent:Entity" in c for c in cyphers)


def test_build_ingest_queries_entity_params_use_global_id():
    queries = layer5.build_ingest_queries(make_graph())
    entity_params = next(p for c, p in queries if "MERGE (ent:Entity" in c)
    global_ids = {e["global_id"] for e in entity_params["entities"]}
    assert global_ids == {"INC-1:e0", "INC-1:e1", "INC-1:e2"}


def test_build_ingest_queries_skips_entity_query_when_no_entities():
    graph = IncidentGraph(incident_id="INC-2", entities=[], relations=[])
    queries = layer5.build_ingest_queries(graph)
    assert len(queries) == 1
    assert "Incident" in queries[0][0]


def test_build_ingest_queries_groups_relations_by_type_into_separate_queries():
    queries = layer5.build_ingest_queries(make_graph())
    relation_queries = {c: p for c, p in queries if "MATCH (a:Entity" in c}
    assert sum("CAUSES]->" in c for c in relation_queries) == 1
    assert sum("MITIGATED_BY]->" in c for c in relation_queries) == 1


def test_build_ingest_queries_relation_rows_reference_global_ids():
    queries = layer5.build_ingest_queries(make_graph())
    causes_params = next(p for c, p in queries if "CAUSES]->" in c)
    assert causes_params["rows"] == [
        {"source": "INC-1:e0", "target": "INC-1:e1", "confidence": 0.9, "evidence": "ev1"}
    ]


def test_build_ingest_queries_two_relations_same_type_become_one_query_two_rows():
    graph = IncidentGraph(
        incident_id="INC-3",
        entities=[
            Entity(id="e0", type="SERVICE", text="a", confidence=0.9),
            Entity(id="e1", type="COMPONENT", text="b", confidence=0.9),
            Entity(id="e2", type="COMPONENT", text="c", confidence=0.9),
        ],
        relations=[
            Relation(source_id="e0", target_id="e1", relation_type="AFFECTS", confidence=0.9),
            Relation(source_id="e0", target_id="e2", relation_type="AFFECTS", confidence=0.9),
        ],
    )
    queries = layer5.build_ingest_queries(graph)
    affects_queries = [(c, p) for c, p in queries if "AFFECTS]->" in c]
    assert len(affects_queries) == 1
    assert len(affects_queries[0][1]["rows"]) == 2


def test_point_id_is_deterministic_and_differs_by_input():
    a = layer5._point_id("INC-1", "c0")
    b = layer5._point_id("INC-1", "c0")
    c = layer5._point_id("INC-1", "c1")
    assert a == b
    assert a != c


def test_build_summary_text_includes_sorted_unique_entity_texts():
    processed = {"title": "CircleCI outage"}
    graph = make_graph()
    summary = layer5.build_summary_text(processed, graph)
    assert summary.startswith("CircleCI outage. Thực thể liên quan:")
    assert "distribution failed" in summary and "rolled back the change" in summary


def test_build_summary_text_falls_back_to_title_when_no_entities():
    graph = IncidentGraph(incident_id="INC-1", entities=[], relations=[])
    assert layer5.build_summary_text({"title": "CircleCI outage"}, graph) == "CircleCI outage"


def _hit(incident_id, title="T", source_url="u", text="chunk text"):
    return SimpleNamespace(payload={"incident_id": incident_id, "title": title, "source_url": source_url, "text": text})


def test_build_context_blocks_dedupes_incidents_and_caps_at_max(monkeypatch):
    monkeypatch.setattr(layer5, "fetch_incident_facts", lambda incident_id: [])
    hits = [_hit("A"), _hit("A"), _hit("B"), _hit("C"), _hit("D")]
    blocks, sources = layer5.build_context_blocks(hits, max_incidents=3)
    assert [s["incident_id"] for s in sources] == ["A", "B", "C"]
    assert len(blocks) == 3


def test_build_context_blocks_appends_facts_when_present(monkeypatch):
    monkeypatch.setattr(
        layer5, "fetch_incident_facts", lambda incident_id: ["X (SERVICE) -[AFFECTS]-> Y (COMPONENT)"]
    )
    blocks, _ = layer5.build_context_blocks([_hit("A")])
    assert "Các quan hệ đã biết" in blocks[0] and "AFFECTS" in blocks[0]


def test_build_context_blocks_omits_facts_section_when_empty(monkeypatch):
    monkeypatch.setattr(layer5, "fetch_incident_facts", lambda incident_id: [])
    blocks, _ = layer5.build_context_blocks([_hit("A")])
    assert "Các quan hệ đã biết" not in blocks[0]
