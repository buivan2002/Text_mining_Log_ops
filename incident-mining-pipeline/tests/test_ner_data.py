from src.iob2 import IGNORE_INDEX, LABEL2ID
from src.ner_data import decode_tags, load_iob2_rows, make_document_folds


def test_folds_partition_documents_without_overlap():
    ids = [f"INC-{i}" for i in range(10)] * 3  # nhiều chunk cùng tài liệu
    folds = make_document_folds(ids, k=4, seed=0)
    flat = [i for fold in folds for i in fold]
    assert len(folds) == 4
    assert sorted(flat) == sorted(set(ids))  # mỗi tài liệu đúng một lần
    assert sorted(len(f) for f in folds) == [2, 2, 3, 3]


def test_folds_are_deterministic_and_capped_by_document_count():
    ids = ["a", "b", "c"]
    assert make_document_folds(ids, 4, seed=1) == make_document_folds(ids, 4, seed=1)
    assert len(make_document_folds(ids, 10, seed=1)) == 3


def test_decode_tags_skips_ignored_positions():
    o, b = LABEL2ID["O"], LABEL2ID["B-SERVICE"]
    true_tags, pred_tags = decode_tags([IGNORE_INDEX, o, b, IGNORE_INDEX], [o, b, b, o])
    assert true_tags == ["O", "B-SERVICE"]
    assert pred_tags == ["B-SERVICE", "B-SERVICE"]


def test_load_iob2_rows_reads_all_incident_files(tmp_path):
    (tmp_path / "INC-2.jsonl").write_text('{"incident_id": "INC-2", "chunk_id": "c0"}\n', encoding="utf-8")
    (tmp_path / "INC-1.jsonl").write_text(
        '{"incident_id": "INC-1", "chunk_id": "c0"}\n{"incident_id": "INC-1", "chunk_id": "c1"}\n', encoding="utf-8"
    )
    (tmp_path / "label_map.json").write_text("{}", encoding="utf-8")  # không phải file dữ liệu
    rows = load_iob2_rows(tmp_path)
    assert [(r["incident_id"], r["chunk_id"]) for r in rows] == [("INC-1", "c0"), ("INC-1", "c1"), ("INC-2", "c0")]
