from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retrieval.retriever import RetrievedChunk, _run_query


def _make_fake_row(cosine_distance: float = 0.2) -> MagicMock:
    row = MagicMock()
    row.chunk_id = 1
    row.document_id = 1
    row.arxiv_id = "2301.00001"
    row.title = "Test Paper"
    row.chunk_text = "This is a test chunk."
    row.chunk_index = 0
    row.cosine_distance = cosine_distance
    return row


def test_retrieve_returns_correct_type():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [_make_fake_row()]
    query_vec = np.random.rand(384).astype(np.float32)
    results = _run_query(query_vec, top_k=5, db=db)
    assert isinstance(results, list)
    assert isinstance(results[0], RetrievedChunk)


def test_retrieve_top_k_passed_to_sql():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []
    query_vec = np.random.rand(384).astype(np.float32)
    _run_query(query_vec, top_k=3, db=db)
    call_kwargs = db.execute.call_args[0][1]
    assert call_kwargs["top_k"] == 3


def test_retrieved_chunk_similarity_score():
    chunk = RetrievedChunk(
        chunk_id=1,
        document_id=1,
        arxiv_id="2301.00001",
        title="Test",
        chunk_text="text",
        chunk_index=0,
        cosine_distance=0.3,
    )
    assert abs(chunk.similarity_score - 0.7) < 1e-9


def test_retrieve_empty_results():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []
    query_vec = np.random.rand(384).astype(np.float32)
    results = _run_query(query_vec, top_k=5, db=db)
    assert results == []


def test_retrieve_multiple_results_ordered():
    db = MagicMock()
    rows = [_make_fake_row(0.1), _make_fake_row(0.3), _make_fake_row(0.5)]
    db.execute.return_value.fetchall.return_value = rows
    query_vec = np.random.rand(384).astype(np.float32)
    results = _run_query(query_vec, top_k=5, db=db)
    assert len(results) == 3
    distances = [r.cosine_distance for r in results]
    assert distances == sorted(distances)
