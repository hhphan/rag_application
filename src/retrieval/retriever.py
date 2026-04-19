from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.session import get_db


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    arxiv_id: str
    title: str
    chunk_text: str
    chunk_index: int
    cosine_distance: float

    @property
    def similarity_score(self) -> float:
        return 1.0 - self.cosine_distance


_QUERY = text("""
    SELECT
        dc.id            AS chunk_id,
        dc.document_id,
        d.arxiv_id,
        d.title,
        dc.chunk_text,
        dc.chunk_index,
        dc.embedding <=> CAST(:embedding AS vector) AS cosine_distance
    FROM document_chunks dc
    JOIN documents d ON d.id = dc.document_id
    WHERE dc.embedding IS NOT NULL
    ORDER BY cosine_distance ASC
    LIMIT :top_k
""")


def _run_query(
    query_embedding: np.ndarray, top_k: int, db: Session
) -> list[RetrievedChunk]:
    rows = db.execute(
        _QUERY,
        {"embedding": query_embedding.tolist(), "top_k": top_k},
    ).fetchall()
    return [
        RetrievedChunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            arxiv_id=row.arxiv_id,
            title=row.title,
            chunk_text=row.chunk_text,
            chunk_index=row.chunk_index,
            cosine_distance=float(row.cosine_distance),
        )
        for row in rows
    ]


def retrieve_similar_chunks(
    query_embedding: np.ndarray,
    top_k: int = 5,
    db: Session | None = None,
    # TODO: add min_similarity threshold to filter low-quality results
) -> list[RetrievedChunk]:
    if db is not None:
        return _run_query(query_embedding, top_k, db)
    with get_db() as session:
        return _run_query(query_embedding, top_k, session)
