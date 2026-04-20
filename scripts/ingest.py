import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import text

from src.database.models import Document, DocumentChunk
from src.database.session import get_db
from src.embedding.embedder import get_embedder
from src.ingestion.chunker import chunk_text
from src.ingestion.fetcher import fetch_arxiv_papers
from src.ingestion.parser import parse_pdf_from_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into the RAG pipeline")
    parser.add_argument("--category", default="cs.AI", help="arXiv category (default: cs.AI)")
    parser.add_argument("--max-results", type=int, default=100, help="Number of papers to fetch")
    return parser.parse_args()


def run_ingestion(category: str, max_results: int) -> None:
    print(f"Fetching {max_results} papers from arXiv category '{category}'...")
    papers = fetch_arxiv_papers(category=category, max_results=max_results)
    print(f"Fetched {len(papers)} papers.")

    chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    embedder = get_embedder()

    with get_db() as db:
        for paper in papers:
            existing = db.execute(
                text("SELECT id FROM documents WHERE arxiv_id = :arxiv_id"),
                {"arxiv_id": paper["arxiv_id"]},
            ).fetchone()
            if existing:
                print(f"Skipping duplicate: {paper['arxiv_id']}")
                continue

            doc = Document(
                arxiv_id=paper["arxiv_id"],
                title=paper["title"],
                authors=paper["authors"],
                abstract=paper["abstract"],
                pdf_url=paper["pdf_url"],
                published_at=paper["published_at"] if paper["published_at"] else None,
            )
            db.add(doc)
            db.flush()

            full_text = parse_pdf_from_url(paper["pdf_url"])
            if not full_text.strip():
                full_text = paper["abstract"]
                print(f"Using abstract fallback: {paper['arxiv_id']}")

            texts = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not texts:
                print(f"No chunks produced for: {paper['arxiv_id']} — skipping.")
                continue

            embeddings = embedder.encode(texts, batch_size=64)

            for idx, (text_str, emb) in enumerate(zip(texts, embeddings)):
                db.add(
                    DocumentChunk(
                        document_id=doc.id,
                        chunk_index=idx,
                        chunk_text=text_str,
                        embedding=emb.tolist(),
                    )
                )

            db.commit()
            print(f"Ingested: {paper['arxiv_id']} ({len(texts)} chunks)")

    print("Ingestion complete.")


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(category=args.category, max_results=args.max_results)
