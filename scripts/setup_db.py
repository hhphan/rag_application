import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine, text

from src.database.models import Base
from src.database.session import engine


def _enable_pgvector() -> None:
    superuser_url = os.getenv("SUPERUSER_DATABASE_URL")
    if superuser_url:
        su_engine = create_engine(superuser_url)
    else:
        su_engine = engine

    try:
        with su_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        print("pgvector extension enabled.")
    except Exception as exc:
        if "already exists" in str(exc):
            print("pgvector extension already enabled.")
        elif "permission denied" in str(exc):
            print(
                "WARNING: Cannot enable pgvector — permission denied.\n"
                "Run manually as superuser:\n"
                "  PGPASSWORD=postgres psql -U postgres -h localhost -p 5433 -d rag_db "
                "-c \"CREATE EXTENSION IF NOT EXISTS vector;\""
            )
        else:
            raise


def main() -> None:
    print("Setting up database...")

    _enable_pgvector()

    Base.metadata.create_all(engine)
    print("Tables created: documents, document_chunks.")

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops);
        """))
        conn.commit()
    print("HNSW index created on document_chunks.embedding.")

    print("Database setup complete.")


if __name__ == "__main__":
    main()
