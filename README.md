# RAG Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that ingests arXiv research papers, stores vector embeddings in PostgreSQL, and exposes a Streamlit GUI for natural-language querying.

## Architecture

```
[arXiv API]
     ↓
[Ingestion]  → fetch metadata + PDFs → parse text → chunk (512 tokens)
     ↓
[Embedding]  → sentence-transformers (all-MiniLM-L6-v2) → 384-dim vectors
     ↓
[PostgreSQL + pgvector]  → cosine similarity search
     ↓
[Generation] → Anthropic Claude API (claude-sonnet-4-6)
     ↓
[Streamlit GUI]  → chat interface with source citations
```

## GUI Preview

![Streamlit GUI](docs/streamlit_GUI_screenshot.png)

*Chat interface showing a query about ingested arXiv papers, with source chunk citations in the sidebar.*

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| Docker Desktop | latest | Must be running before any Docker commands |
| PostgreSQL client | any | For `psql` CLI (comes with PostgreSQL install) |
| Anthropic API key | — | Required for answer generation |

> **Windows note:** Local PostgreSQL 18 occupies port 5432. The Docker container runs on port **5433** to avoid conflicts.

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd rag_pipeline
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set your Anthropic API key:

```bash
ANTHROPIC_API_KEY=your_key_here
DATABASE_URL=postgresql://rag_user:rag_user@localhost:5433/rag_db
SUPERUSER_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/rag_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=5
```

### 5. Start the database

Make sure Docker Desktop is running, then:

```bash
docker-compose up -d
```

This starts a PostgreSQL 16 + pgvector container on port **5433**.

### 6. Create the database user

Run once after the container starts:

```bash
PGPASSWORD=postgres psql -U postgres -h localhost -p 5433 \
  -c "CREATE DATABASE rag_db;" \
  -c "CREATE USER rag_user WITH PASSWORD 'rag_user';" \
  -c "GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;"

PGPASSWORD=postgres psql -U postgres -h localhost -p 5433 -d rag_db \
  -c "GRANT ALL ON SCHEMA public TO rag_user;"
```

### 7. Initialize the database schema

```bash
python scripts/setup_db.py
```

This creates the `documents` and `document_chunks` tables, enables the pgvector extension, and creates an HNSW index for fast similarity search.

### 8. Download the embedding model

Download the model defined in `EMBEDDING_MODEL` (`.env`) before running ingestion or the GUI. This avoids mid-run interruptions and caches the model for all future runs.

```bash
python scripts/download_model.py
```

The model (`all-MiniLM-L6-v2`, ~90 MB) is saved to `~/.cache/huggingface/` and reused automatically. The script retries up to 5 times on network errors.

---

## Usage

### Ingest arXiv papers

> **First run note:** The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded from HuggingFace automatically on first run and cached at `~/.cache/huggingface/`. This may take a few minutes depending on your connection. Subsequent runs load from cache instantly.

```bash
# Small test run (recommended first)
python scripts/ingest.py --category cs.AI --max-results 5

# Full ingestion (500 papers — takes several minutes)
python scripts/ingest.py --category cs.AI --max-results 500
```

Available categories: `cs.AI`, `cs.CL`, `cs.LG`, `cs.CV`, `cs.NE`, etc.

### Launch the GUI

```bash
streamlit run src/gui/app.py
```

Open your browser at `http://localhost:8501` and start asking questions about the ingested papers.

---

## Project Structure

```
rag_pipeline/
├── docker-compose.yml       # PostgreSQL + pgvector container (port 5433)
├── requirements.txt
├── .env.example
├── src/
│   ├── database/
│   │   ├── models.py        # SQLAlchemy ORM models (Document, DocumentChunk)
│   │   └── session.py       # Engine + get_db() context manager
│   ├── ingestion/
│   │   ├── fetcher.py       # arXiv API client
│   │   ├── parser.py        # PDF download + text extraction
│   │   └── chunker.py       # 512-token chunking with tiktoken
│   ├── embedding/
│   │   └── embedder.py      # sentence-transformers wrapper
│   ├── retrieval/
│   │   └── retriever.py     # pgvector cosine similarity search
│   ├── generation/
│   │   ├── prompts.py       # All prompt templates
│   │   └── generator.py     # Anthropic Claude API integration
│   └── gui/
│       └── app.py           # Streamlit chat interface
├── scripts/
│   ├── download_model.py    # Pre-download embedding model from .env
│   ├── setup_db.py          # One-time DB initialisation
│   └── ingest.py            # Ingestion pipeline CLI
└── tests/
    ├── test_chunker.py
    ├── test_embedder.py
    └── test_retriever.py
```

---

## Running Tests

```bash
pytest tests/
```

---

## Troubleshooting

### `extension "vector" is not available`
Your Python app is connecting to a PostgreSQL instance without pgvector. Check that:
- Docker Desktop is running (`docker ps`)
- The container is on port 5433 (`docker-compose ps`)
- `DATABASE_URL` in `.env` uses port `5433`

### `unable to get image ... dockerDesktopLinuxEngine`
Docker Desktop is not running. Open it from the Start menu and wait for the whale icon to show "Docker Desktop is running", then retry.

### Port 5432 conflict
Local PostgreSQL occupies port 5432. This project's Docker container is intentionally mapped to **5433**. Do not change this unless you stop the local PostgreSQL service first.

### `permission denied to create extension "vector"`
`CREATE EXTENSION` requires superuser privileges. Set `SUPERUSER_DATABASE_URL` in `.env` pointing to the `postgres` superuser, or run manually:
```bash
PGPASSWORD=postgres psql -U postgres -h localhost -p 5433 -d rag_db \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### `FileNotFoundError: 1_Pooling/config.json` during ingestion
The HuggingFace model download was interrupted, leaving a corrupted cache. Clear it and re-run:
```bash
# Windows
rm -rf "$env:USERPROFILE\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2"

# macOS / Linux
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
```
Then retry `python scripts/ingest.py`. The model will re-download cleanly.

### HuggingFace download keeps retrying / `ConnectionResetError`
This is a transient network issue with the HuggingFace CDN. The download will retry up to 5 times automatically. If it ultimately fails, just re-run the ingest command — the download resumes from where it left off once connectivity improves.
