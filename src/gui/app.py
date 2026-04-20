import os

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from sqlalchemy import text

from src.database.session import get_db
from src.embedding.embedder import Embedder, get_embedder
from src.generation.generator import generate_answer
from src.retrieval.retriever import RetrievedChunk, retrieve_similar_chunks


@st.cache_resource
def load_embedder() -> Embedder:
    return get_embedder()


@st.cache_data(ttl=300)
def get_db_stats() -> dict[str, int]:
    with get_db() as db:
        doc_count = db.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
        chunk_count = db.execute(text("SELECT COUNT(*) FROM document_chunks")).scalar() or 0
    return {"documents": int(doc_count), "chunks": int(chunk_count)}


def render_sources(chunks: list[RetrievedChunk]) -> None:
    with st.expander(f"Sources ({len(chunks)} chunks)"):
        for chunk in chunks:
            st.markdown(f"**{chunk.title}** (arXiv:{chunk.arxiv_id})")
            st.caption(
                f"Similarity: {chunk.similarity_score:.3f} | Chunk {chunk.chunk_index}"
            )
            preview = (
                chunk.chunk_text[:400] + "..."
                if len(chunk.chunk_text) > 400
                else chunk.chunk_text
            )
            st.text(preview)
            st.divider()


def main() -> None:
    st.set_page_config(page_title="RAG Research Assistant", page_icon="📚", layout="wide")

    with st.sidebar:
        st.title("📚 RAG Research Assistant")
        st.caption("Powered by arXiv + Claude")
        st.divider()
        # TODO: add clear chat button
        stats = get_db_stats()
        st.metric("Documents", stats["documents"])
        st.metric("Chunks", stats["chunks"])
        st.divider()

        with st.expander("About"):
            st.markdown(
                f"**Embedding model:** `{os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}`\n\n"
                f"**LLM:** `claude-sonnet-4-6`\n\n"
                f"**Top-k retrieval:** {os.getenv('RETRIEVAL_TOP_K', '5')}"
            )

    st.header("Ask a question about AI/ML research")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    if prompt := st.chat_input("e.g. What are the key ideas behind attention mechanisms?"):
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                embedder = load_embedder()
                query_vec = embedder.encode([prompt])[0]
                top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
                chunks = retrieve_similar_chunks(query_vec, top_k=top_k)

            with st.spinner("Generating answer..."):
                answer = generate_answer(prompt, chunks)

            st.markdown(answer)
            if chunks:
                render_sources(chunks)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": chunks}
        )


if __name__ == "__main__":
    main()
