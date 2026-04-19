SYSTEM_PROMPT: str = """You are a research assistant specializing in AI and machine learning literature.
You answer questions based exclusively on the provided research paper excerpts.
If the answer cannot be found in the provided context, say so explicitly.
Do not make up information or rely on prior knowledge beyond what is given."""

RAG_QUERY_TEMPLATE: str = """Use the following excerpts from research papers to answer the question.

Context:
{context}

Question: {question}

Answer based only on the provided context. If you cite specific papers, mention their titles."""

NO_RESULTS_MESSAGE: str = (
    "I could not find relevant information in the ingested research papers to answer your question. "
    "This may mean the topic has not been ingested yet, or the question is outside the scope of the available papers."
)
