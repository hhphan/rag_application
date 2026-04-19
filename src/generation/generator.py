import os

import anthropic

from src.generation.prompts import NO_RESULTS_MESSAGE, RAG_QUERY_TEMPLATE, SYSTEM_PROMPT
from src.retrieval.retriever import RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f'[{i}] From "{chunk.title}" (arXiv:{chunk.arxiv_id}):\n{chunk.chunk_text}'
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024")),
) -> str:
    if not retrieved_chunks:
        return NO_RESULTS_MESSAGE

    context = _format_context(retrieved_chunks)
    user_message = RAG_QUERY_TEMPLATE.format(context=context, question=query)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text
