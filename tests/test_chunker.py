from src.ingestion.chunker import chunk_text


def test_chunk_returns_list():
    chunks = chunk_text("Hello world. " * 10)
    assert isinstance(chunks, list)


def test_chunk_non_empty_input():
    chunks = chunk_text("Hello world. " * 100)
    assert len(chunks) >= 1
