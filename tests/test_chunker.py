import tiktoken

from src.ingestion.chunker import chunk_text

_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def test_chunk_returns_list():
    chunks = chunk_text("Hello world. " * 10)
    assert isinstance(chunks, list)


def test_chunk_non_empty_input():
    chunks = chunk_text("Hello world. " * 100)
    assert len(chunks) >= 1


def test_chunk_text_basic_split():
    long_text = "This is a sentence. " * 200
    chunks = chunk_text(long_text, chunk_size=512, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert _token_count(chunk) <= 512


def test_chunk_empty_string():
    assert chunk_text("") == []


def test_chunk_single_short_text():
    short = "Hello world."
    chunks = chunk_text(short, chunk_size=512, chunk_overlap=50)
    assert chunks == [short]


def test_chunk_strips_whitespace():
    text = "  Hello world.  " * 100
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=50)
    for chunk in chunks:
        assert chunk == chunk.strip()


def test_chunk_no_empty_chunks():
    text = "\n\n".join(["Sentence number {}.".format(i) for i in range(300)])
    chunks = chunk_text(text, chunk_size=512, chunk_overlap=50)
    for chunk in chunks:
        assert chunk.strip() != ""
