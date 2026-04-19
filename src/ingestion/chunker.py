import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_encoding: tiktoken.Encoding | None = None


def _token_length(text: str) -> int:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")  # TODO: support configurable tokenizer
    return len(_encoding.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]
