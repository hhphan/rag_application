import os

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        full_name = f"sentence-transformers/{model_name}"
        try:
            self._model = SentenceTransformer(full_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(full_name, local_files_only=False)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            _embedder = Embedder(model_name)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Embedding model cache is corrupted. Clear it and restart:\n"
                f"  rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--{model_name.replace('/', '--')}\n"
                f"Original error: {exc}"
            ) from exc
    return _embedder


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return get_embedder().encode(texts, batch_size=batch_size)
