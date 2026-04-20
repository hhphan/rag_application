import numpy as np
import pytest

from src.embedding.embedder import embed_texts, get_embedder


def test_embed_returns_correct_shape():
    result = embed_texts(["hello world"])
    assert result.shape == (1, 384)


def test_embed_batch_shape():
    result = embed_texts(["a", "b", "c"])
    assert result.shape == (3, 384)


def test_embed_normalized():
    result = embed_texts(["test sentence for normalization"])
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_batch_all_normalized():
    texts = ["sentence one", "sentence two", "sentence three"]
    result = embed_texts(texts)
    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_get_embedder_singleton():
    e1 = get_embedder()
    e2 = get_embedder()
    assert e1 is e2
