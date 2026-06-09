from __future__ import annotations

from hashlib import blake2b
import math
import os
import re

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        self.backend = "hashing-fallback"
        self.load_error: str | None = None

        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
            self.backend = "sentence-transformers-local"
            return
        except Exception as exc:
            self.load_error = str(exc)

        if os.getenv("EMBEDDING_ALLOW_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
            try:
                self.model = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
                self.load_error = None
            except Exception as exc:
                self.load_error = str(exc)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.model is None:
            return [_hash_embedding(text) for text in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_HASH_DIMENSION = 384


def _hash_embedding(text: str) -> list[float]:
    tokens = _TOKEN_RE.findall(text.lower())
    features = tokens[:]
    features.extend(
        f"{left} {right}"
        for left, right in zip(tokens, tokens[1:], strict=False)
    )

    vector = [0.0] * _HASH_DIMENSION
    for feature in features:
        digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % _HASH_DIMENSION
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
