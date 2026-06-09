from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from rag.parser import parse_policy_markdown


class ChromaPolicyStore:
    """Chroma-backed policy index for markdown policy chunks."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: Any,
        collection_name: str = "policy_chunks",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_index(self, markdown_path: Path) -> None:
        if self.collection.count() == 0:
            self.rebuild(markdown_path)

    def rebuild(self, markdown_path: Path) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        markdown_text = Path(markdown_path).read_text(encoding="utf-8")
        chunks = parse_policy_markdown(markdown_text)
        if not chunks:
            raise ValueError(f"No policy chunks parsed from {markdown_path}")

        documents = [chunk["rendered_text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(documents)
        ids = [f"policy_chunk_{index:04d}" for index in range(len(chunks))]
        metadatas = [
            {
                "section_h2": chunk["section_h2"],
                "section_h3": chunk["section_h3"],
                "citation": chunk["citation"],
                "content": chunk["content"],
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        if top_k <= 0 or self.collection.count() == 0:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            metadata = metadata or {}
            hits.append(
                {
                    "id": chunk_id,
                    "citation": metadata.get("citation", ""),
                    "section_h2": metadata.get("section_h2", ""),
                    "section_h3": metadata.get("section_h3", ""),
                    "content": metadata.get("content", document),
                    "rendered_text": document,
                    "distance": float(distance),
                }
            )
        return hits
