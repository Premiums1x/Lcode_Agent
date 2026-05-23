"""Vector store using ChromaDB for RAG."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from lcode.core.config import settings


class VectorStore:
    """ChromaDB-based vector store for document retrieval."""

    def __init__(
        self, collection_name: str = "lcode_docs", persist_dir: Path | None = None
    ) -> None:
        self.persist_dir = persist_dir or settings.vector_db_path
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.Client(
            ChromaSettings(
                persist_directory=str(self.persist_dir),
                is_persistent=True,
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self, documents: list[Any], embeddings: list[list[float]] | None = None
    ) -> None:
        """Add documents with optional pre-computed embeddings.

        Args:
            documents: List of Document objects.
            embeddings: Optional pre-computed embeddings.
        """
        from lcode.llm.openai_provider import OpenAIProvider

        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]

        if embeddings is None:
            provider = OpenAIProvider()
            embeddings = provider.embed(texts)

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
        )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector store.

        Args:
            query_text: The query string.
            top_k: Number of results.
            embedding: Optional pre-computed query embedding.

        Returns:
            List of result dicts with content, metadata, and distance.
        """
        from lcode.llm.openai_provider import OpenAIProvider

        if embedding is None:
            provider = OpenAIProvider()
            embedding = provider.embed([query_text])[0]

        results = self.collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=top_k,
        )

        output = []
        ids = results.get("ids", [[]])
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        distances = results.get("distances", [[]])
        if ids and ids[0]:
            for i in range(len(ids[0])):
                output.append(
                    {
                        "id": ids[0][i],
                        "content": documents[0][i] if documents else "",
                        "metadata": metadatas[0][i] if metadatas and metadatas[0] else {},
                        "distance": distances[0][i] if distances and distances[0] else 0,
                    }
                )
        return output

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection.name)

    def count(self) -> int:
        """Return number of documents in collection."""
        return self.collection.count()
