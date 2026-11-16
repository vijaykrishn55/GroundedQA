"""Qdrant vector-store adapter."""

from typing import Any

from rag_pipeline.errors import VectorStoreError
from rag_pipeline.schemas import Chunk, RetrievedChunk


class QdrantVectorStore:
    """Persist chunk vectors and perform cosine-similarity Top-K search."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        api_key: str = "",
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models

            self.models = models
            self.client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        except Exception as exc:
            raise VectorStoreError(f"Could not initialize Qdrant: {exc}") from exc

        self.collection_name = collection_name
        self.vector_size = vector_size

    def _collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(collection_name=self.collection_name))
        except Exception as exc:
            raise VectorStoreError(f"Could not check the Qdrant collection: {exc}") from exc

    def _create_collection(self) -> None:
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.models.VectorParams(
                    size=self.vector_size,
                    distance=self.models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not create Qdrant collection: {exc}") from exc

    def ensure_collection(self) -> None:
        """Create the collection when it does not exist."""
        if not self._collection_exists():
            self._create_collection()

    def recreate_collection(self) -> None:
        """Replace the collection for a full knowledge-base re-ingestion."""
        if self._collection_exists():
            try:
                self.client.delete_collection(collection_name=self.collection_name)
            except Exception as exc:
                raise VectorStoreError(f"Could not clear the Qdrant collection: {exc}") from exc
        self._create_collection()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Store chunk vectors and their source text as Qdrant payloads."""
        if len(chunks) != len(vectors):
            raise VectorStoreError("Chunk and embedding counts do not match.")
        if not chunks:
            return
        self.ensure_collection()

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != self.vector_size:
                raise VectorStoreError(
                    f"Embedding dimension mismatch: expected {self.vector_size}, got {len(vector)}."
                )
            points.append(
                self.models.PointStruct(
                    id=chunk.id,
                    vector=vector,
                    payload={
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "source": chunk.source,
                    },
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not upsert points into Qdrant: {exc}") from exc

    def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the highest-scoring chunks, filtered by a cosine-score threshold."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not self._collection_exists():
            return []

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=min_score if min_score > 0 else None,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant semantic search failed: {exc}") from exc

        retrieved: list[RetrievedChunk] = []
        for point in response.points:
            payload: dict[str, Any] = point.payload or {}
            retrieved.append(
                RetrievedChunk(
                    id=str(point.id),
                    document_id=str(payload.get("document_id", "")),
                    title=str(payload.get("title", "Untitled source")),
                    content=str(payload.get("content", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    source=str(payload.get("source", "")),
                    score=float(point.score or 0.0),
                )
            )
        return retrieved

    def close(self) -> None:
        """Close the underlying Qdrant client."""
        self.client.close()
