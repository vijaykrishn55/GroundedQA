"""Protocols that keep the pipeline testable without external services."""

from typing import Protocol

from rag_pipeline.schemas import Chunk, RetrievedChunk


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def recreate_collection(self) -> None: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self, query_vector: list[float], top_k: int, min_score: float
    ) -> list[RetrievedChunk]: ...


class AnswerGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...
