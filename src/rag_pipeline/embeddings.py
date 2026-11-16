"""Local text embedding adapter using Qdrant FastEmbed."""

from typing import Any

from rag_pipeline.errors import EmbeddingError


class FastEmbedProvider:
    """Generate document and query embeddings locally."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        model: Any | None = None,
    ) -> None:
        try:
            if model is None:
                from fastembed import TextEmbedding

                model = TextEmbedding(model_name=model_name)

            self.model = model

        except Exception as exc:
            raise EmbeddingError(f"Could not initialize FastEmbed: {exc}") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for document passages."""
        if not texts:
            return []

        try:
            inputs = [f"passage: {text}" for text in texts]
            vectors = list(self.model.embed(inputs))
            return [vector.tolist() for vector in vectors]

        except Exception as exc:
            raise EmbeddingError(f"Document embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a user query."""
        if not text.strip():
            raise ValueError("Query cannot be empty.")

        try:
            vectors = list(self.model.query_embed(f"query: {text}"))

            if not vectors:
                raise EmbeddingError("FastEmbed returned no query embedding.")

            return vectors[0].tolist()

        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc
