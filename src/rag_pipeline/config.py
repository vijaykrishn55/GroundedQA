"""Environment-based configuration for local and container execution."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from rag_pipeline.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    generation_model: str = "openai/gpt-oss-120b"
    embedding_dimension: int = 384
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "basic_rag_knowledge"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_relevance_score: float = 0.25

    @classmethod
    def from_env(cls) -> "Settings":
        """Load and validate environment configuration."""
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY", "").strip()

        if not api_key:
            raise ConfigurationError("GROQ_API_KEY is missing. Add your Groq API key to .env.")

        try:
            settings = cls(
                groq_api_key=api_key,
                embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
                generation_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
                qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
                qdrant_collection=os.getenv("QDRANT_COLLECTION", "basic_rag_knowledge"),
                chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
                top_k=int(os.getenv("TOP_K", "4")),
                min_relevance_score=float(os.getenv("MIN_RELEVANCE_SCORE", "0.25")),
            )
        except ValueError as exc:
            raise ConfigurationError(
                f"An environment setting has an invalid numeric value: {exc}"
            ) from exc

        if settings.embedding_dimension <= 0:
            raise ConfigurationError("Embedding dimension must be positive.")

        if settings.chunk_size <= 0 or settings.chunk_overlap < 0:
            raise ConfigurationError("Chunk size must be positive and overlap cannot be negative.")

        if settings.chunk_overlap >= settings.chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if settings.top_k <= 0:
            raise ConfigurationError("TOP_K must be positive.")

        if not 0.0 <= settings.min_relevance_score <= 1.0:
            raise ConfigurationError("MIN_RELEVANCE_SCORE must be between 0 and 1.")

        if not settings.qdrant_collection.strip():
            raise ConfigurationError("QDRANT_COLLECTION cannot be empty.")

        return settings
