"""Application-specific exceptions used at integration boundaries."""


class RAGError(Exception):
    """Base exception for expected RAG application failures."""


class ConfigurationError(RAGError):
    """Raised when required application configuration is missing or invalid."""


class DocumentLoadError(RAGError):
    """Raised when the knowledge-base file cannot be read or validated."""


class EmbeddingError(RAGError):
    """Raised when local embedding generation fails."""


class VectorStoreError(RAGError):
    """Raised when a Qdrant operation fails."""


class GenerationError(RAGError):
    """Raised when Groq answer generation fails."""
