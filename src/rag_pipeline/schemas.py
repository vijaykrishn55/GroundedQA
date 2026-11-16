"""Small, explicit data structures passed between pipeline components."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    content: str
    source: str = ""


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    title: str
    content: str
    chunk_index: int
    source: str = ""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: str
    document_id: str
    title: str
    content: str
    chunk_index: int
    source: str
    score: float


@dataclass(frozen=True, slots=True)
class SourceReference:
    chunk_id: str
    document_id: str
    title: str
    source: str
    chunk_index: int
    score: float


@dataclass(frozen=True, slots=True)
class RAGResponse:
    query: str
    answer: str
    sources: tuple[SourceReference, ...]
    used_context: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable response structure."""
        return asdict(self)
