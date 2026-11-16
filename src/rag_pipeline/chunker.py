"""A small deterministic character chunker with configurable overlap."""

from uuid import NAMESPACE_URL, uuid5

from rag_pipeline.schemas import Chunk, Document


class TextChunker:
    """Split document text into bounded, slightly overlapping chunks."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document: Document) -> list[Chunk]:
        """Split one document, preferring whitespace boundaries when possible."""
        text = document.content.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + self.chunk_size // 2, end)
                if boundary > start:
                    end = boundary

            content = text[start:end].strip()
            if content:
                chunk_id = str(uuid5(NAMESPACE_URL, f"{document.id}:{chunk_index}:{content}"))
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document.id,
                        title=document.title,
                        content=content,
                        chunk_index=chunk_index,
                        source=document.source,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split a collection of documents into chunks."""
        return [chunk for document in documents for chunk in self.split_document(document)]
