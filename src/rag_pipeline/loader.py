"""Load and validate a structured JSON knowledge base."""

import json
from pathlib import Path
from typing import Any

from rag_pipeline.errors import DocumentLoadError
from rag_pipeline.schemas import Document


_REQUIRED_FIELDS = ("id", "title", "content")


def load_documents(file_path: str | Path) -> list[Document]:
    """Load a JSON array of documents with unique IDs and required text fields."""
    path = Path(file_path)
    try:
        raw_data: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DocumentLoadError(f"Knowledge-base file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DocumentLoadError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise DocumentLoadError(f"Unable to read {path}: {exc}") from exc

    if not isinstance(raw_data, list):
        raise DocumentLoadError("The knowledge base must be a JSON array of document objects.")
    if not raw_data:
        raise DocumentLoadError("The knowledge base is empty; add at least one document.")

    documents: list[Document] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise DocumentLoadError(f"Document at index {index} must be a JSON object.")

        missing = [field for field in _REQUIRED_FIELDS if not isinstance(item.get(field), str)]
        if missing:
            raise DocumentLoadError(
                f"Document at index {index} must contain string fields: {', '.join(missing)}."
            )

        document_id = item["id"].strip()
        title = item["title"].strip()
        content = item["content"].strip()
        if not document_id or not title or not content:
            raise DocumentLoadError(
                f"Document at index {index} has an empty id, title, or content field."
            )
        if document_id in seen_ids:
            raise DocumentLoadError(f"Duplicate document ID found: {document_id}")
        seen_ids.add(document_id)

        source_value = item.get("source", path.name)
        source = source_value.strip() if isinstance(source_value, str) else path.name
        documents.append(
            Document(id=document_id, title=title, content=content, source=source or path.name)
        )

    return documents
