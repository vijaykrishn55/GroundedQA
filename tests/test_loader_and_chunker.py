import json

import pytest

from rag_pipeline.chunker import TextChunker
from rag_pipeline.errors import DocumentLoadError
from rag_pipeline.loader import load_documents
from rag_pipeline.schemas import Document


def test_loader_reads_valid_json_documents(tmp_path):
    path = tmp_path / "knowledge.json"
    path.write_text(
        json.dumps([{"id": "kb-1", "title": "Setup", "content": "Install the service."}]),
        encoding="utf-8",
    )

    documents = load_documents(path)

    assert len(documents) == 1
    assert documents[0].id == "kb-1"
    assert documents[0].source == "knowledge.json"


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "knowledge.json"
    duplicate_docs = [
        {"id": "same", "title": "First", "content": "First body."},
        {"id": "same", "title": "Second", "content": "Second body."},
    ]
    path.write_text(json.dumps(duplicate_docs), encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Duplicate document ID"):
        load_documents(path)


def test_chunker_creates_overlapping_bounded_chunks():
    document = Document(
        id="long-doc",
        title="Long document",
        content="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        source="sample.json",
    )
    chunker = TextChunker(chunk_size=24, chunk_overlap=6)

    chunks = chunker.split_document(document)

    assert len(chunks) > 1
    assert all(0 < len(chunk.content) <= 24 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.id for chunk in chunks}) == len(chunks)
