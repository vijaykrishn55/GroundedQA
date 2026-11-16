from rag_pipeline.chunker import TextChunker
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.prompt import INSUFFICIENT_CONTEXT_ANSWER
from rag_pipeline.schemas import Document, RetrievedChunk


class FakeGemini:
    def __init__(self):
        self.generated_prompt = None
        self.generation_calls = 0

    def embed_documents(self, texts):
        return [[1.0, float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0, 0.0]

    def generate(self, prompt):
        self.generation_calls += 1
        self.generated_prompt = prompt
        return "Set MAX_CONN_POOL=500 and restart the deployment node. [Source 1]"


class FakeVectorStore:
    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self.chunks = []
        self.vectors = []
        self.recreated = False
        self.last_top_k = None
        self.last_min_score = None

    def recreate_collection(self):
        self.recreated = True
        self.chunks = []
        self.vectors = []

    def upsert(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vector, top_k, min_score):
        self.last_top_k = top_k
        self.last_min_score = min_score
        return self.search_results[:top_k]


def make_pipeline(search_results=None):
    provider = FakeGemini()
    store = FakeVectorStore(search_results)
    pipeline = RAGPipeline(
        embedding_provider=provider,
        vector_store=store,
        answer_generator=provider,
        chunker=TextChunker(chunk_size=80, chunk_overlap=10),
        top_k=3,
        min_relevance_score=0.25,
    )
    return pipeline, provider, store


def test_ingestion_embeds_chunks_and_replaces_vector_collection():
    pipeline, _, store = make_pipeline()
    documents = [
        Document(
            id="err-1",
            title="Database Error Codes",
            content="ERR-9021-X happens when the connection pool is exhausted. "
            "Set MAX_CONN_POOL=500 and restart the deployment node.",
            source="knowledge_base.json",
        )
    ]

    chunk_count = pipeline.ingest_documents(documents)

    assert store.recreated is True
    assert chunk_count == len(store.chunks)
    assert len(store.vectors) == chunk_count
    assert all(len(vector) == 2 for vector in store.vectors)


def test_ask_retrieves_top_k_builds_prompt_and_returns_sources():
    retrieved = [
        RetrievedChunk(
            id="chunk-1",
            document_id="err-1",
            title="Database Error Codes",
            content="Set MAX_CONN_POOL=500 and restart the deployment node.",
            chunk_index=0,
            source="knowledge_base.json",
            score=0.84,
        )
    ]
    pipeline, provider, store = make_pipeline(retrieved)

    response = pipeline.ask("How do I fix ERR-9021-X?", top_k=1)

    assert response.used_context is True
    assert "MAX_CONN_POOL=500" in response.answer
    assert response.sources[0].document_id == "err-1"
    assert store.last_top_k == 1
    assert store.last_min_score == 0.25
    assert "[Source 1]" in provider.generated_prompt
    assert provider.generation_calls == 1


def test_empty_retrieval_abstains_without_calling_generator():
    pipeline, provider, store = make_pipeline(search_results=[])

    response = pipeline.ask("What is the capital of Mars?")

    assert response.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert response.sources == ()
    assert response.used_context is False
    assert provider.generation_calls == 0
    assert store.last_top_k == 3
