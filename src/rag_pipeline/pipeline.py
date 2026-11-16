"""End-to-end RAG orchestration."""

from rag_pipeline.chunker import TextChunker
from rag_pipeline.interfaces import AnswerGenerator, EmbeddingProvider, VectorStore
from rag_pipeline.prompt import INSUFFICIENT_CONTEXT_ANSWER, PromptBuilder
from rag_pipeline.schemas import Document, RAGResponse, SourceReference


class RAGPipeline:
    """Coordinate ingestion, embedding, retrieval, prompt construction, and generation."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator,
        chunker: TextChunker,
        top_k: int = 4,
        min_relevance_score: float = 0.0,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not 0.0 <= min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between 0 and 1.")
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.answer_generator = answer_generator
        self.chunker = chunker
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        self.prompt_builder = prompt_builder or PromptBuilder()

    def ingest_documents(self, documents: list[Document]) -> int:
        """Chunk, embed, and replace the indexed knowledge base. Return chunk count."""
        if not documents:
            raise ValueError("At least one document is required for ingestion.")

        chunks = self.chunker.split_documents(documents)
        if not chunks:
            raise ValueError("No non-empty text chunks were produced from the documents.")

        vectors = self.embedding_provider.embed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("The embedding provider returned an unexpected number of vectors.")

        # This assessment treats ingestion as a full knowledge-base snapshot. Recreating the
        # collection prevents stale chunks from older versions of the JSON file from remaining.
        self.vector_store.recreate_collection()
        self.vector_store.upsert(chunks, vectors)
        return len(chunks)

    def ask(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Retrieve context and generate an answer, or abstain when retrieval is empty."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        requested_top_k = top_k if top_k is not None else self.top_k
        if requested_top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_vector = self.embedding_provider.embed_query(question)
        matches = self.vector_store.search(
            query_vector=query_vector,
            top_k=requested_top_k,
            min_score=self.min_relevance_score,
        )
        if not matches:
            return RAGResponse(
                query=question,
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                sources=(),
                used_context=False,
            )

        prompt = self.prompt_builder.build(question, matches)
        answer = self.answer_generator.generate(prompt).strip()
        if not answer:
            answer = INSUFFICIENT_CONTEXT_ANSWER

        sources = tuple(
            SourceReference(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                title=chunk.title,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
            )
            for chunk in matches
        )
        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            used_context=True,
        )
