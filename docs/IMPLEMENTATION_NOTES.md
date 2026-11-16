# Implementation notes

## Scope

This repository implements the Basic RAG assessment. It intentionally does not add BM25, hybrid search, Reciprocal Rank Fusion, query rewriting, or cross-encoder reranking. Those belong to a later Advanced RAG exercise.

## Main design choices

- **Gemini Developer API:** `gemini-embedding-001` creates document/query vectors; a configurable Gemini text model generates answers.
- **Qdrant:** stores chunk vectors and the source text/metadata in payloads. Cosine similarity is used for semantic Top-K retrieval.
- **Chunking:** a lightweight character-based splitter with overlap. It keeps the implementation understandable and avoids an extra text-splitting dependency.
- **Full-snapshot ingestion:** every `ingest` command recreates the configured collection before writing the current knowledge base. This prevents old chunks from remaining after documents are removed or changed. It is appropriate for this single-dataset assessment, not for a shared multi-tenant production database.
- **Abstention:** when Qdrant returns no results above the configured score threshold, the application returns a fixed insufficient-context response and skips the generation call.
- **Testability:** the pipeline depends on small protocols. Unit tests use fakes, so CI does not need a Gemini key or a running Qdrant instance.

## Assumptions

1. The input file is a JSON array. Each item has string `id`, `title`, and `content` values; `source` is optional.
2. All documents in one ingestion run belong to the same logical knowledge base and embedding model/dimension.
3. The local Qdrant service is reachable at `http://localhost:6333`; Docker Compose uses the service hostname `qdrant` internally.
4. The API key has access to the configured embedding and generation models.
5. The cosine score threshold is dataset-dependent and should be adjusted after checking retrieval results.

## Known limitations

- Character-based chunking is simple; it does not preserve semantic sections or token boundaries.
- The cosine threshold is a heuristic, not a calibrated confidence score.
- The prompt asks the model to cite source labels, but generated citations are not independently verified. The CLI separately prints the actual retrieved source records.
- Re-ingestion deletes and recreates one collection. Do not point this configuration at a shared collection containing unrelated data.
- The implementation has no user authentication, multi-tenancy, conversation memory, streaming responses, or web interface because they are outside the assessment requirements.
- Unit tests mock the external services. A live end-to-end run still needs a valid Gemini API key and a running Qdrant server.

## Challenges and resolutions to mention in the submission

- **Embedding dimension consistency:** the same configured dimension is used for Gemini output and Qdrant collection creation. Changing the dimension requires re-ingestion.
- **Stale chunks after data edits:** ingestion recreates the collection so the indexed data matches the current JSON file.
- **Unsupported questions:** a score threshold filters weak matches; empty retrieval causes abstention before calling the LLM.
- **Secret management in CI:** API credentials are not required for unit tests and are not stored in the repository. Real credentials remain in a local `.env` file or deployment secret store.
