# Loom / YouTube demonstration outline

Target length: 3-5 minutes.

1. **Introduce the project (20-30 seconds).** Explain that it is a basic end-to-end RAG pipeline using Gemini for embeddings and generation and Qdrant for vector storage.
2. **Show the architecture (30 seconds).** Walk through JSON loader → chunking → Gemini document embeddings → Qdrant → query embedding → Top-K retrieval → prompt construction → Gemini answer.
3. **Show ingestion (40 seconds).** Run `python -m rag_pipeline ingest --file data/knowledge_base.json`. Point out the number of source documents and chunks.
4. **Show a successful answer (45-60 seconds).** Ask `What specific resolution step fixes error code ERR-9021-X?`. Show the generated answer and retrieved source IDs/scores.
5. **Show safe empty retrieval (30-40 seconds).** Ask an unrelated question and show the insufficient-context response when no source clears the configured score threshold.
6. **Show tests and GitHub Actions (30-40 seconds).** Run `pytest -q`, then show the CI workflow after pushing the repository.
7. **Show Docker (40-60 seconds).** Run the same ingest and ask commands with `docker compose run --rm rag-app ...`; show Qdrant running as a separate container.
8. **Close (15 seconds).** Mention assumptions, the character-based chunking limitation, configurable score threshold, and the fact that CI uses fakes rather than paid API calls.

Before recording, ensure the `.env` file is not visible and never show the API key in the terminal or editor.
