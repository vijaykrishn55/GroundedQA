
"""Command-line entry point for the assessment pipeline."""

import argparse
import logging
from pathlib import Path

from rag_pipeline.chunker import TextChunker
from rag_pipeline.config import Settings
from rag_pipeline.embeddings import FastEmbedProvider
from rag_pipeline.errors import RAGError
from rag_pipeline.groq import GroqProvider
from rag_pipeline.loader import load_documents
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.vector_store import QdrantVectorStore

logger = logging.getLogger("rag_pipeline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="basic-rag",
        description="A basic Groq + FastEmbed + Qdrant RAG pipeline.",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ingest_parser = commands.add_parser(
        "ingest",
        help="Load, chunk, embed, and index the knowledge base.",
    )

    ingest_parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/knowledge_base.json"),
    )

    ask_parser = commands.add_parser(
        "ask",
        help="Ask a question against the indexed knowledge base.",
    )

    ask_parser.add_argument(
        "question",
        help="Question to answer using retrieved context.",
    )

    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
    )

    return parser


def _build_pipeline(settings: Settings):
    """Initialize providers, vector storage, and pipeline."""

    embedder = FastEmbedProvider(
        model_name=settings.embedding_model,
    )

    generator = GroqProvider(
        api_key=settings.groq_api_key,
        model=settings.generation_model,
    )

    try:
        store = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
            vector_size=settings.embedding_dimension,
        )

    except Exception:
        generator.close()
        raise

    pipeline = RAGPipeline(
        embedding_provider=embedder,
        vector_store=store,
        answer_generator=generator,
        chunker=TextChunker(
            settings.chunk_size,
            settings.chunk_overlap,
        ),
        top_k=settings.top_k,
        min_relevance_score=settings.min_relevance_score,
    )

    return pipeline, embedder, generator, store


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = build_parser().parse_args(argv)

    generator = None
    store = None

    try:
        settings = Settings.from_env()

        pipeline, embedder, generator, store = (
            _build_pipeline(settings)
        )

        if args.command == "ingest":
            documents = load_documents(args.file)

            chunk_count = pipeline.ingest_documents(documents)

            print(
                f"Ingestion complete: {len(documents)} documents "
                f"became {chunk_count} chunks.\n"
                f"Qdrant collection: {settings.qdrant_collection}"
            )

            return 0

        if args.command == "ask":
            response = pipeline.ask(
                args.question,
                top_k=args.top_k,
            )

            print(f"\nQuestion: {response.query}\n")
            print(f"Answer:\n{response.answer}\n")

            if response.sources:
                print("Retrieved sources:")

                for index, source in enumerate(
                    response.sources,
                    start=1,
                ):
                    print(
                        f"  [{index}] {source.title} "
                        f"| document={source.document_id} "
                        f"| chunk={source.chunk_index} "
                        f"| score={source.score:.4f}"
                    )
            else:
                print("Retrieved sources: none")

            return 0

    except RAGError as exc:
        logger.error("%s", exc)
        return 1

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130

    finally:
        if store is not None:
            store.close()

        if generator is not None:
            generator.close()

    return 0
