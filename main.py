"""Convenience script for environments that prefer ``python main.py``."""

from rag_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
