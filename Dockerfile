FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the package and its runtime dependencies.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Run as a non-root user inside the application container.
RUN useradd --create-home --uid 10001 appuser
USER appuser

ENTRYPOINT ["python", "-m", "rag_pipeline"]
CMD ["--help"]
