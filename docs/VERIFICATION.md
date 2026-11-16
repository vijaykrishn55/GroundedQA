# Verification record

## Automated checks

Run:

```bash
ruff check src tests
ruff format --check src tests
pytest -q
```

The unit suite covers:

| Area | Check |
|---|---|
| Document loading | Valid JSON loads; duplicate IDs are rejected |
| Chunking | Produces bounded, indexed chunks |
| Ingestion | Chunks are embedded and written after collection reset |
| Retrieval + generation | Top-K is passed through, context is placed in the prompt, and sources are returned |
| Empty retrieval | A safe fallback is returned without calling Gemini generation |

## Live integration checklist

- [ ] Start Qdrant and confirm `http://localhost:6333/dashboard` is accessible.
- [ ] Add a valid `GEMINI_API_KEY` to `.env`.
- [ ] Run the ingestion command and confirm the expected document/chunk count.
- [ ] Ask: `What specific resolution step fixes error code ERR-9021-X?`
- [ ] Confirm the answer mentions `MAX_CONN_POOL=500` and a cluster deployment restart, with `doc_001` among the retrieved sources.
- [ ] Ask an unrelated question, such as `What is the population of Mars?`, and confirm the application abstains when no relevant source clears the configured score threshold.
- [ ] Repeat the same flow with Docker Compose.
- [ ] Capture terminal output and the Qdrant dashboard for the demonstration.

## Results to record before submission

| Check | Result | Evidence |
|---|---|---|
| Unit tests | Pending local run | Terminal screenshot / CI run |
| Lint and formatting | Pending local run | Terminal screenshot / CI run |
| Live ingestion | Pending API key + Qdrant | Terminal screenshot |
| Live retrieval and generation | Pending API key + Qdrant | Terminal screenshot |
| Docker execution | Pending Docker run | Terminal screenshot |
| GitHub Actions | Pending push to GitHub | Actions run URL |

Do not mark the live checks as passed until you have run them on your machine.
