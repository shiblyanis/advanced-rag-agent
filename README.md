# Advanced RAG Agent with Context Compression & Semantic Memory

A deployable FastAPI + Cohere + Qdrant RAG application with PDF/TXT/Markdown ingestion, parent-child chunking, dense retrieval with lexical reciprocal-rank fusion, Cohere reranking, bounded context compression, optional Mem0 semantic memory, source citations, and a browser chat UI.

## Architecture

1. Uploads are parsed into page-aware text.
2. Parent passages (~1000 characters) are split into child retrieval chunks (~200 characters).
3. Cohere `embed-english-v3.0` embeds child chunks; Qdrant stores dense vectors and payload metadata.
4. Query embedding retrieves up to 20 candidates. A lexical ranking is fused with dense ranking using RRF.
5. Cohere `rerank-v3.5` reranks candidates and returns up to 5.
6. Retrieved child chunks expand to their parent passages. The prompt compressor bounds the final context to `MAX_CONTEXT_CHARS` and removes unrelated candidates through reranking.
7. Optional Mem0 retrieves persistent user memories and injects them only as personalization context.
8. Cohere Command generates a grounded answer with source markers.

> Note: This implementation's sparse leg is a lexical overlap rank over the dense candidate pool, fused with dense rank using RRF. For a full independent sparse index, add Qdrant's sparse vectors/BM25 sidecar and fuse both independent result sets.

## Quick start (Docker)

```bash
cp .env.example .env
# Add COHERE_API_KEY and hosted Qdrant credentials to .env
docker compose up --build
```

Open http://localhost:8000. API docs: http://localhost:8000/docs. Health: http://localhost:8000/health.

For local Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Environment variables

See `.env.example`. Use a Qdrant Cloud cluster for Render deployment; do not point Render at `localhost` for Qdrant. Enable Mem0 by setting `MEM0_ENABLED=true` and `MEM0_API_KEY`.

## API

- `GET /health`
- `POST /api/ingest` multipart form field `file` (PDF/TXT/MD)
- `POST /api/chat` JSON: `{"message":"...","user_id":"...","session_id":"..."}`
- `POST /api/memories` JSON: `{"user_id":"...","text":"..."}`
- `GET /api/memories/{user_id}`
- `GET /api/metrics`

Postman collection: `postman/Advanced-RAG-Agent.postman_collection.json`.

## Tests

```bash
pytest -q
```

## Render deployment

1. Push this repository to GitHub.
2. Create a Qdrant Cloud cluster and copy its HTTPS URL/API key.
3. In Render, create a new **Web Service** from the GitHub repository; select Docker.
4. Add `COHERE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `MEM0_API_KEY`; set `MEM0_ENABLED=true`.
5. Set health check path to `/health`; deploy. Render injects `PORT`; Docker command binds to it.
6. Verify `/health`, upload a small document, and ask a question.

The Render blueprint is included as `render.yaml`. Do not commit `.env` or API keys.

## Limitations / production hardening

- Add authentication and per-user document namespaces before exposing publicly.
- Add upload size limits, virus scanning, OCR for scanned PDFs, deletion endpoints, rate limits, and request tracing.
- In-memory telemetry resets on restart; use Prometheus/OpenTelemetry for production.
- Mem0 integration is optional; without credentials the application remains usable but memory endpoints return a configuration error.
- The current `session_id` is accepted for client compatibility; persistent chat transcript storage is a next hardening step.
