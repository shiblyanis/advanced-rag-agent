import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.schemas import ChatRequest, ChatResponse, MemoryCreate
from app.ingestion import read_upload, make_parent_child_chunks
from app.rag import RAGService
from app.memory import MemoryService

app = FastAPI(title=settings.app_name, version="1.0.0", description="Hybrid RAG with reranking, context compression and semantic memory")
app.mount("/static", StaticFiles(directory="static"), name="static")
rag = RAGService()
memory = MemoryService()
telemetry = {"requests": 0, "total_latency_ms": 0.0}

@app.get("/", response_class=HTMLResponse)
def home():
    return Path("static/index.html").read_text(encoding="utf-8")

@app.get("/health")
def health():
    return {"status": "ok", "cohere_configured": bool(settings.cohere_api_key), "mem0_enabled": bool(memory.client), "vector_store_configured": bool(settings.qdrant_url)}

@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    try:
        content = await file.read()
        pages = read_upload(file.filename or "upload.txt", content)
        parents, children = make_parent_child_chunks(pages, file.filename or "upload")
        # Child vectors are indexed; parent text is retained in payload for expanded context.
        parent_map = {p["id"]: p for p in parents}
        records = [{**c, "parent_text": parent_map[c["parent_id"]]["text"]} for c in children]
        count = rag.ingest(records)
        return {"filename": file.filename, "parents": len(parents), "children_indexed": count}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    started = time.perf_counter()
    try:
        user_memory = memory.search(body.user_id, body.message)
        hits = rag.retrieve(body.message)
        # Context compression: keep only the retrieved parent passages and cap total tokens/chars.
        context_parts, sources = [], []
        for i, hit in enumerate(hits, 1):
            passage = hit.get("parent_text") or hit["text"]
            context_parts.append(f"[{i}] {passage}")
            sources.append({"filename": hit.get("filename", "unknown"), "page": hit.get("page"), "score": hit.get("rerank_score"), "text": hit["text"][:700]})
        context = "\n\n".join(context_parts)[:settings.max_context_chars]
        answer = rag.answer(body.message, context, user_memory)
        elapsed = (time.perf_counter() - started) * 1000
        telemetry["requests"] += 1
        telemetry["total_latency_ms"] += elapsed
        return ChatResponse(answer=answer, sources=sources, latency_ms=round(elapsed, 2), retrieved=settings.retrieval_k if hits else 0, reranked=len(hits), memory_used=bool(user_memory))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/api/memories")
def create_memory(body: MemoryCreate):
    try:
        result = memory.add(body.user_id, body.text)
        return {"status": "saved", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/api/memories/{user_id}")
def list_memories(user_id: str):
    return {"user_id": user_id, "memories": memory.get_all(user_id)}

@app.get("/api/metrics")
def metrics():
    count = telemetry["requests"]
    return {"requests": count, "average_latency_ms": round(telemetry["total_latency_ms"] / count, 2) if count else 0}
